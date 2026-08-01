"""Fit and sample a demographic identity cloud in a face-embedding space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .selection import unit_rows


@dataclass(frozen=True)
class CloudModel:
    """PCA model of what varies inside one tightly defined demographic."""

    keepers: NDArray[np.float64]
    mean: NDArray[np.float64]
    components: NDArray[np.float64]
    variances: NDArray[np.float64]
    radii: NDArray[np.float64]
    default_gate: float


@dataclass(frozen=True)
class SampleResult:
    vectors: NDArray[np.float64]
    attempts: int
    radius_low: float
    radius_high: float
    gate: float


def fit_cloud(embeddings: ArrayLike, *, gate_floor: float = 0.45) -> CloudModel:
    """Fit a covariance-shaped PCA cloud to normalized keeper embeddings."""
    keepers = unit_rows(embeddings)
    if len(keepers) < 2:
        raise ValueError("at least two keepers are required")
    mean = keepers.mean(axis=0)
    centered = keepers - mean
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    variances = singular_values**2 / (len(keepers) - 1)
    radii = np.linalg.norm(centered, axis=1)

    similarity = keepers @ keepers.T
    off_diagonal = similarity[~np.eye(len(keepers), dtype=bool)]
    default_gate = max(float(np.percentile(off_diagonal, 95)), gate_floor)
    return CloudModel(
        keepers=keepers,
        mean=mean,
        components=components,
        variances=variances,
        radii=radii,
        default_gate=default_gate,
    )


def sample_cloud(
    model: CloudModel,
    count: int,
    *,
    radius_percentiles: tuple[float, float] = (60.0, 99.0),
    gate: float | None = None,
    seed: int = 6001,
    max_attempts: int | None = None,
) -> SampleResult:
    """Sample separated identities on an empirical outer shell.

    The cloud mean carries shared traits. PCA deviations carry individuality.
    Sampling an outer shell avoids drifting back to the generic average face,
    and the cosine pre-gate rejects candidates too close to a keeper or sibling.
    """
    if count < 1:
        raise ValueError("count must be positive")
    low_p, high_p = radius_percentiles
    if not 0 <= low_p < high_p <= 100:
        raise ValueError("radius percentiles must satisfy 0 <= low < high <= 100")
    chosen_gate = model.default_gate if gate is None else float(gate)
    radius_low, radius_high = np.percentile(model.radii, [low_p, high_p])
    if max_attempts is not None and max_attempts < 1:
        raise ValueError("max_attempts must be positive when provided")
    attempt_limit = max(400, count * 50) if max_attempts is None else max_attempts
    rng = np.random.default_rng(seed)
    accepted: list[NDArray[np.float64]] = []

    attempts = 0
    while len(accepted) < count and attempts < attempt_limit:
        attempts += 1
        coefficients = (
            rng.standard_normal(len(model.variances))
            * np.sqrt(model.variances)
        )
        deviation = coefficients @ model.components
        deviation_norm = np.linalg.norm(deviation)
        if deviation_norm <= 1e-12:
            continue
        radius = rng.uniform(radius_low, radius_high)
        candidate = model.mean + deviation * (radius / deviation_norm)
        candidate_norm = np.linalg.norm(candidate)
        if candidate_norm <= 1e-12:
            continue
        candidate /= candidate_norm

        references = model.keepers
        if accepted:
            references = np.concatenate([references, np.stack(accepted)])
        if float((references @ candidate).max()) > chosen_gate:
            continue
        accepted.append(candidate.copy())

    if len(accepted) != count:
        raise RuntimeError(
            f"accepted {len(accepted)}/{count} identities in {attempts} attempts; "
            "raise the gate, widen the shell, or increase max_attempts"
        )
    return SampleResult(
        vectors=np.stack(accepted),
        attempts=attempts,
        radius_low=float(radius_low),
        radius_high=float(radius_high),
        gate=chosen_gate,
    )
