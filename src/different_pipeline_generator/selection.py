"""Selection algorithms shared by the CPU and GPU stages."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def unit_rows(values: ArrayLike) -> NDArray[np.float64]:
    """Return a two-dimensional array with every row L2-normalized."""
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 2 or len(rows) == 0:
        raise ValueError("expected a non-empty 2D embedding array")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("embeddings must have non-zero norm")
    return rows / norms


def farthest_point_indices(
    embeddings: ArrayLike,
    k: int,
    *,
    seed_index: int | None = None,
) -> NDArray[np.int64]:
    """Greedily choose ``k`` rows that maximize their minimum cosine distance.

    When no seed is supplied, selection starts at the face furthest from the
    normalized cloud centroid. The implementation is deterministic: ties keep
    the lowest input index.
    """
    rows = unit_rows(embeddings)
    n = len(rows)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}, got {k}")
    if seed_index is None:
        centroid = rows.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm <= 1e-12:
            seed_index = 0
        else:
            seed_index = int(np.argmin(rows @ (centroid / norm)))
    if not 0 <= seed_index < n:
        raise ValueError(f"seed_index must be between 0 and {n - 1}")

    selected = [seed_index]
    used = np.zeros(n, dtype=bool)
    used[seed_index] = True
    min_distance = 1.0 - rows @ rows[seed_index]

    while len(selected) < k:
        candidates = np.where(used, -np.inf, min_distance)
        next_index = int(np.argmax(candidates))
        selected.append(next_index)
        used[next_index] = True
        min_distance = np.minimum(min_distance, 1.0 - rows @ rows[next_index])

    return np.asarray(selected, dtype=np.int64)


def tournament_indices(
    centroids: ArrayLike,
    intra_consistency: ArrayLike,
    k: int,
) -> NDArray[np.int64]:
    """Keep identities that are both stable and unlike everyone else.

    The weakest remaining identity is repeatedly removed using

    ``score = intra_consistency - max_cosine_similarity_to_any_survivor``.

    Recomputing the closest neighbor after every removal means one member of a
    near-twin pair can survive instead of penalizing both forever.
    """
    rows = unit_rows(centroids)
    intra = np.asarray(intra_consistency, dtype=np.float64)
    n = len(rows)
    if intra.shape != (n,):
        raise ValueError(f"intra_consistency must have shape ({n},)")
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}, got {k}")

    similarity = rows @ rows.T
    np.fill_diagonal(similarity, -np.inf)
    alive = list(range(n))
    while len(alive) > k:
        sub = similarity[np.ix_(alive, alive)]
        closest = sub.max(axis=1)
        scores = intra[alive] - closest
        del alive[int(np.argmin(scores))]
    return np.asarray(alive, dtype=np.int64)

