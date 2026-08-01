from inspect import signature

import numpy as np
import pytest

from different_pipeline_generator.sampling import fit_cloud, sample_cloud


def test_cloud_samples_are_normalized_and_pre_gated():
    rng = np.random.default_rng(10)
    raw = rng.normal(size=(60, 16))
    raw[:, 0] += 3.0
    model = fit_cloud(raw, gate_floor=0.97)
    result = sample_cloud(model, 8, gate=0.97, seed=11, max_attempts=5000)

    assert result.vectors.shape == (8, 16)
    np.testing.assert_allclose(np.linalg.norm(result.vectors, axis=1), 1.0)
    references = np.concatenate([model.keepers, result.vectors])
    similarity = references @ references.T
    np.fill_diagonal(similarity, -1)
    assert float(similarity.max()) <= 0.97 + 1e-9


def test_cloud_sampler_is_reproducible():
    rng = np.random.default_rng(20)
    model = fit_cloud(rng.normal(size=(50, 12)), gate_floor=0.98)
    first = sample_cloud(model, 5, gate=0.98, seed=42)
    second = sample_cloud(model, 5, gate=0.98, seed=42)
    np.testing.assert_allclose(first.vectors, second.vectors)


def test_zero_max_attempts_is_rejected():
    rng = np.random.default_rng(30)
    model = fit_cloud(rng.normal(size=(20, 8)), gate_floor=0.99)
    with pytest.raises(ValueError, match="max_attempts"):
        sample_cloud(model, 2, gate=0.99, max_attempts=0)


def test_sampler_does_not_advertise_the_historical_noop_temperature():
    assert "temperature" not in signature(sample_cloud).parameters
