from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "experiments" / "vast_reproduction" / "measure.py"
SPEC = importlib.util.spec_from_file_location("vast_measure", SCRIPT)
measure = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(measure)


def test_pair_summary_reports_maximum_cosine_distance():
    rows = np.asarray([[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]])
    pairs = measure.pair_rows(["a", "b", "c"], rows)
    summary = measure.summarize(pairs)

    assert summary["pair_count"] == 3
    assert summary["min"] == 0.0
    assert summary["maximum_cosine_distance"] == 1.0
    assert summary["most_distant_pair"]["a"] == "a"
    assert summary["most_distant_pair"]["b"] == "b"


def test_prompt_centroids_average_repeated_seeds_before_comparison():
    records = [
        {"index": 0, "id": "first"},
        {"index": 1, "id": "first"},
        {"index": 2, "id": "second"},
    ]
    embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    names, centroids = measure.prompt_centroids(records, embeddings)

    assert names == ["first", "second"]
    np.testing.assert_allclose(centroids, np.eye(2))


def test_farthest_selection_minimizes_nearest_similarity():
    rows = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.1],
            [0.0, 1.0],
        ]
    )
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)

    assert set(measure.farthest_indices(rows, 2)) == {0, 2}
