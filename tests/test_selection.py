import numpy as np

from different_pipeline_generator.selection import (
    farthest_point_indices,
    tournament_indices,
)


def test_farthest_point_selection_spreads_around_circle():
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    chosen = farthest_point_indices(points, 4, seed_index=0)
    assert chosen.tolist() == [0, 4, 2, 6]


def test_tournament_drops_the_less_consistent_twin():
    centroids = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.999, 0.04, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    intra = np.array([0.90, 0.70, 0.85, 0.86])
    kept = tournament_indices(centroids, intra, 3)
    assert kept.tolist() == [0, 2, 3]

