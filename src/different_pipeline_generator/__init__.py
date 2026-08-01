"""Core geometry for the different-face pipeline."""

from .naming import clean_comfy_name, ensure_unique
from .sampling import CloudModel, SampleResult, fit_cloud, sample_cloud
from .selection import farthest_point_indices, tournament_indices

__all__ = [
    "CloudModel",
    "SampleResult",
    "clean_comfy_name",
    "ensure_unique",
    "farthest_point_indices",
    "fit_cloud",
    "sample_cloud",
    "tournament_indices",
]
