from pathlib import Path

import numpy as np
import pytest

from different_pipeline_generator.measurements import (
    ensure_unique_renders,
    pair_rows,
    parse_render,
    summarize_pairs,
)


def test_duplicate_comfy_counters_are_rejected():
    root = Path("/renders")
    records = [
        parse_render(root / "seed_sweeps/realvis/a_s10_00001_.png", root),
        parse_render(root / "seed_sweeps/realvis/a_s10_00002_.png", root),
    ]

    with pytest.raises(ValueError, match="duplicate logical render"):
        ensure_unique_renders([record for record in records if record is not None])


def test_shared_pair_summary_handles_empty_and_regular_inputs():
    assert summarize_pairs([]) == {"pair_count": 0}

    rows = np.eye(2)
    pairs = pair_rows(["a", "b"], rows)
    summary = summarize_pairs(pairs, include_p05=True)

    assert summary["pair_count"] == 1
    assert summary["maximum_cosine_distance"] == 1.0
    assert summary["p05"] == 0.0
