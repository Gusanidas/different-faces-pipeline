import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "03b_curate_seed_faces.py"
SPEC = importlib.util.spec_from_file_location("curate_seed_faces", SCRIPT)
curate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(curate)


def test_selection_manifest_ignores_comments_and_resolves_unique_basenames(tmp_path):
    selection = tmp_path / "curated.txt"
    selection.write_text("# chosen by eye\nmodel_a/a.png\n\nb.png\n")

    entries = curate.read_selection(selection)
    indices = curate.resolve_selection(
        np.array(["model_a/a.png", "model_b/b.png", "model_c/c.png"]), entries
    )

    assert entries == ["model_a/a.png", "b.png"]
    assert indices.tolist() == [0, 1]


def test_selection_rejects_an_ambiguous_basename():
    with pytest.raises(ValueError, match="ambiguous"):
        curate.resolve_selection(np.array(["a/face.png", "b/face.png"]), ["face.png"])


def test_curated_payload_preserves_raw_pool_indices(tmp_path):
    archive = tmp_path / "shortlist.npz"
    np.savez(
        archive,
        files=np.array(["a.png", "b.png", "c.png"]),
        emb_buf=np.eye(3),
        emb_ant=np.eye(3),
        ant_norms=np.array([10.0, 11.0, 12.0]),
        detection_scores=np.array([0.8, 0.9, 0.95]),
        selected_indices=np.array([12, 25, 41]),
    )

    with np.load(archive) as data:
        payload = curate.curated_payload(data, np.array([2, 0]))

    assert payload["files"].tolist() == ["c.png", "a.png"]
    assert payload["selected_indices"].tolist() == [41, 12]
    assert payload["shortlist_indices"].tolist() == [2, 0]


def test_curated_payload_rejects_missing_raw_pool_provenance(tmp_path):
    archive = tmp_path / "shortlist.npz"
    np.savez(
        archive,
        files=np.array(["a.png", "b.png"]),
        emb_buf=np.eye(2),
        emb_ant=np.eye(2),
        ant_norms=np.array([10.0, 11.0]),
    )

    with np.load(archive) as data, pytest.raises(ValueError, match="raw-pool provenance"):
        curate.curated_payload(data, np.array([0]))
