import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "03_select_seed_faces.py"
SPEC = importlib.util.spec_from_file_location("select_seed_faces", SCRIPT)
select = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(select)


def test_selected_payload_rejects_misaligned_arrays(tmp_path):
    archive = tmp_path / "bad.npz"
    np.savez(
        archive,
        files=np.array(["a.png", "b.png"]),
        emb_buf=np.ones((2, 3)),
        emb_ant=np.ones((2, 3)),
        ant_norms=np.ones(1),
    )
    with np.load(archive) as data:
        with pytest.raises(ValueError, match="ant_norms"):
            select.selected_payload(data, np.array([0]))


def test_selected_payload_ignores_scalar_metadata_explicitly(tmp_path):
    archive = tmp_path / "good.npz"
    np.savez(
        archive,
        source_root=np.asarray("/tmp/source"),
        files=np.array(["a.png", "b.png"]),
        emb_buf=np.eye(2),
        emb_ant=np.eye(2),
        ant_norms=np.ones(2),
    )
    with np.load(archive) as data:
        payload = select.selected_payload(data, np.array([1]))
    assert "source_root" not in payload
    assert payload["files"].tolist() == ["b.png"]


def test_selected_images_preserve_relative_paths_with_duplicate_basenames(tmp_path):
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "shortlist"
    (image_dir / "model_a").mkdir(parents=True)
    (image_dir / "model_b").mkdir(parents=True)
    (image_dir / "model_a" / "face.png").write_bytes(b"first")
    (image_dir / "model_b" / "face.png").write_bytes(b"second")

    select.copy_selected_images(
        image_dir,
        output_dir,
        np.array(["model_a/face.png", "model_b/face.png"]),
        np.array([0, 1]),
    )

    assert (output_dir / "model_a" / "face.png").read_bytes() == b"first"
    assert (output_dir / "model_b" / "face.png").read_bytes() == b"second"


def test_top_k_one_writes_outputs_and_exits_successfully(monkeypatch, tmp_path, capsys):
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "shortlist"
    archive = tmp_path / "embeddings.npz"
    image_dir.mkdir()
    (image_dir / "only.png").write_bytes(b"image")
    np.savez(
        archive,
        files=np.array(["only.png"]),
        emb_buf=np.array([[1.0, 0.0]], dtype=np.float32),
        emb_ant=np.array([[1.0, 0.0]], dtype=np.float32),
        ant_norms=np.ones(1, dtype=np.float32),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            str(archive),
            str(image_dir),
            str(output_dir),
            "--top-k",
            "1",
        ],
    )

    select.main()

    assert (output_dir / "only.png").is_file()
    assert (output_dir / "selected.npz").is_file()
    assert (output_dir / "shortlist.txt").read_text() == "only.png\n"
    assert "pairwise cosine unavailable" in capsys.readouterr().out
