import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


def load_script(name: str, filename: str):
    insightface = sys.modules.setdefault("insightface", types.ModuleType("insightface"))
    app = sys.modules.setdefault("insightface.app", types.ModuleType("insightface.app"))
    app.FaceAnalysis = object
    insightface.app = app
    if "cv2" not in sys.modules:
        cv2 = types.ModuleType("cv2")
        cv2.BORDER_REPLICATE = 1
        sys.modules["cv2"] = cv2
    if filename == "05_render_instantid.py":
        torch = sys.modules.setdefault("torch", types.ModuleType("torch"))
        torch.float16 = object()
        diffusers = sys.modules.setdefault("diffusers", types.ModuleType("diffusers"))
        diffusers.ControlNetModel = object
    script = Path(__file__).parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


embed = load_script("embed_faces", "02_embed_faces.py")
rank = load_script("rank_roster", "06_rank_roster.py")
render = load_script("render_instantid", "05_render_instantid.py")


class FakeAnalyzer:
    def get(self, image):
        assert image.flags.c_contiguous
        return [
            SimpleNamespace(
                bbox=np.array([0, 0, 10, 10]),
                det_score=0.95,
                embedding=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            )
        ]


def test_embedding_batch_is_contiguous_checkpointed_and_error_tolerant(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (16, 16), "red").save(image_dir / "good.png")
    (image_dir / "corrupt.png").write_text("not an image")
    output = tmp_path / "embeddings.npz"
    embed.error_path(output).write_text(json.dumps({"stale.png": "old failure"}))

    monkeypatch.setattr(embed, "build_analyzer", lambda *_args, **_kwargs: FakeAnalyzer())
    monkeypatch.setattr(
        sys,
        "argv",
        [str(Path(embed.__file__)), str(image_dir), str(output), "--checkpoint-every", "1"],
    )
    embed.main()

    with np.load(output) as data:
        assert data["files"].tolist() == ["good.png"]
        assert data["emb_buf"].shape == (1, 4)
        assert data["emb_ant"].shape == (1, 4)
    errors = json.loads(embed.error_path(output).read_text())
    assert list(errors) == ["corrupt.png"]


def test_ranker_emits_the_advertised_report_schema(monkeypatch, tmp_path):
    candidates = tmp_path / "candidates.npz"
    image_dir = tmp_path / "renders"
    output_dir = tmp_path / "ranked"
    image_dir.mkdir()
    np.savez(
        candidates,
        names=np.array(["keeper", "sampled", "missing"]),
        origins=np.array(["realvis", "sampled", "zimage"]),
    )

    class FakeFaceAnalysis:
        def __init__(self, **_kwargs):
            pass

        def prepare(self, **_kwargs):
            pass

    vectors = {
        "keeper": np.array([1.0, 0.0, 0.0]),
        "sampled": np.array([0.0, 1.0, 0.0]),
    }

    def fake_embedding(render_path, _analyzer):
        identity = render_path.stem.rsplit("_", 1)[0]
        return vectors.get(identity)

    monkeypatch.setattr(rank, "FaceAnalysis", FakeFaceAnalysis)
    monkeypatch.setattr(rank, "embedding_with_retry", fake_embedding)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path(rank.__file__)),
            str(candidates),
            str(image_dir),
            str(output_dir),
            "--top-k",
            "2",
        ],
    )

    rank.main()

    report = json.loads((output_dir / "report.json").read_text())
    assert report["schema_version"] == "different-pipeline-generator/v1"
    assert report["top_k"] == 2
    assert report["origin_stats_top50"] == {"realvis": 1, "sampled": 1}
    assert report["origin_totals"] == {"realvis": 1, "sampled": 1, "zimage": 1}
    assert report["rejected_for_missing_faces"] == ["missing"]
    assert len(report["top50"]) == 2
    assert {row["pid"] for row in report["all"]} == {"keeper", "sampled"}


def test_missing_render_returns_none_before_calling_detector(tmp_path):
    class DetectorThatMustNotRun:
        def get(self, _image):
            raise AssertionError("detector should not receive a missing image")

    assert rank.embedding_with_retry(tmp_path / "missing.png", DetectorThatMustNotRun()) is None


def test_unreadable_keypoint_image_is_rejected_before_detector():
    class DetectorThatMustNotRun:
        def get(self, _image):
            raise AssertionError("detector should not receive an unreadable image")

    assert render.faces_with_retry(None, DetectorThatMustNotRun()) == []
