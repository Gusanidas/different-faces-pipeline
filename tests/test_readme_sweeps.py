import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "experiments" / "readme_sweeps" / "generate.py"
MEASURE_SCRIPT = ROOT / "experiments" / "readme_sweeps" / "measure.py"
MANIFEST = ROOT / "experiments" / "readme_sweeps" / "prompts.json"
SPEC = importlib.util.spec_from_file_location("readme_sweep_generate", SCRIPT)
generate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generate)
MEASURE_SPEC = importlib.util.spec_from_file_location("readme_sweep_measure", MEASURE_SCRIPT)
measure = importlib.util.module_from_spec(MEASURE_SPEC)
assert MEASURE_SPEC.loader is not None
MEASURE_SPEC.loader.exec_module(measure)


def test_readme_sweep_manifest_builds_expected_balanced_jobs():
    jobs = generate.build_jobs(json.loads(MANIFEST.read_text()))

    assert len(jobs) == 144
    assert Counter(job["set"] for job in jobs) == {
        "seed_sweeps": 72,
        "scandinavian_features": 48,
        "lens_framing": 24,
    }
    assert Counter(job["engine"] for job in jobs) == {"realvis": 72, "zimage": 72}
    assert len({(job["set"], job["engine"], job["id"], job["seed"]) for job in jobs}) == 144


def test_seed_sweeps_use_twelve_contiguous_shared_seeds():
    config = json.loads(MANIFEST.read_text())
    seeds = config["seed_sweeps"]["seeds"]

    assert seeds == list(range(seeds[0], seeds[0] + 12))
    jobs = generate.build_jobs(config)
    for prompt in config["seed_sweeps"]["prompts"]:
        for engine in generate.ENGINES:
            used = [
                job["seed"]
                for job in jobs
                if job["set"] == "seed_sweeps"
                and job["id"] == prompt["id"]
                and job["engine"] == engine
            ]
            assert used == seeds


def test_seed_prompts_request_clothing():
    config = json.loads(MANIFEST.read_text())

    for prompt in config["seed_sweeps"]["prompts"]:
        assert "wearing" in prompt["text"].lower()
    assert "shirtless" in config["negative"]


def test_face_detection_retries_close_crops_on_a_padded_canvas():
    class Face:
        bbox = np.asarray([1, 2, 11, 14])

    class Analyzer:
        def __init__(self):
            self.shapes = []

        def get(self, image):
            self.shapes.append(image.shape)
            return [] if len(self.shapes) == 1 else [Face()]

    analyzer = Analyzer()
    face, mode = measure.detect_largest(analyzer, np.zeros((20, 30, 3), dtype=np.uint8))

    assert face is not None
    assert mode == "padded"
    assert analyzer.shapes == [(20, 30, 3), (40, 60, 3)]
