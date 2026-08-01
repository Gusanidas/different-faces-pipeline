import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "04_sample_identity_cloud.py"
SPEC = importlib.util.spec_from_file_location("sample_identity_cloud", SCRIPT)
sample = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(sample)


def test_cli_passes_max_attempts_to_sampler(monkeypatch, tmp_path):
    bank = tmp_path / "bank.npz"
    output = tmp_path / "candidates.npz"
    np.savez(
        bank,
        files=np.array(["a.png", "b.png"]),
        emb_ant=np.eye(2),
        ant_norms=np.ones(2),
    )
    captured = {}

    def fake_sample_cloud(_model, count, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            vectors=np.ones((count, 2)) / np.sqrt(2),
            attempts=7,
            gate=0.9,
            radius_low=0.1,
            radius_high=0.2,
        )

    monkeypatch.setattr(sample, "fit_cloud", lambda _keepers: object())
    monkeypatch.setattr(sample, "sample_cloud", fake_sample_cloud)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--bank",
            f"realvis={bank}",
            "--sampled",
            "1",
            "--max-attempts",
            "1234",
            "--output",
            str(output),
        ],
    )

    sample.main()

    assert captured["max_attempts"] == 1234
    assert output.is_file()
