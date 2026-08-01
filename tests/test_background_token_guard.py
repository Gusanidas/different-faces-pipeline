import importlib.util
import sys
from pathlib import Path

import pytest


EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments" / "background_pairs"


def load_script(name: str, filename: str):
    script = EXPERIMENT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


background_generate = load_script("background_generate", "generate.py")
previous_generate = sys.modules.get("generate")
sys.modules["generate"] = background_generate
try:
    token_guard = load_script("background_token_guard", "tokenize_prompts.py")
finally:
    if previous_generate is None:
        sys.modules.pop("generate", None)
    else:
        sys.modules["generate"] = previous_generate


class FakeTokenizer:
    def __init__(self, overflow_prefix: str | None = None):
        self.overflow_prefix = overflow_prefix

    def __call__(self, prompt: str, *, add_special_tokens: bool):
        assert add_special_tokens
        count = 78 if self.overflow_prefix and prompt.startswith(self.overflow_prefix) else 10
        return {"input_ids": list(range(count))}


def test_production_guard_measures_every_variation_in_both_demographics():
    tokenizers = {
        "sdxl_clip_l": FakeTokenizer(),
        "sdxl_clip_g": FakeTokenizer(),
    }

    summary = token_guard.production_prompt_summary(tokenizers)

    assert summary["all_fit_one_sdxl_window"]
    assert summary["demographics"]["es_m"]["count"] == 16 * 16 * 8 * 4
    assert summary["demographics"]["sc_f"]["count"] == 16 * 16 * 8 * 4


def test_production_guard_rejects_an_over_budget_scandinavian_prompt():
    tokenizers = {
        "sdxl_clip_l": FakeTokenizer("a Scandinavian woman"),
        "sdxl_clip_g": FakeTokenizer(),
    }
    summary = token_guard.production_prompt_summary(tokenizers)

    assert summary["demographics"]["es_m"]["over_77"] == 0
    assert summary["demographics"]["sc_f"]["over_77"] == 16 * 16 * 8 * 4
    with pytest.raises(RuntimeError, match="sc_f"):
        token_guard.require_production_prompt_budget(summary)
