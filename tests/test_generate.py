import importlib.util
import json
import sys
from pathlib import Path

import pytest

from different_pipeline_generator.prompts import all_character_prompts


SCRIPT = Path(__file__).parents[1] / "scripts" / "01_generate_seed_bank.py"
SPEC = importlib.util.spec_from_file_location("generate_seed_bank", SCRIPT)
generate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generate)


def test_seed_blocks_do_not_collide_above_ten_seeds_per_character():
    seeds = {
        generate.seed_for_render(1000, character, offset, 12)
        for character in range(3)
        for offset in range(12)
    }
    assert len(seeds) == 36


def test_prompt_count_is_bounded_and_unique():
    prompts = generate.character_prompts("es_m", 200, 42)
    assert len(prompts) == len(set(prompts)) == 200
    with pytest.raises(ValueError, match="count"):
        generate.character_prompts("es_m", 0, 42)


def test_character_prompts_use_exactly_two_known_background_details():
    demo = generate.DEMOGRAPHICS["es_m"]
    for prompt in generate.character_prompts("es_m", 100, 42):
        background = prompt.split("Background: ", 1)[1].split(
            f", {generate.PRESENTATION}", 1
        )[0]
        details = background.split("; ")
        assert len(details) == 2
        assert details[0] in demo.occupations
        assert details[1] in demo.personal_details


def test_character_prompts_include_facial_cues_and_lens_variation():
    prompts = generate.character_prompts("es_m", 100, 42)

    assert all(sum(cue in prompt for cue in generate.FACIAL_CUES) == 1 for prompt in prompts)
    assert all(sum(lens in prompt for lens in generate.LENS_FRAMINGS) == 1 for prompt in prompts)
    assert {cue for cue in generate.FACIAL_CUES if any(cue in prompt for prompt in prompts)} == set(
        generate.FACIAL_CUES
    )
    assert {lens for lens in generate.LENS_FRAMINGS if any(lens in prompt for prompt in prompts)} == set(
        generate.LENS_FRAMINGS
    )


def test_production_locks_match_the_documented_cohorts_and_full_prompt_space():
    assert generate.DEMOGRAPHICS["es_m"].lock == (
        "a Spanish man in his early 30s, short medium-brown hair, tanned skin"
    )
    assert generate.DEMOGRAPHICS["sc_f"].lock == (
        "a Scandinavian woman in her late 20s, long light-blonde hair, fair skin, blue eyes"
    )
    assert sum(1 for _ in all_character_prompts("es_m")) == 16 * 16 * 8 * 4
    assert sum(1 for _ in all_character_prompts("sc_f")) == 16 * 16 * 8 * 4


def test_comfy_history_requires_explicit_success():
    assert generate.history_result(
        {"status": {"status_str": "success", "completed": True, "messages": []}}
    ) == "success"
    assert generate.history_result(
        {"status": {"status_str": "error", "completed": True, "messages": []}}
    ) == "failed"
    assert generate.history_result(
        {"status": {"status_str": "success", "completed": False, "messages": []}}
    ) == "pending"
    assert generate.history_result(
        {"status": {"status_str": "", "completed": True, "messages": [["execution_error", {}]]}}
    ) == "failed"
    assert generate.history_result(None) == "pending"


def test_generation_state_retries_execution_failure_and_finishes(monkeypatch, tmp_path):
    submitted = []

    def fake_request(server, path, payload=None, **_kwargs):
        assert server == "fake:8188"
        if path == "/queue":
            return {"queue_running": [], "queue_pending": []}
        if path == "/history":
            history = {}
            for prompt_id in submitted:
                failed = prompt_id == "p2"
                history[prompt_id] = {
                    "status": {
                        "status_str": "error" if failed else "success",
                        "completed": True,
                        "messages": [["execution_error", {}]] if failed else [],
                    }
                }
            return history
        if path == "/prompt":
            prompt_id = f"p{len(submitted) + 1}"
            submitted.append(prompt_id)
            return {"prompt_id": prompt_id}
        raise AssertionError(path)

    state_path = tmp_path / "state.json"
    monkeypatch.setattr(generate, "request_json", fake_request)
    monkeypatch.setattr(generate.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "realvis",
            "es_m",
            "--characters",
            "1",
            "--seeds-per-character",
            "2",
            "--server",
            "fake:8188",
            "--state",
            str(state_path),
        ],
    )

    generate.main()

    state = json.loads(state_path.read_text())
    assert submitted == ["p1", "p2", "p3"]
    assert {record["status"] for record in state["jobs"].values()} == {"success"}
    assert sorted(record["execution_attempts"] for record in state["jobs"].values()) == [1, 2]


def test_resume_does_not_resubmit_a_pending_history_entry(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    prompts = generate.character_prompts("es_m", 1, 1000)
    config = {
        "model": "realvis",
        "demographic": "es_m",
        "characters": 1,
        "seeds_per_character": 1,
        "seed": 1000,
        "checkpoint": "RealVisXL_V5.0_fp16.safetensors",
        "server": "fake:8188",
        "prompt_sha256": generate.hashlib.sha256(
            json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "config": config,
                "jobs": {
                    "c000_s0": {
                        "prompt_id": "existing",
                        "seed": 1000,
                        "prefix": "bank_realvis_es_m/c000_s0",
                        "execution_attempts": 1,
                        "status": "submitted",
                    }
                },
                "submission_errors": {},
            }
        )
    )
    history_calls = 0

    def fake_request(_server, path, payload=None, **_kwargs):
        nonlocal history_calls
        if path == "/queue":
            return {"queue_running": [], "queue_pending": []}
        if path == "/history":
            history_calls += 1
            return {
                "existing": {
                    "status": {
                        "status_str": "success",
                        "completed": history_calls > 1,
                        "messages": [],
                    }
                }
            }
        if path == "/prompt":
            raise AssertionError("a pending history entry must not be resubmitted")
        raise AssertionError((path, payload))

    monkeypatch.setattr(generate, "request_json", fake_request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "realvis",
            "es_m",
            "--characters",
            "1",
            "--seeds-per-character",
            "1",
            "--server",
            "fake:8188",
            "--state",
            str(state_path),
        ],
    )

    generate.main()

    state = json.loads(state_path.read_text())
    assert history_calls == 2
    assert state["jobs"]["c000_s0"]["status"] == "success"
