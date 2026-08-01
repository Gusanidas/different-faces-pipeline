from __future__ import annotations

import json

import pytest

from different_pipeline_generator.comfyui import (
    history_result,
    run_generation_jobs,
)


def success_entry() -> dict:
    return {
        "status": {"status_str": "success", "completed": True, "messages": []}
    }


def test_history_completed_without_success_is_a_failure():
    entry = {"status": {"status_str": "error", "completed": True, "messages": []}}
    assert history_result(entry) == "failed"


@pytest.mark.parametrize("queue_name", ["queue_running", "queue_pending"])
def test_restart_does_not_resubmit_an_inflight_job(tmp_path, queue_name):
    key = "seed_sweeps/realvis/prompt_s100"
    state_path = tmp_path / "state.json"
    state = {
        "schema_version": 1,
        "config_sha256": "digest",
        "jobs": {
            key: {
                "prompt_id": "existing",
                "status": "submitted",
                "prefix": f"sweep/{key}",
            }
        },
    }
    queue_calls = 0
    submissions = []

    def fake_request(_server, path, payload=None):
        nonlocal queue_calls
        if path == "/queue":
            queue_calls += 1
            if queue_calls == 1:
                return {
                    "queue_running": [],
                    "queue_pending": [],
                    queue_name: [[0, "existing", {}, {}, []]],
                }
            return {"queue_running": [], "queue_pending": []}
        if path == "/history":
            return {} if queue_calls == 1 else {"existing": success_entry()}
        if path == "/prompt":
            submissions.append(payload)
            return {"prompt_id": "duplicate"}
        raise AssertionError(path)

    run_generation_jobs(
        server="fake:8188",
        jobs={key: {"seed": 100, "engine": "realvis"}},
        state=state,
        state_path=state_path,
        output_prefix="sweep",
        workflow_for=lambda _job, prefix: {"prefix": prefix},
        request=fake_request,
        sleep=lambda _seconds: None,
    )

    assert submissions == []
    assert json.loads(state_path.read_text())["jobs"][key]["status"] == "success"


def test_missing_unowned_job_is_resubmitted_once(tmp_path):
    key = "seed_sweeps/realvis/prompt_s100"
    state_path = tmp_path / "state.json"
    state = {
        "jobs": {key: {"prompt_id": "lost", "status": "submitted"}},
    }
    submitted = []

    def fake_request(_server, path, payload=None):
        if path == "/queue":
            return {"queue_running": [], "queue_pending": []}
        if path == "/history":
            return {"replacement": success_entry()} if submitted else {}
        if path == "/prompt":
            submitted.append(payload)
            return {"prompt_id": "replacement"}
        raise AssertionError(path)

    run_generation_jobs(
        server="fake:8188",
        jobs={key: {"seed": 100, "engine": "realvis"}},
        state=state,
        state_path=state_path,
        output_prefix="sweep",
        workflow_for=lambda _job, prefix: {"prefix": prefix},
        request=fake_request,
        sleep=lambda _seconds: None,
    )

    assert len(submitted) == 1
    assert state["jobs"][key]["prompt_id"] == "replacement"
