#!/usr/bin/env python3
"""Generate the controlled seed, feature-word, and backstory experiments."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from different_pipeline_generator.comfyui import (
    realvis_workflow,
    request_json,
    run_generation_jobs,
    zimage_workflow,
)


SERVER = "127.0.0.1:8188"
PREFIX = "reproduction_20260729"
STATE_PATH = Path("/workspace/reproduction/generation_state.json")


def build_jobs(config: dict) -> list[dict]:
    lock = config["lock"]
    framing = config["framing"]
    negative = config["negative"]
    jobs = []

    base_prompt = f"{lock}, {framing}"
    for seed in config["seed_only"]["seeds"]:
        jobs.append({"set": "seed_only", "engine": "realvis", "id": "base", "seed": seed, "prompt": base_prompt, "negative": negative})

    for item in config["feature_words"]["prompts"]:
        prompt = f"{lock}, {item['addition']}, {framing}"
        for seed in config["feature_words"]["seeds"]:
            jobs.append({"set": "feature_words", "engine": "realvis", "id": item["id"], "seed": seed, "prompt": prompt, "negative": negative})

    for engine in ("realvis", "zimage"):
        for item in config["backstories"]["prompts"]:
            prompt = f"{lock}, {item['text']}, {framing}"
            if engine == "zimage":
                prompt = f"{prompt}, {config['zimage_tail']}"
            for seed in config["backstories"]["seeds"]:
                jobs.append({"set": "backstories", "engine": engine, "id": item["id"], "seed": seed, "prompt": prompt, "negative": negative})
    return jobs


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("prompts.json")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    digest = hashlib.sha256(config_bytes).hexdigest()
    jobs = build_jobs(config)

    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
        if state.get("config_sha256") != digest:
            raise ValueError("the existing state belongs to a different prompt manifest")
    else:
        state = {"schema_version": 1, "config_sha256": digest, "jobs": {}}

    keyed_jobs = {
        f"{job['set']}/{job['engine']}/{job['id']}_s{job['seed']}": job
        for job in jobs
    }
    if len(keyed_jobs) != len(jobs):
        raise ValueError("the prompt manifest contains duplicate generation jobs")

    def workflow_for(job: dict, prefix: str) -> dict:
        if job["engine"] == "realvis":
            return realvis_workflow(
                job["prompt"], job["negative"], job["seed"], prefix
            )
        return zimage_workflow(job["prompt"], job["seed"], prefix)

    run_generation_jobs(
        server=SERVER,
        jobs=keyed_jobs,
        state=state,
        state_path=STATE_PATH,
        output_prefix=PREFIX,
        workflow_for=workflow_for,
        request=request_json,
    )


if __name__ == "__main__":
    main()
