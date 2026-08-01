#!/usr/bin/env python3
"""Generate the README seed, feature, and lens sweeps with two models."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from different_pipeline_generator.comfyui import (
    realvis_workflow,
    request_json,
    run_generation_jobs,
    zimage_workflow,
)


SERVER = os.environ.get("COMFY_SERVER", "127.0.0.1:8188")
PREFIX = os.environ.get("OUTPUT_PREFIX", "readme_sweeps")
STATE_PATH = Path(os.environ.get("GENERATION_STATE", "/workspace/readme-sweeps/generation_state.json"))
ENGINES = ("realvis", "zimage")


def build_jobs(config: dict) -> list[dict]:
    negative = config["negative"]
    jobs = []

    for item in config["seed_sweeps"]["prompts"]:
        for engine in ENGINES:
            for seed in config["seed_sweeps"]["seeds"]:
                jobs.append({"set": "seed_sweeps", "engine": engine, "id": item["id"], "label": item["label"], "seed": seed, "prompt": item["text"], "negative": negative})

    feature_config = config["scandinavian_features"]
    for item in feature_config["prompts"]:
        prompt = f"{feature_config['base']}, {item['addition']}, {feature_config['framing']}"
        for engine in ENGINES:
            for seed in feature_config["seeds"]:
                jobs.append({"set": "scandinavian_features", "engine": engine, "id": item["id"], "label": item["label"], "seed": seed, "prompt": prompt, "negative": negative})

    lens_config = config["lens_framing"]
    for item in lens_config["prompts"]:
        prompt = f"{lens_config['base']}, {item['addition']}"
        for engine in ENGINES:
            for seed in lens_config["seeds"]:
                jobs.append({"set": "lens_framing", "engine": engine, "id": item["id"], "label": item["label"], "seed": seed, "prompt": prompt, "negative": negative})
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
