#!/usr/bin/env python3
"""Generate controlled two-detail and prompt-length background sweeps."""

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
PREFIX = "background_pairs"
STATE_PATH = Path("/workspace/background-pairs/generation_state.json")


def prompt_for(config: dict, detail_ids: list[str], engine: str) -> str:
    details = {item["id"]: item["text"] for item in config["detail_pool"]}
    parts = [config["lock"]]
    if detail_ids:
        parts.append("Background: " + "; ".join(details[item] for item in detail_ids))
    parts.append(config["framing"])
    if engine == "zimage":
        parts.append(config["zimage_tail"])
    return ", ".join(parts)


def build_jobs(config: dict) -> list[dict]:
    jobs = []
    for engine in ("realvis", "zimage"):
        for profile in config["paired_backgrounds"]["profiles"]:
            for seed in config["paired_backgrounds"]["seeds"]:
                jobs.append({
                    "set": "paired_backgrounds",
                    "engine": engine,
                    "id": profile["id"],
                    "detail_ids": profile["details"],
                    "seed": seed,
                    "prompt": prompt_for(config, profile["details"], engine),
                    "negative": config["negative"],
                })
        ordered = config["detail_count"]["ordered_details"]
        for count in config["detail_count"]["counts"]:
            detail_ids = ordered[:count]
            for seed in config["detail_count"]["seeds"]:
                jobs.append({
                    "set": "detail_count",
                    "engine": engine,
                    "id": f"count_{count}",
                    "detail_ids": detail_ids,
                    "seed": seed,
                    "prompt": prompt_for(config, detail_ids, engine),
                    "negative": config["negative"],
                })
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
