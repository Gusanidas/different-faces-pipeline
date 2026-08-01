#!/usr/bin/env python3
"""Over-generate a resumable demographic seed bank with RealVisXL or Z-Image.

The script submits deterministic ComfyUI workflows. Each nominal person gets
two short biographical details, one facial-feature bundle, one lens/framing
choice, and several seeds while the demographic lock stays fixed. These inputs
enrich the candidate cloud, but human review—not the prompt or embedding
distance alone—decides which faces are different people.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from different_pipeline_generator.comfyui import (
    atomic_write_json,
    history_result,
    queued_prompt_ids,
    realvis_workflow,
    request_json,
    zimage_workflow,
)
from different_pipeline_generator.demographics import DEMOGRAPHICS
from different_pipeline_generator.prompts import (
    FACIAL_CUES,
    LENS_FRAMINGS,
    PRESENTATION,
    character_prompt,
)


PLAIN_POSITIVE = (
    "a very plain ordinary everyday person, not a model, not glamorous, average-looking, "
    "asymmetric imperfect natural features, unretouched bare skin, candid photograph"
)
NEGATIVE = (
    "shirtless, bare chest, beautiful, model, attractive, airbrushed, flawless skin, "
    "symmetrical face, glamorous, "
    "heavy makeup, cartoon, illustration, anime, 3d render, cgi, deformed, disfigured, "
    "watermark, text, duplicate, two people, multiple faces"
)


def character_prompts(demo_key: str, count: int, seed: int) -> list[str]:
    demo = DEMOGRAPHICS[demo_key]
    maximum = (
        len(demo.occupations)
        * len(demo.personal_details)
        * len(FACIAL_CUES)
        * len(LENS_FRAMINGS)
    )
    if not 1 <= count <= maximum:
        raise ValueError(f"count must be between 1 and {maximum} for {demo_key}")
    rng = random.Random(seed)
    seen: set[tuple[str, ...]] = set()
    prompts: list[str] = []
    attempts = 0
    attempt_limit = max(10_000, count * 100)
    while len(prompts) < count and attempts < attempt_limit:
        attempts += 1
        variation = (
            rng.choice(demo.occupations),
            rng.choice(demo.personal_details),
            rng.choice(FACIAL_CUES),
            rng.choice(LENS_FRAMINGS),
        )
        if variation in seen:
            continue
        seen.add(variation)
        occupation, personal_detail, facial_cues, lens_framing = variation
        prompts.append(
            character_prompt(
                demo_key,
                occupation,
                personal_detail,
                facial_cues,
                lens_framing,
            )
        )
    if len(prompts) != count:
        raise RuntimeError(
            f"generated only {len(prompts)}/{count} unique prompts in {attempts} attempts"
        )
    return prompts


def seed_for_render(
    base_seed: int,
    character_index: int,
    seed_offset: int,
    seeds_per_character: int,
) -> int:
    """Allocate a collision-free seed block to every character."""
    return base_seed + character_index * seeds_per_character + seed_offset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("realvis", "zimage"))
    parser.add_argument("demographic", choices=tuple(DEMOGRAPHICS))
    parser.add_argument("--characters", type=int, default=100)
    parser.add_argument("--seeds-per-character", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--server", default="127.0.0.1:8188")
    parser.add_argument("--checkpoint", default="RealVisXL_V5.0_fp16.safetensors")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--execution-retries", type=int, default=1)
    args = parser.parse_args()
    if args.characters < 1:
        parser.error("--characters must be positive")
    if args.seeds_per_character < 1:
        parser.error("--seeds-per-character must be positive")
    if args.execution_retries < 0:
        parser.error("--execution-retries cannot be negative")

    prompts = character_prompts(args.demographic, args.characters, args.seed)
    tag = f"bank_{args.model}_{args.demographic}"
    state_path = args.state or Path("work") / f"{tag}_submission.json"
    config = {
        "model": args.model,
        "demographic": args.demographic,
        "characters": args.characters,
        "seeds_per_character": args.seeds_per_character,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "server": args.server,
        "prompt_sha256": hashlib.sha256(
            json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("config") != config:
            raise ValueError(
                f"state file {state_path} belongs to a different configuration; "
                "use a different --state path"
            )
    else:
        state = {"version": 1, "config": config, "jobs": {}, "submission_errors": {}}

    job_specs = {}
    for character_index, prompt in enumerate(prompts):
        for seed_offset in range(args.seeds_per_character):
            render_seed = seed_for_render(
                args.seed, character_index, seed_offset, args.seeds_per_character
            )
            prefix = f"{tag}/c{character_index:03d}_s{seed_offset}"
            workflow = (
                zimage_workflow(
                    prompt,
                    render_seed,
                    prefix,
                    positive_suffix=PLAIN_POSITIVE,
                )
                if args.model == "zimage"
                else realvis_workflow(
                    prompt,
                    NEGATIVE,
                    render_seed,
                    prefix,
                    checkpoint=args.checkpoint,
                )
            )
            key = f"c{character_index:03d}_s{seed_offset}"
            job_specs[key] = {"seed": render_seed, "prefix": prefix, "workflow": workflow}

    queue_ids = queued_prompt_ids(request_json(args.server, "/queue"))
    history = request_json(args.server, "/history")

    def submit(key: str) -> None:
        spec = job_specs[key]
        response = request_json(args.server, "/prompt", {"prompt": spec["workflow"]})
        previous = state["jobs"].get(key, {})
        state["jobs"][key] = {
            "prompt_id": response["prompt_id"],
            "seed": spec["seed"],
            "prefix": spec["prefix"],
            "execution_attempts": int(previous.get("execution_attempts", 0)) + 1,
            "status": "submitted",
        }
        state["submission_errors"].pop(key, None)
        atomic_write_json(state_path, state)

    for key in job_specs:
        record = state["jobs"].get(key)
        if record:
            if record.get("status") == "success":
                continue
            prompt_id = str(record["prompt_id"])
            history_entry = history.get(prompt_id)
            result = history_result(history_entry)
            if result == "success":
                record["status"] = "success"
                continue
            if prompt_id in queue_ids:
                continue
            if result == "pending" and history_entry is not None:
                # A history entry is still owned by ComfyUI even if a separately
                # fetched queue snapshot no longer lists it. Do not create a ghost
                # duplicate while its terminal state is unresolved.
                continue
            if result == "failed" and record.get("execution_attempts", 1) > args.execution_retries:
                record["status"] = "failed"
                continue
        try:
            submit(key)
        except Exception as error:
            state["submission_errors"][key] = f"{type(error).__name__}: {error}"
            atomic_write_json(state_path, state)
            print(f"warning: failed to submit {key}: {error}", flush=True)

    total = len(job_specs)
    print(f"[{tag}] tracking {len(state['jobs'])}/{total} submitted renders", flush=True)
    previous = -1
    while True:
        queue_ids = queued_prompt_ids(request_json(args.server, "/queue"))
        history = request_json(args.server, "/history")
        succeeded, failed, pending = [], [], []
        for key in job_specs:
            record = state["jobs"].get(key)
            if record is None:
                failed.append(key)
                continue
            if record.get("status") == "success":
                succeeded.append(key)
                continue
            prompt_id = str(record["prompt_id"])
            history_entry = history.get(prompt_id)
            result = history_result(history_entry)
            if result == "success":
                record["status"] = "success"
                succeeded.append(key)
            elif result == "failed":
                if record.get("execution_attempts", 1) <= args.execution_retries:
                    try:
                        submit(key)
                        pending.append(key)
                    except Exception as error:
                        state["submission_errors"][key] = f"{type(error).__name__}: {error}"
                        failed.append(key)
                else:
                    record["status"] = "failed"
                    failed.append(key)
            elif prompt_id in queue_ids:
                pending.append(key)
            elif history_entry is None:
                if record.get("execution_attempts", 1) <= args.execution_retries:
                    try:
                        submit(key)
                        pending.append(key)
                    except Exception as error:
                        state["submission_errors"][key] = f"{type(error).__name__}: {error}"
                        failed.append(key)
                else:
                    record["status"] = "failed"
                    failed.append(key)
            else:
                pending.append(key)
        atomic_write_json(state_path, state)

        if len(succeeded) != previous:
            print(
                f"[{tag}] success={len(succeeded)}/{total} "
                f"pending={len(pending)} failed={len(failed)}",
                flush=True,
            )
            previous = len(succeeded)
        if len(succeeded) == total:
            return
        if not pending:
            preview = ", ".join(failed[:10])
            raise RuntimeError(
                f"{len(failed)} renders did not succeed; see {state_path}. Failed: {preview}"
            )
        time.sleep(10)


if __name__ == "__main__":
    main()
