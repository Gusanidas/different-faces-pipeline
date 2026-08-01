"""Shared ComfyUI workflows and resumable experiment execution."""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


JsonObject = dict[str, Any]
RequestJson = Callable[[str, str, JsonObject | None], JsonObject]


def request_json(
    server: str,
    path: str,
    payload: JsonObject | None = None,
    *,
    attempts: int = 4,
) -> JsonObject:
    """Request a ComfyUI JSON endpoint with bounded exponential backoff."""
    url = f"http://{server}{path}"
    for attempt in range(1, attempts + 1):
        try:
            if payload is None:
                return json.load(urllib.request.urlopen(url, timeout=120))
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            return json.load(urllib.request.urlopen(request, timeout=120))
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))
    raise AssertionError("unreachable")


def atomic_write_json(path: Path, payload: JsonObject) -> None:
    """Replace a JSON state file only after its complete successor is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def history_result(entry: JsonObject | None) -> str:
    """Return ``pending``, ``success``, or ``failed`` for a history entry."""
    if entry is None:
        return "pending"
    status = entry.get("status", {})
    status_string = str(status.get("status_str", "")).lower()
    completed = status.get("completed")
    message_types = {
        message[0]
        for message in status.get("messages", [])
        if isinstance(message, (list, tuple)) and message
    }
    if status_string == "success" and completed is not False:
        return "success"
    if (
        status_string in {"error", "failed"}
        or "execution_error" in message_types
        or (completed is True and status_string != "success")
    ):
        return "failed"
    return "pending"


def queued_prompt_ids(queue: JsonObject) -> set[str]:
    """Extract prompt IDs from both running and pending ComfyUI queues."""
    ids: set[str] = set()
    for key in ("queue_running", "queue_pending"):
        for item in queue.get(key, []):
            if isinstance(item, (list, tuple)) and len(item) > 1:
                ids.add(str(item[1]))
    return ids


def realvis_workflow(
    prompt: str,
    negative: str,
    seed: int,
    prefix: str,
    *,
    checkpoint: str = "RealVisXL_V5.0_fp16.safetensors",
) -> JsonObject:
    """Build the shared RealVisXL workflow used by production and experiments."""
    return {
        "checkpoint": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["checkpoint", 1]},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["checkpoint", 1]},
        },
        "latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "sample": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 30,
                "cfg": 5.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["checkpoint", 0],
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "latent_image": ["latent", 0],
            },
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sample", 0], "vae": ["checkpoint", 2]},
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": prefix, "images": ["decode", 0]},
        },
    }


def zimage_workflow(
    prompt: str,
    seed: int,
    prefix: str,
    *,
    positive_suffix: str | None = None,
) -> JsonObject:
    """Build the shared Z-Image workflow used by production and experiments."""
    if positive_suffix:
        prompt = f"{prompt}, {positive_suffix}"
    return {
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "z_image_turbo_bf16.safetensors",
                "weight_dtype": "default",
            },
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_3_4b.safetensors",
                "type": "lumina2",
                "device": "default",
            },
        },
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "model": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["unet", 0], "shift": 3.0},
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["clip", 0]},
        },
        "negative": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["positive", 0]},
        },
        "latent": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "sample": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 8,
                "cfg": 1.0,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["model", 0],
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "latent_image": ["latent", 0],
            },
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]},
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": prefix, "images": ["decode", 0]},
        },
    }


def run_generation_jobs(
    *,
    server: str,
    jobs: Mapping[str, JsonObject],
    state: JsonObject,
    state_path: Path,
    output_prefix: str,
    workflow_for: Callable[[JsonObject, str], JsonObject],
    request: RequestJson = request_json,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = 10,
) -> None:
    """Submit missing jobs and wait without duplicating in-flight prompts.

    Queue state is fetched before history. If a prompt finishes between those
    requests, either the queue snapshot or the newer history snapshot owns it,
    closing the restart race that otherwise creates duplicate renders.
    """

    def snapshots() -> tuple[set[str], JsonObject]:
        queue_ids = queued_prompt_ids(request(server, "/queue", None))
        history = request(server, "/history", None)
        return queue_ids, history

    def submit(key: str, job: JsonObject) -> None:
        prefix = f"{output_prefix}/{key}"
        response = request(server, "/prompt", {"prompt": workflow_for(job, prefix)})
        state["jobs"][key] = {
            **job,
            "prefix": prefix,
            "prompt_id": response["prompt_id"],
            "status": "submitted",
        }
        atomic_write_json(state_path, state)

    queue_ids, history = snapshots()
    for key, job in jobs.items():
        record = state["jobs"].get(key)
        if record:
            if record.get("status") == "success":
                continue
            if record.get("status") == "failed":
                continue
            prompt_id = str(record.get("prompt_id", ""))
            entry = history.get(prompt_id) if prompt_id else None
            result = history_result(entry)
            if result == "success":
                record["status"] = "success"
                continue
            if result == "failed":
                record["status"] = "failed"
                continue
            if prompt_id in queue_ids or entry is not None:
                record["status"] = "pending"
                continue
        submit(key, job)
    atomic_write_json(state_path, state)

    total = len(jobs)
    while True:
        queue_ids, history = snapshots()
        counts = {"success": 0, "failed": 0, "pending": 0}
        for key in jobs:
            record = state["jobs"].get(key)
            if record is None:
                counts["failed"] += 1
                continue
            if record.get("status") == "success":
                counts["success"] += 1
                continue
            if record.get("status") == "failed":
                counts["failed"] += 1
                continue
            prompt_id = str(record.get("prompt_id", ""))
            result = history_result(history.get(prompt_id) if prompt_id else None)
            if result == "pending" and prompt_id in queue_ids:
                record["status"] = "pending"
            else:
                record["status"] = result
            counts[record["status"]] += 1
        atomic_write_json(state_path, state)
        print(
            f"success={counts['success']}/{total} pending={counts['pending']} "
            f"failed={counts['failed']}",
            flush=True,
        )
        if counts["failed"]:
            raise RuntimeError(
                f"{counts['failed']} ComfyUI jobs failed; inspect {state_path}"
            )
        if counts["success"] == total:
            print("GENERATION_COMPLETE", flush=True)
            return
        sleep(poll_interval)
