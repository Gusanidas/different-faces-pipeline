#!/usr/bin/env python3
"""Record exact prompt lengths for the text encoders used by the experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from different_pipeline_generator.demographics import DEMOGRAPHICS
from different_pipeline_generator.prompts import all_character_prompts

from generate import build_jobs


SDXL_TOKENIZERS = ("sdxl_clip_l", "sdxl_clip_g")
SDXL_WINDOW = 77


def token_counts(prompt: str, tokenizers: dict) -> dict[str, int]:
    return {
        name: len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        for name, tokenizer in tokenizers.items()
    }


def production_prompt_summary(tokenizers: dict) -> dict:
    """Measure every prompt the production seed-bank generator can emit."""
    demographics = {}
    for demo_key in DEMOGRAPHICS:
        minima = {name: math.inf for name in SDXL_TOKENIZERS}
        maxima = {name: 0 for name in SDXL_TOKENIZERS}
        count = 0
        over_window = 0
        for prompt in all_character_prompts(demo_key):
            counts = token_counts(prompt, tokenizers)
            count += 1
            for name in SDXL_TOKENIZERS:
                minima[name] = min(minima[name], counts[name])
                maxima[name] = max(maxima[name], counts[name])
            if any(counts[name] > SDXL_WINDOW for name in SDXL_TOKENIZERS):
                over_window += 1
        demographics[demo_key] = {
            "count": count,
            "sdxl_clip_l_min": int(minima["sdxl_clip_l"]),
            "sdxl_clip_l_max": maxima["sdxl_clip_l"],
            "sdxl_clip_g_min": int(minima["sdxl_clip_g"]),
            "sdxl_clip_g_max": maxima["sdxl_clip_g"],
            "over_77": over_window,
            "all_fit_one_sdxl_window": over_window == 0,
        }
    return {
        "demographics": demographics,
        "all_fit_one_sdxl_window": all(
            row["all_fit_one_sdxl_window"] for row in demographics.values()
        ),
    }


def require_production_prompt_budget(summary: dict) -> None:
    if summary["all_fit_one_sdxl_window"]:
        return
    failures = [
        f"{key}: max(CLIP-L={row['sdxl_clip_l_max']}, "
        f"CLIP-G={row['sdxl_clip_g_max']}), over={row['over_77']}"
        for key, row in summary["demographics"].items()
        if not row["all_fit_one_sdxl_window"]
    ]
    raise RuntimeError(
        "a production prompt exceeds one 77-token SDXL window (" + "; ".join(failures) + ")"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config = json.loads(args.config.read_text())
    tokenizers = {
        "sdxl_clip_l": AutoTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer"
        ),
        "sdxl_clip_g": AutoTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2"
        ),
    }
    unique = {}
    for job in build_jobs(config):
        key = (job["set"], job["engine"], job["id"])
        unique.setdefault(key, job)

    records = []
    for (experiment, engine, identity), job in sorted(unique.items()):
        counts = token_counts(job["prompt"], tokenizers)
        records.append(
            {
                "set": experiment,
                "engine": engine,
                "id": identity,
                "detail_count": len(job["detail_ids"]),
                "prompt": job["prompt"],
                "token_counts": counts,
                "sdxl_native_windows": max(
                    math.ceil(counts["sdxl_clip_l"] / SDXL_WINDOW),
                    math.ceil(counts["sdxl_clip_g"] / SDXL_WINDOW),
                ),
            }
        )

    paired_realvis = [
        record
        for record in records
        if record["set"] == "paired_backgrounds" and record["engine"] == "realvis"
    ]
    summary = {
        "paired_backgrounds": {
            "count": len(paired_realvis),
            "sdxl_clip_l_min": min(r["token_counts"]["sdxl_clip_l"] for r in paired_realvis),
            "sdxl_clip_l_max": max(r["token_counts"]["sdxl_clip_l"] for r in paired_realvis),
            "sdxl_clip_g_min": min(r["token_counts"]["sdxl_clip_g"] for r in paired_realvis),
            "sdxl_clip_g_max": max(r["token_counts"]["sdxl_clip_g"] for r in paired_realvis),
            "all_fit_one_sdxl_window": all(r["sdxl_native_windows"] == 1 for r in paired_realvis),
        },
        "production": production_prompt_summary(tokenizers),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "prompts": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["paired_backgrounds"]["all_fit_one_sdxl_window"]:
        raise RuntimeError("an experiment two-detail prompt exceeds one SDXL text window")
    require_production_prompt_budget(summary["production"])


if __name__ == "__main__":
    main()
