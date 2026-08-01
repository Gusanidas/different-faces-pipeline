#!/usr/bin/env python3
"""Combine selected banks and sample new separated InstantID vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from different_pipeline_generator.naming import clean_comfy_name, ensure_unique
from different_pipeline_generator.sampling import fit_cloud, sample_cloud


def parse_bank(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("banks must use LABEL=/path/to/selected.npz") from error
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", action="append", required=True, type=parse_bank)
    parser.add_argument("--sampled", type=int, default=80)
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument("--radius-low", type=float, default=60)
    parser.add_argument("--radius-high", type=float, default=99)
    parser.add_argument("--gate", type=float)
    parser.add_argument(
        "--max-attempts",
        type=int,
        help="maximum proposals before failing (default: max(400, sampled * 50))",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    names: list[str] = []
    origins: list[str] = []
    unit_vectors: list[np.ndarray] = []
    raw_vectors: list[np.ndarray] = []
    norms: list[float] = []
    for label, path in args.bank:
        data = np.load(path, allow_pickle=True)
        lengths = {key: len(data[key]) for key in ("files", "emb_ant", "ant_norms")}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"bank {path} has misaligned arrays: {lengths}")
        for filename, unit, raw_norm in zip(data["files"], data["emb_ant"], data["ant_norms"]):
            names.append(f"{label}_{clean_comfy_name(str(filename))}")
            origins.append(label)
            unit_vectors.append(unit.astype(np.float64))
            raw_vectors.append((unit * raw_norm).astype(np.float32))
            norms.append(float(raw_norm))

    if not unit_vectors:
        raise ValueError("the supplied banks contain no keeper embeddings")
    ensure_unique(names, label="keeper identity names")

    keepers = np.stack(unit_vectors)
    model = fit_cloud(keepers)
    sampled = sample_cloud(
        model,
        args.sampled,
        radius_percentiles=(args.radius_low, args.radius_high),
        gate=args.gate,
        seed=args.seed,
        max_attempts=args.max_attempts,
    )
    mean_raw_norm = float(np.mean(norms))
    for index, unit in enumerate(sampled.vectors):
        names.append(f"sampled_{index:03d}")
        origins.append("sampled")
        unit_vectors.append(unit)
        raw_vectors.append((unit * mean_raw_norm).astype(np.float32))
    ensure_unique(names, label="candidate identity names")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        names=np.asarray(names),
        origins=np.asarray(origins),
        unit=np.stack(unit_vectors).astype(np.float32),
        raw=np.stack(raw_vectors),
    )
    metadata = {
        "keepers": len(keepers),
        "sampled": args.sampled,
        "attempts": sampled.attempts,
        "max_attempts": args.max_attempts or max(400, args.sampled * 50),
        "gate": sampled.gate,
        "radius": [sampled.radius_low, sampled.radius_high],
        "mean_raw_norm": mean_raw_norm,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
