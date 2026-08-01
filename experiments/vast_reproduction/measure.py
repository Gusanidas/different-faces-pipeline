#!/usr/bin/env python3
"""Embed the reproduction images and persist complete pairwise measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from different_pipeline_generator.measurements import (
    ensure_unique_renders,
    pair_rows,
    parse_render,
    summarize_pairs,
)


summarize = partial(summarize_pairs, include_p05=True)


def unit_rows(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def prompt_centroids(records: list[dict], embeddings: np.ndarray) -> tuple[list[str], np.ndarray]:
    names = sorted({record["id"] for record in records})
    rows = []
    for name in names:
        indices = [record["index"] for record in records if record["id"] == name]
        centroid = embeddings[indices].mean(axis=0)
        rows.append(centroid / np.linalg.norm(centroid))
    return names, np.stack(rows)


def farthest_indices(rows: np.ndarray, count: int) -> list[int]:
    similarity = rows @ rows.T
    np.fill_diagonal(similarity, 1.0)
    pair = np.unravel_index(np.argmin(similarity), similarity.shape)
    selected = [int(pair[0]), int(pair[1])]
    while len(selected) < min(count, len(rows)):
        remaining = [index for index in range(len(rows)) if index not in selected]
        selected.append(min(remaining, key=lambda index: float(similarity[index, selected].max())))
    return selected[:count]


def write_pairs(path: Path, pairs: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("a", "b", "cosine_similarity", "cosine_distance"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(pairs)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contact_sheet(items: list[tuple[Path, str]], output: Path, columns: int = 6) -> None:
    cell, label_height = 220, 28
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * (cell + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(items):
        image = ImageOps.fit(Image.open(path).convert("RGB"), (cell, cell))
        x = (index % columns) * cell
        y = (index // columns) * (cell + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 6, y + cell + 6), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance-id")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from insightface.app import FaceAnalysis

    analyzer = FaceAnalysis(
        name="buffalo_l",
        root="/workspace/insightface",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    parsed_renders = []
    for path in sorted(args.images.rglob("*.png")):
        record = parse_render(path, args.images)
        if record is not None:
            parsed_renders.append((path, record))
    ensure_unique_renders([record for _, record in parsed_renders])

    records, embeddings, failures = [], [], []
    for path, record in parsed_renders:
        bgr = np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1])
        faces = analyzer.get(bgr)
        if not faces:
            failures.append(record["path"])
            continue
        face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
        embedding = face.embedding.astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        record.update({
            "index": len(embeddings),
            "detection_score": float(face.det_score),
        })
        records.append(record)
        embeddings.append(embedding)

    if not embeddings:
        raise RuntimeError("no faces were embedded")
    matrix = np.stack(embeddings)
    np.savez(
        args.output / "embeddings.npz",
        paths=np.asarray([record["path"] for record in records]),
        embeddings=matrix,
    )

    metrics = {
        "schema_version": "different-faces-vast-reproduction/v1",
        "execution": {
            "provider": "Vast.ai",
            "instance_id": args.instance_id,
            "gpu": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
            ).strip(),
        },
        "embedding_model": "InsightFace buffalo_l recognition model",
        "model_sha256": {
            "RealVisXL_V5.0_fp16.safetensors": sha256(
                Path("/workspace/ComfyUI/models/checkpoints/RealVisXL_V5.0_fp16.safetensors")
            ),
            "z_image_turbo_bf16.safetensors": sha256(
                Path("/workspace/ComfyUI/models/diffusion_models/z_image_turbo_bf16.safetensors")
            ),
            "qwen_3_4b.safetensors": sha256(
                Path("/workspace/ComfyUI/models/text_encoders/qwen_3_4b.safetensors")
            ),
        },
        "image_count": len(records),
        "face_detection_failures": failures,
        "experiments": {},
    }

    seed_records = [record for record in records if record["set"] == "seed_only"]
    seed_names = [f"seed-{record['seed']}" for record in seed_records]
    seed_rows = matrix[[record["index"] for record in seed_records]]
    pairs = pair_rows(seed_names, seed_rows)
    metrics["experiments"]["seed_only"] = summarize(pairs)
    write_pairs(args.output / "pairwise_seed_only.csv", pairs)

    for experiment, engines in (("feature_words", ("realvis",)), ("backstories", ("realvis", "zimage"))):
        metrics["experiments"][experiment] = {}
        for engine in engines:
            subset = [record for record in records if record["set"] == experiment and record["engine"] == engine]
            names, centroids = prompt_centroids(subset, matrix)
            centroid_pairs = pair_rows(names, centroids)
            image_names = [f"{record['id']}-seed-{record['seed']}" for record in subset]
            image_rows = matrix[[record["index"] for record in subset]]
            image_pairs = pair_rows(image_names, image_rows)
            selected = farthest_indices(centroids, min(6, len(names)))
            render_count = 12 if experiment == "backstories" else 6
            selected_renders = farthest_indices(image_rows, min(render_count, len(image_rows)))
            selected_render_pairs = pair_rows(
                [image_names[index] for index in selected_renders],
                image_rows[selected_renders],
            )
            key = engine
            metrics["experiments"][experiment][key] = {
                "prompt_centroids": summarize(centroid_pairs),
                "all_rendered_images": summarize(image_pairs),
                "farthest_prompt_ids": [names[index] for index in selected],
                "farthest_prompt_maximum_similarity": float(
                    (centroids[selected] @ centroids[selected].T)[np.triu_indices(len(selected), 1)].max()
                ),
                "farthest_render_selection": {
                    "selected": [subset[index]["path"] for index in selected_renders],
                    **summarize(selected_render_pairs),
                },
            }
            write_pairs(args.output / f"pairwise_{experiment}_{engine}_centroids.csv", centroid_pairs)
            write_pairs(
                args.output / f"pairwise_{experiment}_{engine}_selected_renders.csv",
                selected_render_pairs,
            )

    config = json.loads(args.config.read_text())
    metrics["prompt_manifest"] = config
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    sheets = args.output / "contact-sheets"
    contact_sheet(
        [(args.images / record["path"], f"seed {record['seed']}") for record in seed_records],
        sheets / "seed-only.jpg",
    )
    for experiment, engine in (("feature_words", "realvis"), ("backstories", "realvis"), ("backstories", "zimage")):
        subset = [record for record in records if record["set"] == experiment and record["engine"] == engine]
        first_by_id = {}
        for record in subset:
            first_by_id.setdefault(record["id"], record)
        contact_sheet(
            [(args.images / record["path"], identity) for identity, record in sorted(first_by_id.items())],
            sheets / f"{experiment}-{engine}.jpg",
        )
        render_count = 12 if experiment == "backstories" else 6
        selected = farthest_indices(
            matrix[[record["index"] for record in subset]],
            min(render_count, len(subset)),
        )
        contact_sheet(
            [
                (args.images / subset[index]["path"], f"{subset[index]['id']} · {subset[index]['seed']}")
                for index in selected
            ],
            sheets / f"{experiment}-{engine}-farthest.jpg",
        )
    print(json.dumps(metrics["experiments"], indent=2))


if __name__ == "__main__":
    main()
