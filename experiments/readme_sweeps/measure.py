#!/usr/bin/env python3
"""Embed README sweep renders and write pairwise ArcFace measurements."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from different_pipeline_generator.measurements import (
    ensure_unique_renders,
    pair_rows,
    parse_render,
    summarize_pairs as summarize,
)


def prompt_centroids(records: list[dict], embeddings: np.ndarray) -> tuple[list[str], np.ndarray]:
    names = sorted({record["id"] for record in records})
    rows = []
    for name in names:
        indices = [record["index"] for record in records if record["id"] == name]
        centroid = embeddings[indices].mean(axis=0)
        rows.append(centroid / np.linalg.norm(centroid))
    return names, np.stack(rows)


def write_pairs(path: Path, pairs: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("a", "b", "cosine_similarity", "cosine_distance"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(pairs)


def contact_sheet(items: list[tuple[Path, str]], output: Path, columns: int) -> None:
    cell, label_height = 220, 30
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * (cell + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(items):
        image = ImageOps.fit(Image.open(path).convert("RGB"), (cell, cell))
        x = (index % columns) * cell
        y = (index // columns) * (cell + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 6, y + cell + 7), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)


def detect_largest(analyzer, bgr: np.ndarray) -> tuple[object | None, str]:
    faces = analyzer.get(bgr)
    mode = "original"
    if not faces:
        height, width = bgr.shape[:2]
        padded = np.full((height * 2, width * 2, 3), 127, dtype=bgr.dtype)
        padded[height // 2 : height // 2 + height, width // 2 : width // 2 + width] = bgr
        faces = analyzer.get(padded)
        mode = "padded"
    if not faces:
        return None, "failed"
    return max(
        faces,
        key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])),
    ), mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        face, detection_mode = detect_largest(analyzer, bgr)
        if face is None:
            failures.append(record["path"])
            continue
        embedding = face.embedding.astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        record.update(
            index=len(embeddings),
            detection_score=float(face.det_score),
            detection_mode=detection_mode,
        )
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
        "schema_version": "different-faces-readme-sweeps/v1",
        "embedding_model": "InsightFace buffalo_l recognition model",
        "image_count": len(records),
        "face_detection_failures": failures,
        "experiments": defaultdict(dict),
        "prompt_manifest": json.loads(args.config.read_text()),
    }

    sheets = args.output / "contact-sheets"
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["set"], record["engine"], record["id"])].append(record)

    for (experiment, engine, identity), subset in sorted(grouped.items()):
        subset.sort(key=lambda record: record["seed"])
        names = [f"{identity}-seed-{record['seed']}" for record in subset]
        rows = matrix[[record["index"] for record in subset]]
        pairs = pair_rows(names, rows)
        metrics["experiments"].setdefault(experiment, {}).setdefault(engine, {})[identity] = {
            "rendered_images": summarize(pairs)
        }
        write_pairs(args.output / f"pairwise_{experiment}_{engine}_{identity}.csv", pairs)
        contact_sheet(
            [(args.images / record["path"], str(record["seed"])) for record in subset],
            sheets / experiment / f"{engine}-{identity}.jpg",
            columns=4,
        )

    for experiment in ("scandinavian_features", "lens_framing"):
        for engine in ("realvis", "zimage"):
            subset = [
                record
                for record in records
                if record["set"] == experiment and record["engine"] == engine
            ]
            names, centroids = prompt_centroids(subset, matrix)
            pairs = pair_rows(names, centroids)
            metrics["experiments"][experiment][engine]["prompt_centroids"] = summarize(pairs)
            write_pairs(args.output / f"pairwise_{experiment}_{engine}_centroids.csv", pairs)
            first_by_id = {}
            for record in subset:
                first_by_id.setdefault(record["id"], record)
            contact_sheet(
                [
                    (args.images / first_by_id[name]["path"], name.replace("_", " "))
                    for name in names
                ],
                sheets / experiment / f"{engine}-prompts.jpg",
                columns=3,
            )

    metrics["experiments"] = dict(metrics["experiments"])
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics["experiments"], indent=2))


if __name__ == "__main__":
    main()
