#!/usr/bin/env python3
"""Embed background-pair renders and write controlled comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from different_pipeline_generator.measurements import (
    ensure_unique_renders,
    pair_rows,
    parse_render,
    summarize_pairs as summarize,
)


README_PROFILE_IDS = ("p04", "p05", "p15", "p08", "p14", "p11")


def write_pairs(path: Path, pairs: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("a", "b", "cosine_similarity", "cosine_distance"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(pairs)


def detect_largest(analyzer, bgr: np.ndarray):
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
        key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
    ), mode


def normalized_centroid(records: list[dict], matrix: np.ndarray) -> np.ndarray:
    row = matrix[[record["index"] for record in records]].mean(axis=0)
    return row / np.linalg.norm(row)


def prompt_centroids(records: list[dict], matrix: np.ndarray) -> tuple[list[str], np.ndarray]:
    names = sorted({record["id"] for record in records})
    rows = [
        normalized_centroid([record for record in records if record["id"] == name], matrix)
        for name in names
    ]
    return names, np.stack(rows)


def fit_text(text: str, width: int, draw: ImageDraw.ImageDraw, font) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def load_font(size: int, *, bold: bool = False):
    names = (
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
    )
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def profile_matrix(
    config: dict,
    records: list[dict],
    images: Path,
    output: Path,
    seed: int,
    profile_ids: tuple[str, ...] | None = None,
) -> None:
    profiles = config["paired_backgrounds"]["profiles"]
    if profile_ids is not None:
        wanted = set(profile_ids)
        profiles = [profile for profile in profiles if profile["id"] in wanted]
        profiles.sort(key=lambda profile: profile_ids.index(profile["id"]))
    details = {item["id"]: item["text"] for item in config["detail_pool"]}
    font = load_font(17)
    label_font = load_font(16, bold=True)
    card_w, card_h, image_size, columns = 640, 390, 300, 2
    rows = (len(profiles) + columns - 1) // columns
    sheet = Image.new("RGB", (card_w * columns, card_h * rows), "white")
    draw = ImageDraw.Draw(sheet)
    lookup = {(r["engine"], r["id"], r["seed"]): images / r["path"] for r in records}
    for index, profile in enumerate(profiles):
        x = (index % columns) * card_w
        y = (index // columns) * card_h
        for offset, item in enumerate(profile["details"]):
            line = f"• {details[item]}"
            draw.text((x + 12, y + 10 + 24 * offset), line, fill="#222222", font=font)
        image_y = y + 82
        for engine_index, engine in enumerate(("realvis", "zimage")):
            path = lookup[(engine, profile["id"], seed)]
            image = ImageOps.fit(Image.open(path).convert("RGB"), (image_size, image_size))
            image_x = x + engine_index * image_size + 12
            sheet.paste(image, (image_x, image_y))
            draw.text(
                (image_x + 8, image_y + 8),
                "RealVisXL" if engine == "realvis" else "Z-Image",
                fill="white",
                stroke_fill="black",
                stroke_width=2,
                font=label_font,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90, optimize=True, progressive=True)


def count_matrix(records: list[dict], images: Path, output: Path, seed: int) -> None:
    counts = (0, 1, 2, 4, 8)
    image_size, label_w = 250, 120
    sheet = Image.new("RGB", (label_w + image_size * 2, image_size * len(counts)), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(16)
    label_font = load_font(15, bold=True)
    lookup = {(r["engine"], r["id"], r["seed"]): images / r["path"] for r in records}
    for row, count in enumerate(counts):
        y = row * image_size
        draw.text((10, y + 108), f"{count} details", fill="black", font=font)
        for column, engine in enumerate(("realvis", "zimage")):
            path = lookup[(engine, f"count_{count}", seed)]
            image = ImageOps.fit(Image.open(path).convert("RGB"), (image_size, image_size))
            sheet.paste(image, (label_w + column * image_size, y))
            draw.text(
                (label_w + column * image_size + 8, y + 8),
                "RealVisXL" if engine == "realvis" else "Z-Image",
                fill="white",
                stroke_fill="black",
                stroke_width=2,
                font=label_font,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90, optimize=True, progressive=True)


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
    image_records = [dict(record) for _, record in parsed_renders]
    for path, record in parsed_renders:
        bgr = np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1])
        face, mode = detect_largest(analyzer, bgr)
        if face is None:
            failures.append(record["path"])
            continue
        embedding = face.embedding.astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        record.update(index=len(embeddings), detection_score=float(face.det_score), detection_mode=mode)
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
        "schema_version": "different-faces-background-pairs/v1",
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip(),
        "embedding_model": "InsightFace buffalo_l recognition model",
        "image_count": len(records),
        "face_detection_failures": failures,
        "prompt_manifest": json.loads(args.config.read_text()),
        "experiments": {},
    }

    for engine in ("realvis", "zimage"):
        subset = [
            record for record in records
            if record["set"] == "paired_backgrounds" and record["engine"] == engine
        ]
        names, centroids = prompt_centroids(subset, matrix)
        centroid_pairs = pair_rows(names, centroids)
        write_pairs(args.output / f"pairwise_paired_backgrounds_{engine}_centroids.csv", centroid_pairs)
        within = []
        for name in names:
            same = [record for record in subset if record["id"] == name]
            pairs = pair_rows(
                [f"{name}-seed-{record['seed']}" for record in same],
                matrix[[record["index"] for record in same]],
            )
            within.extend(pairs)
        metrics["experiments"].setdefault("paired_backgrounds", {})[engine] = {
            "prompt_centroids": summarize(centroid_pairs),
            "same_profile_across_seeds": summarize(within),
        }

        count_subset = [
            record for record in records
            if record["set"] == "detail_count" and record["engine"] == engine
        ]
        count_names, count_centroids = prompt_centroids(count_subset, matrix)
        centroid_by_name = dict(zip(count_names, count_centroids))
        baseline = centroid_by_name["count_0"]
        movement = {}
        for count in (1, 2, 4, 8):
            name = f"count_{count}"
            matched = []
            for seed in metrics["prompt_manifest"]["detail_count"]["seeds"]:
                a = next(record for record in count_subset if record["id"] == "count_0" and record["seed"] == seed)
                b = next(record for record in count_subset if record["id"] == name and record["seed"] == seed)
                similarity = float(matrix[a["index"]] @ matrix[b["index"]])
                matched.append(similarity)
            movement[name] = {
                "centroid_cosine_to_zero_details": float(baseline @ centroid_by_name[name]),
                "centroid_cosine_distance_to_zero_details": float(1.0 - baseline @ centroid_by_name[name]),
                "matched_seed_mean_cosine_to_zero_details": float(np.mean(matched)),
                "matched_seed_min_cosine_to_zero_details": float(np.min(matched)),
            }
        metrics["experiments"].setdefault("detail_count", {})[engine] = movement

    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    sheets = args.output / "contact-sheets"
    profile_matrix(
        metrics["prompt_manifest"],
        image_records,
        args.images,
        sheets / "paired-backgrounds-all.jpg",
        5200,
    )
    profile_matrix(
        metrics["prompt_manifest"],
        image_records,
        args.images,
        sheets / "paired-backgrounds.jpg",
        5200,
        profile_ids=README_PROFILE_IDS,
    )
    count_matrix(image_records, args.images, sheets / "detail-count.jpg", 5300)
    print(json.dumps(metrics["experiments"], indent=2))


if __name__ == "__main__":
    main()
