#!/usr/bin/env python3
"""Judge decoded photos and run the final consistency-vs-distance tournament."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from different_pipeline_generator.naming import ensure_unique
from different_pipeline_generator.selection import tournament_indices, unit_rows


VARIANTS = ("front", "side", "wink")


def largest_face(faces):
    return max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))


def embedding_with_retry(path: Path, analyzer: FaceAnalysis) -> np.ndarray | None:
    if not path.is_file():
        return None
    image = cv2.imread(str(path))
    if image is None:
        return None
    faces = analyzer.get(image)
    if not faces:
        padding = int(max(image.shape[:2]) * 0.6)
        image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
        faces = analyzer.get(image)
    if not faces:
        return None
    return largest_face(faces).normed_embedding.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path, help="NPZ created by 04_sample_identity_cloud.py")
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    candidates = np.load(args.candidates, allow_pickle=True)
    all_names = candidates["names"].astype(str)
    all_origins = candidates["origins"].astype(str)
    if len(all_names) != len(all_origins):
        raise ValueError(
            f"candidate names/origins arrays are misaligned: {len(all_names)} vs {len(all_origins)}"
        )
    ensure_unique(all_names, label="candidate identity names")
    origin_by_name = dict(zip(all_names, all_origins))
    analyzer = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    names, origins, centroids, intra = [], [], [], []
    rejected = []
    for name in all_names:
        embeddings = [
            embedding_with_retry(args.image_dir / f"{name}_{variant}.png", analyzer)
            for variant in VARIANTS
        ]
        embeddings = [embedding for embedding in embeddings if embedding is not None]
        if len(embeddings) < 2:
            rejected.append(name)
            continue
        rows = unit_rows(np.stack(embeddings))
        pairwise = rows @ rows.T
        upper = pairwise[np.triu_indices(len(rows), 1)]
        centroid = rows.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        names.append(name)
        origins.append(origin_by_name[name])
        centroids.append(centroid)
        intra.append(float(upper.mean()))

    if not centroids:
        raise RuntimeError(
            "no identity had at least two detectable renders; check the render directory and face detector"
        )
    centroid_rows = np.stack(centroids)
    intra_values = np.asarray(intra)
    keep_indices = tournament_indices(centroid_rows, intra_values, min(args.top_k, len(names)))
    kept_names = [names[index] for index in keep_indices]
    kept_centroids = unit_rows(centroid_rows[keep_indices])
    kept_similarity = kept_centroids @ kept_centroids.T
    np.fill_diagonal(kept_similarity, -1)

    rankings = []
    for local_index, global_index in enumerate(keep_indices):
        max_other = float(kept_similarity[local_index].max())
        rankings.append(
            {
                "pid": names[global_index],
                "origin": origins[global_index],
                "intra": round(float(intra_values[global_index]), 4),
                "max_other": round(max_other, 4),
                "score": round(float(intra_values[global_index]) - max_other, 4),
            }
        )
    rankings.sort(key=lambda row: row["score"], reverse=True)
    origin_counts: dict[str, int] = {}
    for row in rankings:
        origin_counts[row["origin"]] = origin_counts.get(row["origin"], 0) + 1
    origin_totals: dict[str, int] = {}
    for origin in all_origins:
        origin_totals[origin] = origin_totals.get(origin, 0) + 1

    winners_dir = args.output_dir / "winners"
    winners_dir.mkdir(parents=True, exist_ok=True)
    for name in kept_names:
        for variant in VARIANTS:
            source = args.image_dir / f"{name}_{variant}.png"
            if source.exists():
                shutil.copy2(source, winners_dir / source.name)
    kept_name_set = set(kept_names)
    report = {
        "schema_version": "different-pipeline-generator/v1",
        "top_k": len(rankings),
        "origin_stats_top50": origin_counts,
        "origin_totals": origin_totals,
        "top50": rankings,
        "all": [
            {
                "pid": name,
                "origin": origin,
                "intra": round(float(consistency), 4),
                "kept": name in kept_name_set,
            }
            for name, origin, consistency in zip(names, origins, intra)
        ],
        "rejected_for_missing_faces": rejected,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(origin_counts, indent=2))


if __name__ == "__main__":
    main()
