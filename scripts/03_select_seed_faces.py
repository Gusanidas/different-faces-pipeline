#!/usr/bin/env python3
"""Build an ArcFace-diverse shortlist without deleting the raw pool."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from different_pipeline_generator.selection import farthest_point_indices


def pairwise_cosine_summary(rows: np.ndarray) -> tuple[float, float] | None:
    """Return maximum and mean pair cosine, or ``None`` for one face."""
    if len(rows) < 2:
        return None
    unit = rows.astype(np.float64)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    similarity = unit @ unit.T
    upper = similarity[np.triu_indices(len(unit), 1)]
    return float(upper.max()), float(upper.mean())


def selected_payload(data, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Validate row-aligned embedding arrays and select the requested rows."""
    required = ("files", "emb_buf", "emb_ant", "ant_norms")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"embedding archive is missing required arrays: {', '.join(missing)}")
    row_count = len(data["files"])
    payload = {}
    for key in (*required, "detection_scores"):
        if key not in data.files:
            continue
        value = data[key]
        if value.ndim == 0 or len(value) != row_count:
            raise ValueError(
                f"array {key!r} must have {row_count} rows, got shape {value.shape}"
            )
        payload[key] = value[indices]
    payload["selected_indices"] = indices
    return payload


def copy_selected_images(
    image_dir: Path,
    output_dir: Path,
    files: np.ndarray,
    indices: np.ndarray,
) -> None:
    """Copy selected images without flattening their relative paths."""
    for index in indices:
        relative = Path(str(files[index]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"image path must stay below the image directory: {relative}")
        source = image_dir / relative
        if not source.is_file():
            raise FileNotFoundError(f"selected source image does not exist: {source}")
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--top-k",
        type=int,
        default=80,
        help="ArcFace shortlist size; the human curation stage defaults to 40",
    )
    args = parser.parse_args()

    data = np.load(args.embeddings, allow_pickle=True)
    indices = farthest_point_indices(data["emb_buf"], args.top_k)
    files = data["files"].astype(str)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    copy_selected_images(args.image_dir, args.output_dir, files, indices)
    selected_path = args.output_dir / "selected.npz"
    np.savez(selected_path, **selected_payload(data, indices))
    (args.output_dir / "shortlist.txt").write_text(
        "".join(f"{files[index]}\n" for index in indices)
    )

    pairwise = pairwise_cosine_summary(data["emb_buf"][indices])
    if pairwise is None:
        pairwise_text = "pairwise cosine unavailable for a one-face shortlist"
    else:
        maximum, mean = pairwise
        pairwise_text = f"max pair cosine={maximum:.3f}, mean={mean:.3f}"
    print(f"shortlisted {len(indices)} faces -> {args.output_dir}; {pairwise_text}")


if __name__ == "__main__":
    main()
