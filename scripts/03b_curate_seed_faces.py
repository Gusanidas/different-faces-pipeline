#!/usr/bin/env python3
"""Package the human-selected seed bank from an ArcFace shortlist.

The selection manifest is a UTF-8 text file with one shortlisted filename per
line. Lines may contain the exact stored relative path or, when unambiguous, its
basename. Blank lines and lines beginning with ``#`` are ignored.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from different_pipeline_generator.naming import ensure_unique


ROW_ARRAYS = ("files", "emb_buf", "emb_ant", "ant_norms", "detection_scores")


def read_selection(path: Path) -> list[str]:
    """Read the ordered, comment-friendly human selection manifest."""
    entries = []
    for raw_line in path.read_text().splitlines():
        value = raw_line.strip()
        if value and not value.startswith("#"):
            entries.append(value)
    ensure_unique(entries, label="curated selection entries")
    return entries


def resolve_selection(files: np.ndarray, entries: list[str]) -> np.ndarray:
    """Resolve exact stored paths or unique basenames to shortlist rows."""
    stored = [str(value) for value in files]
    exact = {value: index for index, value in enumerate(stored)}
    by_basename: dict[str, list[int]] = {}
    for index, value in enumerate(stored):
        by_basename.setdefault(Path(value).name, []).append(index)

    resolved = []
    for entry in entries:
        if entry in exact:
            resolved.append(exact[entry])
            continue
        matches = by_basename.get(Path(entry).name, [])
        if not matches:
            raise ValueError(f"curated file is not in the shortlist: {entry}")
        if len(matches) > 1:
            choices = ", ".join(stored[index] for index in matches)
            raise ValueError(f"ambiguous curated basename {entry!r}: {choices}")
        resolved.append(matches[0])

    if len(set(resolved)) != len(resolved):
        raise ValueError("curated selection resolves to duplicate shortlist rows")
    return np.asarray(resolved, dtype=np.int64)


def curated_payload(data, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Select every row-aligned array while preserving raw-pool provenance."""
    missing = [key for key in ROW_ARRAYS[:4] if key not in data.files]
    if missing:
        raise ValueError(f"shortlist archive is missing required arrays: {', '.join(missing)}")
    row_count = len(data["files"])
    payload: dict[str, np.ndarray] = {}
    for key in ROW_ARRAYS:
        if key not in data.files:
            continue
        value = data[key]
        if value.ndim == 0 or len(value) != row_count:
            raise ValueError(
                f"array {key!r} must have {row_count} rows, got shape {value.shape}"
            )
        payload[key] = value[indices]

    if "selected_indices" not in data.files:
        raise ValueError(
            "shortlist archive is missing 'selected_indices'; raw-pool provenance "
            "cannot be reconstructed from shortlist row numbers"
        )
    source_indices = data["selected_indices"]
    if source_indices.ndim == 0 or len(source_indices) != row_count:
        raise ValueError(
            "array 'selected_indices' must be aligned with the shortlist rows"
        )
    payload["selected_indices"] = source_indices[indices]
    payload["shortlist_indices"] = indices
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shortlist", type=Path, help="selected.npz from Stage 3")
    parser.add_argument("image_dir", type=Path, help="root of the original 500-image pool")
    parser.add_argument("selection", type=Path, help="one human-selected filename per line")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-count", type=int, default=40)
    args = parser.parse_args()
    if args.expected_count < 1:
        parser.error("--expected-count must be positive")

    entries = read_selection(args.selection)
    if len(entries) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} human-selected faces, got {len(entries)}"
        )

    with np.load(args.shortlist, allow_pickle=True) as data:
        files = data["files"].astype(str)
        indices = resolve_selection(files, entries)
        payload = curated_payload(data, indices)

    chosen_files = payload["files"].astype(str)
    destination_names = [Path(value).name for value in chosen_files]
    ensure_unique(destination_names, label="curated destination filenames")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stored, destination_name in zip(chosen_files, destination_names):
        source = args.image_dir / stored
        if not source.is_file():
            raise FileNotFoundError(f"curated source image does not exist: {source}")
        shutil.copy2(source, args.output_dir / destination_name)

    output = args.output_dir / "selected.npz"
    np.savez(output, **payload)
    metadata = {
        "shortlist": str(args.shortlist),
        "selection_manifest": str(args.selection),
        "shortlist_count": int(len(files)),
        "curated_count": int(len(indices)),
        "files": chosen_files.tolist(),
    }
    (args.output_dir / "curation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"packaged {len(indices)} human-selected seed faces -> {args.output_dir}")


if __name__ == "__main__":
    main()
