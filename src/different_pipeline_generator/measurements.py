"""Shared pairwise measurements for reproducible face experiments."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np


def pair_rows(names: list[str], rows: np.ndarray) -> list[dict]:
    """Return every unique pair and its cosine similarity and distance."""
    if len(names) != len(rows):
        raise ValueError(f"names and rows are misaligned: {len(names)} vs {len(rows)}")
    return [
        {
            "a": names[i],
            "b": names[j],
            "cosine_similarity": float(rows[i] @ rows[j]),
            "cosine_distance": float(1.0 - rows[i] @ rows[j]),
        }
        for i, j in combinations(range(len(rows)), 2)
    ]


def summarize_pairs(pairs: list[dict], *, include_p05: bool = False) -> dict:
    """Summarize pairwise cosine values with a stable empty-set result."""
    if not pairs:
        return {"pair_count": 0}
    similarities = np.asarray([row["cosine_similarity"] for row in pairs])
    summary = {
        "pair_count": len(pairs),
        "mean": float(similarities.mean()),
        "std": float(similarities.std()),
        "min": float(similarities.min()),
        "median": float(np.median(similarities)),
        "p95": float(np.percentile(similarities, 95)),
        "max": float(similarities.max()),
        "maximum_cosine_distance": float(1.0 - similarities.min()),
        "most_distant_pair": pairs[int(np.argmin(similarities))],
        "most_similar_pair": pairs[int(np.argmax(similarities))],
    }
    if include_p05:
        summary["p05"] = float(np.percentile(similarities, 5))
    return summary


def parse_render(path: Path, root: Path) -> dict | None:
    """Parse a three-level ComfyUI experiment render path."""
    relative = path.relative_to(root)
    if len(relative.parts) != 3 or "_s" not in relative.name:
        return None
    experiment, engine, filename = relative.parts
    identity, suffix = filename.split("_s", 1)
    try:
        seed = int(suffix.split("_", 1)[0])
    except ValueError:
        return None
    return {
        "path": str(relative),
        "set": experiment,
        "engine": engine,
        "id": identity,
        "seed": seed,
    }


def ensure_unique_renders(records: list[dict]) -> None:
    """Reject duplicate ComfyUI counters for one logical experiment render."""
    seen: dict[tuple[str, str, str, int], str] = {}
    for record in records:
        key = (record["set"], record["engine"], record["id"], record["seed"])
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                "duplicate logical render "
                f"{key}: {previous!r} and {record['path']!r}; remove the duplicate output"
            )
        seen[key] = record["path"]
