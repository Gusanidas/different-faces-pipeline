"""Naming safeguards for artifacts passed between pipeline stages."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import quote


_COMFY_COUNTER = re.compile(r"_\d{5}_?$")


def clean_comfy_name(filename: str) -> str:
    """Remove the output counter while retaining collision-safe path context."""
    path = Path(filename)
    clean_stem = _COMFY_COUNTER.sub("", path.stem).rstrip("_")
    clean_path = path.with_name(clean_stem)
    return quote(clean_path.as_posix(), safe="._-")


def ensure_unique(values: Iterable[str], *, label: str = "values") -> None:
    """Raise with all duplicate values instead of allowing silent collisions."""
    counts = Counter(values)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(repr(value) for value in duplicates[:8])
        suffix = " ..." if len(duplicates) > 8 else ""
        raise ValueError(f"duplicate {label}: {preview}{suffix}")
