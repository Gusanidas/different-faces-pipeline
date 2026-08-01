#!/usr/bin/env python3
"""Embed a directory in two spaces, checkpointing progress and per-file errors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis


def build_analyzer(name: str, root: str | None = None) -> FaceAnalysis:
    kwargs = {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
    if root:
        kwargs["root"] = root
    analyzer = FaceAnalysis(name=name, **kwargs)
    analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
    return analyzer


def largest_face(faces):
    return max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))


def error_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.errors.json")


def atomic_save_npz(output: Path, **payload) -> None:
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    temporary.replace(output)


def save_checkpoint(
    output: Path,
    source_root: Path,
    files: list[str],
    judge_embeddings: list[np.ndarray],
    conditioner_embeddings: list[np.ndarray],
    raw_norms: list[float],
    scores: list[float],
    errors: dict[str, str],
) -> None:
    if files:
        atomic_save_npz(
            output,
            source_root=np.asarray(str(source_root.resolve())),
            files=np.asarray(files),
            emb_buf=np.stack(judge_embeddings),
            emb_ant=np.stack(conditioner_embeddings),
            ant_norms=np.asarray(raw_norms, dtype=np.float32),
            detection_scores=np.asarray(scores, dtype=np.float32),
        )
    path = error_path(output)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(errors, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_checkpoint(output: Path, expected_source_root: Path):
    data = np.load(output, allow_pickle=True)
    required = ("files", "emb_buf", "emb_ant", "ant_norms", "detection_scores")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"existing checkpoint is missing arrays: {', '.join(missing)}")
    if "source_root" in data.files:
        stored_root = Path(str(data["source_root"].item())).resolve()
        if stored_root != expected_source_root.resolve():
            raise ValueError(
                f"existing checkpoint belongs to {stored_root}, not {expected_source_root.resolve()}"
            )
    row_count = len(data["files"])
    for key in required[1:]:
        if len(data[key]) != row_count:
            raise ValueError(f"existing checkpoint array {key!r} is misaligned")
    return (
        data["files"].astype(str).tolist(),
        list(data["emb_buf"]),
        list(data["emb_ant"]),
        data["ant_norms"].astype(float).tolist(),
        data["detection_scores"].astype(float).tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iid-root", default="/workspace/iid")
    parser.add_argument("--min-detection-score", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    judge = build_analyzer("buffalo_l")
    conditioner = build_analyzer("antelopev2", args.iid_root)
    paths = sorted(
        path for path in args.image_dir.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )

    has_checkpoint = args.resume and args.output.exists()
    if has_checkpoint:
        files, judge_embeddings, conditioner_embeddings, raw_norms, scores = load_checkpoint(
            args.output, args.image_dir
        )
    else:
        files, judge_embeddings, conditioner_embeddings, raw_norms, scores = [], [], [], [], []
        if not args.resume:
            args.output.unlink(missing_ok=True)
            error_path(args.output).unlink(missing_ok=True)
    errors = {}
    if has_checkpoint and error_path(args.output).exists():
        errors = json.loads(error_path(args.output).read_text())
    successful = set(files)
    processed_since_checkpoint = 0

    try:
        for path in paths:
            relative = str(path.relative_to(args.image_dir))
            if relative in successful:
                continue
            try:
                rgb = np.asarray(Image.open(path).convert("RGB"))
                bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                judge_faces = judge.get(bgr)
                conditioner_faces = conditioner.get(bgr)
                if not judge_faces or not conditioner_faces:
                    raise ValueError("face was not detected in both embedding spaces")
                judge_face = largest_face(judge_faces)
                conditioner_face = largest_face(conditioner_faces)
                if float(judge_face.det_score) < args.min_detection_score:
                    raise ValueError(
                        f"detection score {float(judge_face.det_score):.3f} is below "
                        f"{args.min_detection_score:.3f}"
                    )
                judge_raw = judge_face.embedding.astype(np.float32)
                conditioner_raw = conditioner_face.embedding.astype(np.float32)
                judge_norm = float(np.linalg.norm(judge_raw))
                conditioner_norm = float(np.linalg.norm(conditioner_raw))
                if judge_norm <= 1e-12:
                    raise ValueError("judge embedding has zero norm")
                if conditioner_norm <= 1e-12:
                    raise ValueError("conditioning embedding has zero norm")
                files.append(relative)
                judge_embeddings.append(judge_raw / judge_norm)
                conditioner_embeddings.append(conditioner_raw / conditioner_norm)
                raw_norms.append(conditioner_norm)
                scores.append(float(judge_face.det_score))
                successful.add(relative)
                errors.pop(relative, None)
            except Exception as error:  # continue a long batch; details are persisted below
                errors[relative] = f"{type(error).__name__}: {error}"
                print(f"warning: {relative}: {errors[relative]}", file=sys.stderr, flush=True)
            processed_since_checkpoint += 1
            if processed_since_checkpoint >= args.checkpoint_every:
                save_checkpoint(
                    args.output, args.image_dir, files, judge_embeddings,
                    conditioner_embeddings, raw_norms, scores, errors,
                )
                processed_since_checkpoint = 0
    finally:
        save_checkpoint(
            args.output, args.image_dir, files, judge_embeddings,
            conditioner_embeddings, raw_norms, scores, errors,
        )

    if not files:
        raise RuntimeError("no usable faces were detected")
    print(f"embedded {len(files)}/{len(paths)} faces -> {args.output}; errors={len(errors)}")


if __name__ == "__main__":
    main()
