#!/usr/bin/env python3
"""Decode keeper and sampled glintr100 vectors with InstantID + RealVisXL.

Identity comes from the vector, the shared look comes from the demographic
prompt, and pose comes from explicit keypoint images. Arc2Face is deliberately
absent from the production path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel
from insightface.app import FaceAnalysis
from PIL import Image

from different_pipeline_generator.demographics import DEMOGRAPHICS
from different_pipeline_generator.naming import ensure_unique


NEGATIVE = (
    "beautiful, model, attractive, airbrushed, flawless skin, glamorous, cartoon, illustration, "
    "anime, 3d render, cgi, deformed, watermark, text, duplicate, two people, multiple faces"
)


def largest_face(faces):
    return max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))


def faces_with_retry(image: np.ndarray | None, analyzer: FaceAnalysis):
    if image is None:
        return []
    faces = analyzer.get(image)
    if faces:
        return faces
    padding = int(max(image.shape[:2]) * 0.6)
    padded = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    return analyzer.get(padded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--demographic", choices=tuple(DEMOGRAPHICS), required=True)
    parser.add_argument("--instantid-repo", type=Path, default=Path("/workspace/InstantID"))
    parser.add_argument("--iid-root", default="/workspace/iid")
    parser.add_argument("--checkpoint", default="SG161222/RealVisXL_V5.0")
    parser.add_argument("--controlnet", default="/workspace/InstantID/checkpoints/ControlNetModel")
    parser.add_argument("--ip-adapter", default="/workspace/InstantID/checkpoints/ip-adapter.bin")
    parser.add_argument("--front-kps", type=Path, default=Path("/workspace/kps06.png"))
    parser.add_argument("--side-kps", type=Path, default=Path("/workspace/kps00.png"))
    parser.add_argument("--ip-scale", type=float, default=1.0)
    parser.add_argument("--controlnet-scale", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7000)
    args = parser.parse_args()

    sys.path.insert(0, str(args.instantid_repo))
    from pipeline_stable_diffusion_xl_instantid import (  # noqa: PLC0415
        StableDiffusionXLInstantIDPipeline,
        draw_kps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.candidates, allow_pickle=True)
    names = data["names"].astype(str)
    raw_vectors = data["raw"].astype(np.float32)
    if len(names) != len(raw_vectors):
        raise ValueError(
            f"candidate names/raw arrays are misaligned: {len(names)} vs {len(raw_vectors)}"
        )
    ensure_unique(names, label="candidate identity names")
    demo = DEMOGRAPHICS[args.demographic]
    base_prompt = (
        f"head-and-shoulders portrait, even soft light, candid snapshot, {demo.lock}, "
        "natural skin texture, plain neutral background"
    )
    variants = (
        ("front", args.front_kps, f"{base_prompt}, neutral expression"),
        ("side", args.side_kps, f"{base_prompt}, neutral expression"),
        ("wink", args.front_kps, f"{base_prompt}, playful wink, one eye closed, grinning"),
    )

    # This analyzer touches only the small pose-template images. Keeping it on
    # CPU avoids reserving another CUDA provider before the diffusion pipeline.
    analyzer = FaceAnalysis(
        name="antelopev2",
        root=args.iid_root,
        providers=["CPUExecutionProvider"],
    )
    analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
    pose_images = {}
    for variant, path, _ in variants:
        bgr = cv2.imread(str(path))
        faces = faces_with_retry(bgr, analyzer)
        if not faces:
            raise RuntimeError(f"no face found in keypoint template {path}")
        pose_images[variant] = draw_kps(
            Image.new("RGB", (1024, 1024)), largest_face(faces).kps
        )

    dtype = torch.float16
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    pipeline = StableDiffusionXLInstantIDPipeline.from_pretrained(
        args.checkpoint,
        controlnet=controlnet,
        torch_dtype=dtype,
        variant="fp16",
    ).to("cuda")
    pipeline.load_ip_adapter_instantid(args.ip_adapter)
    pipeline.set_ip_adapter_scale(args.ip_scale)
    pipeline.set_progress_bar_config(disable=True)

    for identity_index, (name, raw_vector) in enumerate(zip(names, raw_vectors), start=1):
        for variant, _, prompt in variants:
            output = args.output_dir / f"{name}_{variant}.png"
            if output.exists():
                continue
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            image = pipeline(
                prompt=prompt,
                negative_prompt=NEGATIVE,
                image_embeds=raw_vector,
                image=pose_images[variant],
                controlnet_conditioning_scale=args.controlnet_scale,
                num_inference_steps=30,
                guidance_scale=5.0,
                generator=generator,
            ).images[0]
            image.save(output)
        if identity_index % 10 == 0:
            print(f"rendered {identity_index}/{len(names)} identities", flush=True)


if __name__ == "__main__":
    main()
