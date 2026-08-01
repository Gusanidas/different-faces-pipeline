#!/usr/bin/env python3
"""Mint NEW identities for one demographic by sampling the cluster's ArcFace subspace
and decoding with Arc2Face.

Model of the demographic (antelopev2 space, from the 80 bank keepers):
  mu   = cluster mean          -> what makes it "Spanish man, 30s" (held)
  PCA  = within-cluster axes   -> what makes an individual (sampled)
Candidates are sampled ON THE SHELL of the empirical radius distribution (p40..p95 of
keeper deviations) — never inside it — to avoid the known midpoints->average collapse.

Acceptance is on DECODED IMAGES, not vectors:
  1. pre-gate in antelopev2 space vs keepers + accepted (cheap resample)
  2. decode PROBE_SEEDS images with Arc2Face
  3. intra-consistency: mean buffalo_l sim among probes >= INTRA_MIN  (it's one person)
  4. gate: max buffalo_l sim vs 80 bank keepers AND accepted minted < GATE  (a new person)
  5. genderage sanity from antelopev2 detector
Accepted identities then render FINAL_SEEDS more images.

Env: RTN_DEMO (es_m), RTN_TARGET (40), RTN_GATE (0.5)
Inputs:  /workspace/mint/<demo>/keep_dual.npz   (from embed_dual.py over the 80 keepers)
Outputs: /workspace/mint/<demo>/out/<mintNN>_s<seed>.png + mint_report.json + accepted npz
Run in the Arc2Face venv.
"""
import os, sys, json
import numpy as np
import torch
from PIL import Image
sys.path.append('/workspace/Arc2Face')
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from arc2face import CLIPTextModelWrapper, project_face_embs
from insightface.app import FaceAnalysis

DEMO   = os.environ.get("RTN_DEMO", "es_m")
TARGET = int(os.environ.get("RTN_TARGET", "40"))
GATE   = float(os.environ.get("RTN_GATE", "0.5"))
EXPECT_SEX = DEMO.rsplit("_", 1)[-1].upper()          # 'M' / 'F'
BASE   = f"/workspace/mint/{DEMO}"
OUT    = f"{BASE}/out"
os.makedirs(OUT, exist_ok=True)

PROBE_SEEDS = [1000, 1001, 1002]
FINAL_SEEDS = [1003, 1004, 1005]
STEPS, GUID = 25, 3.0
INTRA_MIN   = 0.60
MAX_CANDIDATES = 400
RNG = np.random.default_rng(4242)

device, dtype = "cuda", torch.float16

# ---------- demographic subspace model ----------
d = np.load(f"{BASE}/keep_dual.npz", allow_pickle=True)
A  = d["emb_ant"].astype("float64")     # (80,512) antelopev2, normalized
B  = d["emb_buf"].astype("float32")     # (80,512) buffalo_l,  normalized (gate refs)
mu = A.mean(0)
X  = A - mu
U, S, Vt = np.linalg.svd(X, full_matrices=False)      # Vt: (79,512) PCs
lam = (S**2) / (len(A) - 1)
radii = np.linalg.norm(X, axis=1)
r_lo, r_hi = np.percentile(radii, 40), np.percentile(radii, 95)
sim_keep = A @ A.T
np.fill_diagonal(sim_keep, -1)
# keepers are farthest-point picks -> their pairwise sims are artificially low; floor at 0.45
pre_gate = max(float(np.percentile(sim_keep[sim_keep > -1], 95)), 0.45)
print(f"[{DEMO}] keepers={len(A)} |mu|={np.linalg.norm(mu):.3f} "
      f"radius p40/p95={r_lo:.3f}/{r_hi:.3f} ant-space pairwise p95={pre_gate:.3f} "
      f"var@10PC={lam[:10].sum()/lam.sum():.2f}", flush=True)

def sample_candidate():
    z = RNG.standard_normal(len(lam)) * np.sqrt(lam)
    dev = z @ Vt
    dev *= RNG.uniform(r_lo, r_hi) / np.linalg.norm(dev)      # shell, not ball
    e = mu + dev
    return (e / np.linalg.norm(e)).astype("float32")

# ---------- models ----------
ant = FaceAnalysis(name="antelopev2", root="/workspace/Arc2Face",
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
ant.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
buf = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
buf.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

encoder = CLIPTextModelWrapper.from_pretrained('/workspace/Arc2Face/models', subfolder="encoder", torch_dtype=dtype)
unet = UNet2DConditionModel.from_pretrained('/workspace/Arc2Face/models', subfolder="arc2face", torch_dtype=dtype)
pipe = StableDiffusionPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5',
        text_encoder=encoder, unet=unet, torch_dtype=dtype, safety_checker=None)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True)

def decode(e512, seed):
    t = torch.tensor(e512, dtype=dtype)[None].to(device)
    t = t / torch.norm(t, dim=1, keepdim=True)
    pe = project_face_embs(pipe, t)
    g = torch.Generator(device=device).manual_seed(seed)
    return pipe(prompt_embeds=pe, num_inference_steps=STEPS, guidance_scale=GUID,
                num_images_per_prompt=1, generator=g).images[0]

def largest(fs):
    return sorted(fs, key=lambda x: (x["bbox"][2]-x["bbox"][0])*(x["bbox"][3]-x["bbox"][1]))[-1]

def pad(arr, frac=0.6):
    """Arc2Face frames faces full-frame; SCRFD misses those. Replicate-pad before retry."""
    import cv2
    p = int(max(arr.shape[:2]) * frac)
    return cv2.copyMakeBorder(arr, p, p, p, p, cv2.BORDER_REPLICATE)

def analyze(img):
    """-> (emb_buf, emb_ant, sex, age) or None if no face."""
    arr = np.ascontiguousarray(np.array(img)[:, :, ::-1])
    fb, fa = buf.get(arr), ant.get(arr)
    if not fb or not fa:
        arr = pad(arr)
        fb, fa = buf.get(arr), ant.get(arr)
    if not fb or not fa: return None
    b, a = largest(fb), largest(fa)
    eb = b["embedding"].astype("float32"); eb /= np.linalg.norm(eb)
    ea = a["embedding"].astype("float32"); ea /= np.linalg.norm(ea)
    return eb, ea, ("M" if a.get("gender", 1) == 1 else "F"), int(a.get("age", -1))

# ---------- mint loop ----------
accepted = []            # dicts: name, emb_target(ant), emb_buf(mean probe), stats
acc_buf, acc_ant = [], []   # gate refs of accepted (buffalo_l probe means / ant targets)
report = {"demo": DEMO, "gate": GATE, "pre_gate": pre_gate,
          "radius": [float(r_lo), float(r_hi)], "candidates": []}
n_cand = 0

while len(accepted) < TARGET and n_cand < MAX_CANDIDATES:
    n_cand += 1
    e = sample_candidate()
    pre = max(float((A.astype("float32") @ e).max()),
              float(max((np.stack(acc_ant) @ e).max(), -1) if acc_ant else -1))
    if pre > pre_gate:
        report["candidates"].append({"n": n_cand, "stage": "pre_gate", "pre": pre}); continue

    imgs, embs_b, realiz, sexes, ages = [], [], [], [], []
    for s in PROBE_SEEDS:
        img = decode(e, s)
        r = analyze(img)
        if r is None: break
        eb, ea, sex, age = r
        imgs.append((s, img)); embs_b.append(eb)
        realiz.append(float(ea @ e)); sexes.append(sex); ages.append(age)
    rec = {"n": n_cand, "pre": pre}
    if len(imgs) < len(PROBE_SEEDS):
        rec["stage"] = "no_face"; report["candidates"].append(rec); continue

    Eb = np.stack(embs_b)
    S2 = Eb @ Eb.T; iu = np.triu_indices(len(Eb), 1)
    intra = float(S2[iu].mean())
    gate_refs = np.concatenate([B] + ([np.stack(acc_buf)] if acc_buf else []))
    gmax = float((gate_refs @ Eb.T).max())
    rec.update(intra=round(intra, 3), gate_max=round(gmax, 3),
               realization=round(float(np.mean(realiz)), 3), sex="".join(sexes),
               age=round(float(np.mean(ages)), 1))
    if intra < INTRA_MIN:            rec["stage"] = "intra"
    elif gmax >= GATE:               rec["stage"] = "gate"
    elif sexes.count(EXPECT_SEX) < 2: rec["stage"] = "sex"
    else:
        rec["stage"] = "ACCEPT"
        name = f"mint{len(accepted):02d}"
        rec["name"] = name
        for s, img in imgs:
            img.save(f"{OUT}/{name}_s{s}.png")
        for s in FINAL_SEEDS:
            decode(e, s).save(f"{OUT}/{name}_s{s}.png")
        accepted.append({"name": name, **{k: rec[k] for k in
                        ("intra", "gate_max", "realization", "age")}})
        acc_buf.append(Eb.mean(0) / np.linalg.norm(Eb.mean(0)))
        acc_ant.append(e)
    report["candidates"].append(rec)
    print(f"[{n_cand:03d}] {rec['stage']:>8}  pre={pre:.3f} intra={rec.get('intra','-')} "
          f"gate={rec.get('gate_max','-')} realiz={rec.get('realization','-')} "
          f"sex={rec.get('sex','-')} age={rec.get('age','-')}  ({len(accepted)}/{TARGET})", flush=True)

report["accepted"] = accepted
report["n_candidates"] = n_cand
with open(f"{BASE}/mint_report.json", "w") as f:
    json.dump(report, f, indent=1)
if acc_ant:
    np.savez(f"{BASE}/minted.npz",
             names=np.array([a["name"] for a in accepted]),
             emb_ant_target=np.stack(acc_ant), emb_buf_probe=np.stack(acc_buf))
print(f"MINT_DONE accepted={len(accepted)} candidates={n_cand}", flush=True)
