#!/usr/bin/env python3
"""Can InstantID decode SYNTHETIC identity vectors? Subspace-sample glintr100 space
directly (no Arc2Face) and render with InstantID+RealVisXL @1024.

Sample sets (10 identities each, same shell method as mint_subspace.py):
  base  radius p40-p95, covariance temperature 1.0
  far   radius p60-p99, temperature 1.0          (more distinct, more drift risk)
  hot   radius p40-p95, intended temperature 1.3 (see historical erratum below)
Configs: A=(base, ip1.3)  B=(base, ip0.9)  C=(far, ip1.3)  D=(hot, ip1.3); cn=0.7.
A vs B isolates ip scale on the SAME vectors; C varies the shell.

HISTORICAL ERRATUM (2026-07-29): ``temp`` is a no-op because line 68
renormalizes every deviation to an independently sampled shell radius. The hot
and base samplers therefore have the same distribution; their realized vectors
differ only because they consumed different sections of the shared RNG stream.
Configuration D cannot support a covariance-temperature conclusion. The code is
retained unchanged below so the original experiment remains exactly replayable.
Per identity: 2 poses x 2 seeds = 4 images. 160 images total, no accept/reject —
measure everything: realization (glintr100 out vs target), intra (buffalo), gate vs
bank+siblings (buffalo), gender/age.

Inputs: /workspace/keep_glint.npz (embed_dual.py with root=/workspace/iid),
        /workspace/kps00.png /workspace/kps06.png
Out:    /workspace/sample_out/<cfg>_<id>_<pose>_s<seed>.png + sample_report.json
Run in /workspace/.venv.
"""
import os, sys, glob, json, itertools
import numpy as np
import cv2
import torch
from PIL import Image
sys.path.append('/workspace/InstantID')
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from diffusers import ControlNetModel
from insightface.app import FaceAnalysis

OUT = "/workspace/sample_out"
os.makedirs(OUT, exist_ok=True)
POSES = ["kps00", "kps06"]
SEEDS = [7000, 7001]
FRAME = "frontal head-and-shoulders portrait, neutral expression, even soft light, candid snapshot"
LOCK = "a Spanish man in his 30s, short dark hair, tanned skin, natural skin texture, plain neutral background"
PROMPT = f"{FRAME}, {LOCK}"
NEG = ("beautiful, model, attractive, airbrushed, flawless skin, symmetrical face, glamorous, "
       "heavy makeup, cartoon, illustration, anime, 3d render, cgi, deformed, disfigured, "
       "watermark, text, duplicate, two people, multiple faces")
N_PER = 10
RNG = np.random.default_rng(5150)
device, dtype = "cuda", torch.float16

# ---------- subspace model in glintr100 space ----------
d = np.load("/workspace/keep_glint.npz", allow_pickle=True)
A = d["emb_ant"].astype("float64")          # 80x512 glintr100, unit-norm
B = d["emb_buf"].astype("float32")          # 80x512 buffalo_l (gate refs)
RAW_NORM = float(d["ant_norms"].mean())     # rescale samples to raw scale for InstantID
mu = A.mean(0); X = A - mu
U, S, Vt = np.linalg.svd(X, full_matrices=False)
lam = (S**2) / (len(A) - 1)
radii = np.linalg.norm(X, axis=1)
simk = A @ A.T; np.fill_diagonal(simk, -1)
pre_gate = max(float(np.percentile(simk[simk > -1], 95)), 0.45)
print(f"keepers=80 |mu|={np.linalg.norm(mu):.3f} raw_norm={RAW_NORM:.1f} pre_gate={pre_gate:.3f}", flush=True)

def sample_set(name, p_lo, p_hi, temp, n=N_PER):
    r_lo, r_hi = np.percentile(radii, p_lo), np.percentile(radii, p_hi)
    out, tries = [], 0
    while len(out) < n and tries < 400:
        tries += 1
        z = RNG.standard_normal(len(lam)) * np.sqrt(lam) * temp  # historical no-op; see erratum
        dev = z @ Vt
        dev *= RNG.uniform(r_lo, r_hi) / np.linalg.norm(dev)
        e = mu + dev; e /= np.linalg.norm(e)
        others = np.concatenate([A, np.stack(out)]) if out else A
        if float((others @ e).max()) > pre_gate: continue
        out.append(e.astype("float32"))
    print(f"set {name}: {len(out)} vectors in {tries} tries (shell {r_lo:.3f}-{r_hi:.3f}, temp {temp})", flush=True)
    return out

sets = {"base": sample_set("base", 40, 95, 1.0),
        "far":  sample_set("far", 60, 99, 1.0),
        "hot":  sample_set("hot", 40, 95, 1.3)}
CONFIGS = [("A", "base", 1.3), ("B", "base", 0.9), ("C", "far", 1.3), ("D", "hot", 1.3)]
CN = 0.7

# ---------- models ----------
app = FaceAnalysis(name="antelopev2", root="/workspace/iid", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
buf = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
buf.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

def faces_of_arr(img, ana):
    fs = ana.get(img)
    if not fs:
        p = int(max(img.shape[:2]) * 0.6)
        fs = ana.get(cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REPLICATE))
    return fs

def largest(fs):
    return max(fs, key=lambda x: (x["bbox"][2]-x["bbox"][0])*(x["bbox"][3]-x["bbox"][1]))

kps_imgs = {}
for k in POSES:
    img = cv2.imread(f"/workspace/{k}.png")
    kps_imgs[k] = draw_kps(Image.new("RGB", (1024, 1024)), largest(faces_of_arr(img, app))["kps"])

controlnet = ControlNetModel.from_pretrained("/workspace/InstantID/checkpoints/ControlNetModel", torch_dtype=dtype)
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0", controlnet=controlnet, torch_dtype=dtype, variant="fp16").to(device)
pipe.load_ip_adapter_instantid("/workspace/InstantID/checkpoints/ip-adapter.bin")
pipe.set_progress_bar_config(disable=True)

# ---------- generate + measure ----------
report = {"pre_gate": pre_gate, "raw_norm": RAW_NORM, "configs": {}}
for cfg, set_name, ip in CONFIGS:
    pipe.set_ip_adapter_scale(ip)
    idents = []
    for i, e in enumerate(sets[set_name]):
        raw = e * RAW_NORM
        embs_b, realiz, sexes, ages = [], [], [], []
        for k in POSES:
            for s in SEEDS:
                g = torch.Generator(device=device).manual_seed(s)
                img = pipe(prompt=PROMPT, negative_prompt=NEG, image_embeds=raw,
                           image=kps_imgs[k], controlnet_conditioning_scale=CN,
                           num_inference_steps=30, guidance_scale=5.0, generator=g).images[0]
                img.save(f"{OUT}/{cfg}_{i:02d}_{k}_s{s}.png")
                arr = np.ascontiguousarray(np.array(img)[:, :, ::-1])
                fb, fa = faces_of_arr(arr, buf), faces_of_arr(arr, app)
                if fb and fa:
                    eb = largest(fb).normed_embedding.astype("float32")
                    fA = largest(fa)
                    ea = fA["embedding"].astype("float32"); ea /= np.linalg.norm(ea)
                    embs_b.append(eb); realiz.append(float(ea @ e))
                    sexes.append("M" if fA.get("gender", 1) == 1 else "F")
                    ages.append(int(fA.get("age", -1)))
        rec = {"id": i, "n_faces": len(embs_b)}
        if len(embs_b) >= 2:
            E = np.stack(embs_b); Sm = E @ E.T; iu = np.triu_indices(len(E), 1)
            c = E.mean(0); c /= np.linalg.norm(c)
            rec.update(intra=round(float(Sm[iu].mean()), 3),
                       realization=round(float(np.mean(realiz)), 3),
                       gate_vs_bank=round(float((B @ c).max()), 3),
                       sex="".join(sexes), age=round(float(np.mean(ages)), 1))
            rec["centroid"] = c.tolist()
        idents.append(rec)
        print(f"[{cfg}{i:02d}] intra={rec.get('intra','-')} realiz={rec.get('realization','-')} "
              f"gate={rec.get('gate_vs_bank','-')} sex={rec.get('sex','-')}", flush=True)
    cents = [np.array(r["centroid"], dtype="float32") for r in idents if "centroid" in r]
    sib_max = None
    if len(cents) >= 2:
        Cm = np.stack(cents) @ np.stack(cents).T; np.fill_diagonal(Cm, -1)
        sib_max = round(float(Cm.max()), 3)
    for r in idents: r.pop("centroid", None)
    ok = [r for r in idents if "intra" in r]
    report["configs"][cfg] = {
        "set": set_name, "ip": ip,
        "intra_mean": round(float(np.mean([r["intra"] for r in ok])), 3) if ok else None,
        "realization_mean": round(float(np.mean([r["realization"] for r in ok])), 3) if ok else None,
        "gate_vs_bank_max": round(float(np.max([r["gate_vs_bank"] for r in ok])), 3) if ok else None,
        "sibling_max": sib_max, "identities": idents}
    print(f"== config {cfg} ({set_name}, ip{ip}): {report['configs'][cfg]['intra_mean']} intra, "
          f"{report['configs'][cfg]['realization_mean']} realiz, sib_max {sib_max}", flush=True)

with open("/workspace/sample_report.json", "w") as f:
    json.dump(report, f, indent=1)
print("SAMPLE_PROBE_DONE", flush=True)
