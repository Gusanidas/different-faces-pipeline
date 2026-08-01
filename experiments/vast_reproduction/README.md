# Vast.ai reproduction

This directory reproduces the prompt-only experiments used in the project
story. It records every prompt, seed, generated image, ArcFace-family
(`buffalo_l`) embedding, and pairwise comparison.

The controlled lock is held fixed throughout:

> a Spanish man in his early 30s, short medium-brown hair, tanned skin

Three interventions are compared:

1. **Seeds only:** one prompt, 36 unique seeds.
2. **Feature words:** six exact facial-feature prompts, each rendered with the
   same six seeds.
3. **Character backstories:** twelve professions and life histories, each
   rendered with the same four seeds in RealVisXL and Z-Image.

Using the same seed set for every prompt in an intervention reduces seed noise
when comparing prompt centroids. `measure.py` still reports the complete
image-level distributions as well as the centroid comparisons.

## Vast.ai run

The dated evidence in `data/reproduction-2026-07-29/` was generated on Vast.ai
instance `46212073`, an NVIDIA L40S. To repeat it on a fresh Ubuntu/PyTorch
instance:

```bash
bash setup_vast.sh
python generate.py prompts.json
python measure.py \
  --config prompts.json \
  --images /workspace/ComfyUI/output/reproduction_20260729 \
  --output /workspace/reproduction/results \
  --instance-id 46212073
```

`generate.py` persists ComfyUI prompt IDs and completion status in
`/workspace/reproduction/generation_state.json`, so interrupted runs resume.
`measure.py` writes `metrics.json`, `embeddings.npz`, all relevant pairwise CSV
files, and contact sheets.
