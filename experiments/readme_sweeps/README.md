# README comparison sweeps

This manifest reproduces the image pools used for the README's seed, facial-feature, and lens/framing comparisons.

- `seed_sweeps`: three prompts, 12 contiguous seeds, two generators (`72` images)
- `scandinavian_features`: six feature prompts, four shared seeds, two generators (`48` images)
- `lens_framing`: three camera descriptions, four shared seeds, two generators (`24` images)

The 144 original renders are review inputs. They remain outside the tracked repository until a human selects the examples that support the story. The selected examples belong in `assets/`; structured prompts, embeddings, measurements, and selection manifests belong in `data/`.

Run the same ComfyUI setup used by `experiments/vast_reproduction`, then:

```bash
python experiments/readme_sweeps/generate.py experiments/readme_sweeps/prompts.json
```

The generator records every submitted prompt, seed, model, ComfyUI prompt ID, and completion status in `generation_state.json`.

After generation, measure the rendered faces with ArcFace:

```bash
python experiments/readme_sweeps/measure.py \
  --config experiments/readme_sweeps/prompts.json \
  --images /path/to/readme_sweeps \
  --output /path/to/readme-sweep-data
```
