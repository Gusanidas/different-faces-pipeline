# Two-detail character backgrounds

This experiment tests a concise version of the character-background idea. A
large library supplies short biographical details, but each generated person
receives exactly two. The demographic lock, portrait framing, seeds, and model
settings remain fixed.

Two controlled comparisons are generated with both RealVisXL and Z-Image:

1. Sixteen two-detail profiles, each rendered with the same three seeds.
2. One cumulative profile rendered with 0, 1, 2, 4, and 8 details using the
   same four seeds. This is a prompt-length and attention-dilution check, not a
   proposed production setting.

The details are intentionally biographical rather than facial. Any movement
in face identity therefore comes from the model's learned associations, not
from directly requesting a nose, jaw, or eye shape.

## Run

On a ComfyUI machine prepared by `../vast_reproduction/setup_vast.sh`:

```bash
python generate.py prompts.json
python measure.py \
  --config prompts.json \
  --images /workspace/ComfyUI/output/background_pairs \
  --output /workspace/background-pairs/results
python tokenize_prompts.py prompts.json \
  --output /workspace/background-pairs/results/prompt_tokens.json
```

`generate.py` records every submitted prompt and resumes only jobs confirmed
successful by ComfyUI. `measure.py` stores the detected-face embeddings,
pairwise tables, summaries, and compact comparison sheets. The tokenization
preflight requires the `tokenization` package extra. It checks both the
experiment prompts and all 8,192 prompts that the production formatter can emit
for each supported demographic; it fails if either SDXL CLIP tokenizer exceeds
one 77-token window.
