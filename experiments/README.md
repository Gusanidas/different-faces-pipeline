# Experiment records

These files preserve the inputs and measurements behind the overview. They are not the recommended production entry points.

- `prompt_feature_prompts.json` — the exact fixed-feature baseline and feature-slot prompts.
- `readme_sweeps/` — the two-model seed, Scandinavian feature, and lens/framing comparisons used by the overview.
- `vast_reproduction/` — the prompt ablation, including every seed, exact feature and character-backstory prompt, generation driver, and pairwise measurement code.
- `arc2face_decoder.py` — the superseded Arc2Face shell-sampling and decoded-image acceptance loop.
- `direct_decode_probe.py` — the experiment that proved InstantID could decode synthetic `glintr100` vectors directly.
- `ERRATA.md` — correction to the probe's invalid “hot covariance” interpretation.

The archived scripts preserve their original `/workspace/...` assumptions and behavior so the experiment record remains replayable. An inline notice and `ERRATA.md` correct the historical temperature claim. Use `scripts/01...06` for the cleaned pipeline.
