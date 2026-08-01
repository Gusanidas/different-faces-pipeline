# Measurements and manifests

This directory contains only the compact measurements needed to audit the
overview. Exact prompts and executable experiment definitions live under
`experiments/`; embeddings, pairwise tables, generation state, and full
generation pools are reproducible outputs and are intentionally omitted.

## Production reports

`es_m_prod_report.json` and `sc_f_prod_report.json` are the original 160-candidate tournament reports. Each contains:

- the sampled-shell radius and pre-gate;
- origin counts among the top 50;
- every winner's intra-person consistency, nearest surviving identity, and combined score;
- the full candidate list and winner flag.

They are included so the summary numbers in the main README can be audited directly.

These files predate the cleaned repository and do not contain a schema version. `scripts/06_rank_roster.py` preserves their main keys and adds `schema_version`, `top_k`, and `rejected_for_missing_faces`; exact bytes are not expected to match across model/runtime versions.

## Controlled reproduction

`reproduction-2026-07-29/metrics.json` contains model checksums, exact prompts,
seeds, summary statistics, and farthest selections for the controlled prompt
ablation. The included generation and measurement programs regenerate the
image-level embeddings and pairwise tables when those are needed.

## README comparison sweeps

`readme-sweeps/metrics.json` summarizes the two-model seed, Scandinavian
facial-feature, and lens/framing comparisons. The exact generation inputs and
analysis code live in `experiments/readme_sweeps/`.

## Two-detail backgrounds

`background-pairs/metrics.json` summarizes the two-detail background and
0/1/2/4/8-detail length experiments. The prompt manifest, tokenization step,
generation program, and analysis code live in `experiments/background_pairs/`;
the two compact visual matrices used in the overview live in
`assets/story/5-character-backgrounds/`.
