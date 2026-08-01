# Pipeline reference

## The production invariant

Three controls must remain independent:

| Concern | Control | Why |
|---|---|---|
| Common traits | one locked text prompt | every roster member should read as part of the same cohort |
| Identity | a stored raw-scale `glintr100` vector from a reviewed seed or accepted sample | text-conditioned identity collapses toward the checkpoint mean |
| Pose | an `image_kps` template | prompt phrases such as “side profile” lost against identity conditioning |

## Stage 1 — over-generate with two priors

`01_generate_seed_bank.py` holds the demographic lock fixed while varying four candidate-pool inputs: one concise facial-feature bundle, one lens/framing choice, exactly two short biographical details (an occupation and a personal detail), and several random seeds. A normal run submits 100 prompt combinations × 5 seeds for each model, producing about 500 RealVisXL and 500 Z-Image candidates:

- RealVisXL V5.0 at 1024 px, 30 steps, CFG 5; or
- Z-Image Turbo at 1024 px, 8 steps, CFG 1.

The exact production locks are defined once in `demographics.py`: Spanish men
in their early 30s with short medium-brown hair and tanned skin, and
Scandinavian women in their late 20s with long light-blonde hair, fair skin,
and blue eyes. The experiment token preflight enumerates the complete prompt
space from the same formatter and checks both cohorts against the 77-token SDXL
window.

Each cohort library contains 16 occupations and 16 personal details. The production prompt also draws from eight facial-feature bundles and four camera setups, ranging from a 26 mm phone close-up to an 85 mm tight headshot. The cue and camera clauses are kept concise and placed before the background details. These variations make the source cloud richer, but none is trusted to guarantee a new identity; the downstream shortlist and human review must reject cases where ArcFace responds mainly to perspective, crop, or styling.

Every submitted ComfyUI job is recorded in a state file. Restarting the stage
resumes unfinished work, distinguishes queued jobs from completed jobs, and
retries explicit execution failures within the configured allowance. Submission
errors are stored per item so one bad request does not discard the rest of a
batch.

## Stage 2 — embed in two different spaces

`02_embed_faces.py` stores:

- `emb_buf`: L2-normalized `buffalo_l` embeddings. This is the neutral judge used for selection and final reporting.
- `emb_ant`: L2-normalized antelopev2 `glintr100` embeddings. This is InstantID's conditioning space and the space whose covariance is sampled.
- `ant_norms`: the original vector norms. InstantID expects raw-scale vectors (roughly norm 16), not unit vectors.

Do not mix model packs. Arc2Face's `arcface.onnx`, antelopev2's `glintr100.onnx`, and `buffalo_l` are different coordinate systems even though every vector has 512 dimensions.

Embedding progress is checkpointed every 25 attempted images by default. Successful rows are skipped on restart, while corrupt images and detection failures are written to `<output>.errors.json` and do not abort the remaining batch.

## Stage 3 — build two ArcFace-diverse shortlists

`03_select_seed_faces.py` uses greedy farthest-point selection in `buffalo_l` space:

1. Start at the point furthest from the normalized centroid.
2. Add the candidate with the largest minimum cosine distance to the selected set.
3. Stop at 80 candidates.

Run it separately on the 500-image RealVisXL and Z-Image pools. It copies an 80-image shortlist and writes `shortlist.txt` plus the row-aligned `selected.npz`; it deliberately does not delete the raw pool.

ArcFace distance is not human identity distance. Pose, framing, lighting, styling, and rendering defects can move an embedding even when a person still sees the same face. Farthest-point selection is therefore used to make review tractable, not to make the final subjective decision.

## Stage 3b — choose the seed bank by eye

Review both 80-image shortlists and choose approximately 40 convincing identities from each model. A seed should:

- look like a different person from every other selected seed;
- retain the locked demographic traits;
- have a clean, usable face without metric-driven artifacts or distracting framing.

Copy `shortlist.txt`, remove the unselected entries, and pass the edited manifest to `03b_curate_seed_faces.py`. The script validates that every choice came from the shortlist, preserves its raw-pool indices and aligned embeddings, copies the selected images, and writes a curated `selected.npz` plus `curation.json`. The default expected count is 40 but is configurable; a numerical quota should not force an obvious duplicate into the bank.

This produces about 40 RealVisXL plus 40 Z-Image seeds. The keepers are intentionally biased toward the useful outside of the generated pool, first by ArcFace and then by human judgment. Consequently, “empirical p60–p99 radius” below means empirical among the human-reviewed seeds, not among every generated candidate.

## Stage 4 — fit and sample the identity cloud

Let the approximately 80 human-selected unit `glintr100` vectors be rows of `A`.

```text
mu = mean(A)
X  = A - mu
X  = U S V^T
lambda = S^2 / (n - 1)
```

For every proposed identity:

1. Draw covariance-shaped PCA coefficients `z_i ~ Normal(0, lambda_i)`.
2. Map them back through `V` to obtain a deviation.
3. Rescale the deviation to a uniformly sampled empirical p60–p99 radius.
4. Add `mu` and L2-normalize.
5. Reject if similarity to any keeper or accepted sibling exceeds `max(keeper pairwise p95, 0.45)`.

The outer shell is essential: the center of the cloud is the generic demographic mean. Rejection statistics—requested and accepted counts, attempts, gate, and shell—belong in the run report because acceptance depends strongly on the geometry of the reviewed seed bank.

The synthesized direction has no observed raw antelopev2 norm. Stage 4
therefore multiplies every sampled unit direction by the keepers' mean raw norm
before InstantID decoding and records that value in the run metadata. A fixed
mean is a deliberate approximation: it preserves the expected conditioning
scale without adding random adapter-strength variation. Sampling empirical
norms would be a separate experiment, not an assumed improvement.

## Stage 5 — decode once, at native quality

`05_render_instantid.py` sends both the reviewed seed vectors and accepted sampled vectors directly to InstantID with RealVisXL V5.0. Production settings from the study are:

- IP-adapter scale: `1.0` (0.9–1.3 had little effect on identity realization)
- ControlNet scale: `0.7`
- 30 steps, CFG 5
- no img2img refinement
- pose supplied by `image_kps`

Each person is rendered under multiple explicit pose templates, including front and profile views. The vector is stored, so new photographs do not require re-minting the identity.

## Stage 6 — decoded-image automatic filter

`06_rank_roster.py` embeds every required decoded image with `buffalo_l` and computes:

- `intra`: mean pairwise similarity among the same person's renders;
- `max_other`: similarity to the nearest remaining identity;
- `score = intra - max_other`.

It repeatedly removes the lowest score until top-k remain. Recomputing after each removal handles twins correctly: once the weaker twin disappears, the stronger one is no longer permanently penalized.

Its report uses the historical production keys (`origin_stats_top50`, `origin_totals`, `top50`, and `all`) and adds `schema_version`, `top_k`, and `rejected_for_missing_faces`. Missing or unreadable renders reject that identity instead of crashing the whole ranking stage.

Stage 6 deliberately does not use InsightFace's predicted age or sex as an
automatic gate. Those auxiliary predictions are not treated as reliable ground
truth for synthetic faces; off-cohort candidates are removed in the final human
review instead.

The tournament is an automatic shortlist, not the final authority. It judges rendered images rather than target vectors, but the neutral ArcFace model can still disagree with human perception.

## Stage 7 — final human selection

Review the surviving seed and sampled identities together. Remove subjective twins, off-cohort faces, and outputs whose image quality is unacceptable even when their scores are strong. The final roster is allowed to contain any mixture of RealVisXL seeds, Z-Image seeds, and sampled identities.

This last stage answers the actual product question—“do these look like clearly different people?”—which no embedding threshold can answer completely. In the recorded 160-candidate tournaments, sampled identities supplied 30 of 50 Spanish-men finalists and 26 of 50 Scandinavian-women finalists, confirming that the cloud contributes useful candidates rather than merely adding machinery.

## Infrastructure assumptions

The repository does not include model weights. The scripts assume:

- a ComfyUI server at `127.0.0.1:8188` with RealVisXL and Z-Image nodes/weights for stage 1;
- InsightFace `buffalo_l` and a clean antelopev2 pack for stages 2 and 6;
- InstantID and its ControlNet/IP-adapter weights for stage 5;
- CUDA for practical generation speed.

The original runs used rented H100/L40S machines and saved all relevant outputs locally before destroying the instances.
