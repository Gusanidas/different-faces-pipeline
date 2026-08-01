# Experiments and lessons

The failures are retained because each one ruled out a tempting shortcut.

## Same prompt, more seeds

Result: useful but uncontrolled. Seeds visibly changed lighting, clothing, texture, and sometimes identity. The controlled reproduction rendered 36 RealVisXL images of one fully locked Spanish-man prompt and computed all 630 `buffalo_l` pairwise comparisons. Similarity averaged `0.523`; the most distant pair was seeds 1007/1030 at `0.360` similarity (`0.640` cosine distance). The compact result is in [`metrics.json`](../data/reproduction-2026-07-29/metrics.json); the included measurement program regenerates the complete pair table.

Decision: seeds are a good source pool, but not a roster. Over-generate, embed, and use an evenly separated subset as a manageable human-review set.

## Explicit facial-feature prompts

We varied six exact feature clauses—gaunt, heavyset, lean, stocky, soft-featured, and rugged—while locking nationality, age, hair, skin, framing, and negative prompt. All prompts used the same six seeds so their averaged ArcFace centroids isolate prompt movement from seed noise. The centroid pairs averaged `0.857` similarity; heavyset/rugged was the most distant pair at `0.794`, a maximum cosine distance of only `0.206`. Exact text and full measurements are in [`prompts.json`](../experiments/vast_reproduction/prompts.json) and [`metrics.json`](../data/reproduction-2026-07-29/metrics.json).

Decision: keep prompt slots for enriching the initial cloud, never as the final diversity guarantee.

## Character backgrounds

Long biographies containing region, class, occupation, hobby, and visible physical consequences produced large changes, but they also mixed several mechanisms and made it unclear which instructions the model followed. The production version instead draws exactly two short, non-facial life details: one occupation and one personal detail from separate 16-item cohort pools.

A controlled comparison used 16 disjoint detail pairs and the same three seeds in both generators. RealVisXL prompt centroids averaged `0.554` cosine similarity with `0.659` maximum distance. Z-Image averaged `0.802` with `0.377` maximum distance and showed a much stronger recurring face. RealVisXL also allowed some occupations to change the scene despite the neutral-background instruction, so not all measured motion should be read as identity.

A cumulative 0/1/2/4/8-detail check showed no monotonic benefit from length. Relative to the no-detail centroid, RealVisXL distances were `0.105`, `0.255`, `0.386`, and `0.313`; Z-Image distances were `0.059`, `0.165`, `0.140`, and `0.139`. The two-detail prompts stayed within one 77-token SDXL window, while the four- and eight-detail versions required a second chunk.

Decision: use exactly two concise background details to enrich the generated seed cloud, then enforce the final identity requirement through shortlisting, cloud sampling, and human review. Exact prompts and measurements are in [`prompts.json`](../experiments/background_pairs/prompts.json) and [`metrics.json`](../data/background-pairs/metrics.json).

## Checkpoint, lens, CFG, and framing rotation

Rotating checkpoints and 35 mm/85 mm framing decorrelated some outputs. Lower CFG loosened the pull toward the conditional mean. These were small effects and also made the final visual style less coherent.

Decision: useful candidate-pool seasoning, not an identity mechanism.

## InstantID with prompt-generated donors

InstantID broke the locked-feature Spanish baseline: donor identity controlled the person while text controlled the look. Under the stricter Scandinavian women lock (20s, long light-blonde hair, fair skin, blue eyes), however, prompt-generated donors still collapsed after decoding. Twenty-four extreme-geometry donors produced no pairs below 0.35 cosine and a best-four maximum pair similarity of 0.630.

Decision: the donor pool itself needs an identity-space solution.

## More identity-adapter strength

Sweeping InstantID IP weight from 0.9 to 1.3 to 1.7 did not improve donor separation. Higher weight helped the same identity survive large head turns, then plateaued.

Decision: adapter strength is a fidelity control, not a diversity control.

## Arc2Face as the identity decoder

This was the first working identity-space path:

```text
sample ArcFace subspace -> Arc2Face -> reference image -> InstantID + RealVisXL
```

It produced stable, separated identities, but Arc2Face is SD1.5-based and outputs 512 px faces of visibly lower quality. A second decoder stage was needed. It was retired for rendered quality, not because its identity-space route was conceptually unnecessary or invalid.

Decision: geometrically correct but superseded.

## Img2img refinement

A RealVisXL img2img polish pass at denoise 0.35 improved finish but damaged identity, with measured similarity falling into roughly the 0.51–0.63 range.

Decision: do not polish after identity decoding.

## Direct synthetic-vector decoding

InstantID accepted synthetic `glintr100` vectors directly. Configurations across IP scale and cloud radius retained intra-person consistency around 0.77–0.79; the farther p60–p99 shell lowered sibling similarity.

The probe also labeled one configuration as covariance “temperature 1.3.” That scalar was canceled when every deviation was renormalized to a sampled radius, so the hot/base comparison was invalid and is not used as evidence. See [`experiments/ERRATA.md`](../experiments/ERRATA.md).

Decision: remove Arc2Face. Sample in InstantID's own conditioning space and decode once at 1024 px.

## Prompted pose

“Near profile” and similar prompt text barely moved the InstantID output. A same-look clustering experiment reported perfect identity ARI while yaw standard deviation was only 0.009—technically clustered, visually useless.

Decision: use explicit keypoint templates. With an `image_kps` pose ladder, pose-transfer R² reached 0.99 while identity clustering remained perfect in the successful study.

## Vector-space pre-gate only

A sampled target vector can be distant while its rendered image falls back toward the model's favorite face.

Decision: pre-gate cheaply in `glintr100`, then make the final acceptance decision on decoded `buffalo_l` embeddings.

## ArcFace distance versus subjective identity

Faces that are far apart in ArcFace space are not always different people in subjective evaluation. The embedding can react to pose, light, crop, styling, or a defect that a human does not consider identity.

Decision: use embeddings twice, but never as the only judge. First, farthest-point selection reduces each approximately 500-image model pool to an 80-face review set; a human chooses roughly 40 RealVisXL and 40 Z-Image seeds. After cloud sampling and InstantID decoding, a separate ArcFace model removes obvious collisions and unstable identities; a human then chooses the most distinct final faces from the mixed seed-and-sample shortlist.

This hybrid was productive in the recorded tournaments. Sampled identities occupied 30 of the 50 Spanish-men finalists and 26 of the 50 Scandinavian-women finalists. The result supports retaining cloud samples as competitors while retaining human judgment at both points where subjective identity matters.

## Three embedding spaces, one recurring trap

The project used three incompatible 512-dimensional spaces:

| Space | Role |
|---|---|
| InsightFace `buffalo_l` | neutral selection and evaluation judge |
| antelopev2 `glintr100` | InstantID conditioning and production cloud |
| Arc2Face `arcface.onnx` | retired Arc2Face decoder input |

Shape equality does not imply coordinate compatibility. A mistaken model pack can produce deterministic garbage while every array still has the expected dimensions.
