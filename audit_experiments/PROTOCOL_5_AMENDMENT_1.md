# Protocol 5, Amendment 1 (FROZEN 21 November 2026)

Amends `PROTOCOL_5.md` (SHA-256 `128349e28334166855d27274b1637f168f874031613d2e55d9ba3c560f51e40b`), which stays unchanged on disk. Approved by the author on 21 November 2026 (22:47, Asia/Dhaka), with four clarifications incorporated: the RAW-FABRID roll-bootstrap interval, the DINOv2 map-restoration order, the TSFabrics cluster gate, and the meaning of the 0.01 threshold. Its SHA-256 is recorded in `PROTOCOL_5_AMENDMENT_1.sha256`. It governs wherever it differs from the frozen protocol. Later changes go in a new dated amendment.

**Reason for the amendment:** an external pre-run review (21 November 2026) found two blocking problems and several definitional gaps. No Protocol 5 detector output exists, so every change below was made before any result on the new datasets. The two blocking findings were checked independently on the local files:
- VisA mask values are 0 to 8, so a ≥128 threshold empties every mask.
- The AITEX 150-versus-0 discrepancy comes from rounding (floor in the original rule, `round()` in `cut_audit.py`), not from the threshold.

The two open values were fixed by the author on 21 November 2026, before any result: a minimum effect of 0.01 pixel AP, and an exploratory ranking claim.

---

## A1. Mask decoding (replaces "all masks at ≥128")

Each dataset has its own decoder, whose output is a boolean defect mask at native resolution. Each decoder has checks that stop the run if they fail.

| Dataset | Raw encoding | Defect rule | Checks |
|---|---|---|---|
| RAW-FABRID | 0/255 grayscale | value ≥ 128 | marked total matches Protocols 1 to 4 (7.386% unseen under released crops) |
| AITEX | 0/255 grayscale; two images have `_mask1`/`_mask2` | value ≥ 128 (the original `external_aitex.py` rule), OR over multiple masks | reproduces 485,027 marked pixels on the 103-image cohort; empty, missing or misaligned masks are listed and excluded, never treated as normal |
| VisA | category IDs 0 to 8 | value > 0 | every anomalous image has a non-empty mask; per-category counts recorded |
| TSFabrics | to be read from the real files on Kaggle (A7) | union of genuine defect labels only; label 4 (hole) and label 5 (missed stitch) kept as separate classes | cutlines (label 2) are never defect foreground; they are kept as a separate mask for a false-alarm analysis |

Masks are never resized for scoring. Every evaluation happens against the native-resolution mask. For each dataset, 10 random overlays (image, mask, outline) are saved for a visual check before any detector runs.

## A2. Primary question and endpoint (replaces the "B versus reference" endpoint)

**Why it changes:** for a grid without overlap, pixel AP from patch-by-patch scoring equals pixel AP after stitching, because AP does not depend on the order in which pixels are listed. The old endpoint therefore mixed a prediction effect with a counting effect.

**New primary question (prediction layout):** does tiled prediction stitched into the frame (R2) give a different whole-frame pixel AP from native full-frame prediction (R1)?
- Both are graded on the same unique native pixels, with the same masks, decoder and smoothing.
- Where tiles overlap, R2 keeps the maximum score per pixel.
- **Primary endpoint:** (R2 minus R1) whole-frame pixel AP, using the primary detector.

**Fitting regime (primary):** one memory bank, built from normal 256 × 256 tiles, is used for both R1 and R2. Only the prediction layout changes.

**2 × 2 control (secondary):** bank built from tiles or from full frames, crossed with prediction on tiles or on full frames. It shows whether conclusions depend on where the bank came from. Run on AITEX and VisA, and on RAW-FABRID if memory allows.

**R1 is "native full-frame inference", not ground truth.** If Wide-ResNet-50 R1 does not fit in the Mac's memory, the same model and comparison move to a Kaggle GPU. No other model is substituted for R1 alone.

## A3. Secondary questions

**Counting and reporting (no new predictions).** These are computed from the saved R2 tile predictions. Each metric is reported in its own unit, never subtracted across units.
- unique-pixel AP;
- occurrence-weighted pixel AP, where overlapping pixels count once per tile;
- patch AUROC and patch AP, with one score per patch (its maximum) and a patch labelled defective if it has ≥1 defect pixel, plus a 1% foreground sensitivity version;
- parent-image AUROC.

**Practice A (defect-aware crops).** Each dataset's rule is reported as its own intervention, not as one shared rule:
- RAW-FABRID: the released offsets.
- AITEX: the original floor-division rule, as the faithful replication; the `round()` variant is reported separately as a constructed layout.
- VisA and TSFabrics: constructed rules, each documented.

For every layout we report:
- unseen defect pixels;
- unseen total pixels;
- unique windows;
- unique covered pixels;
- pixel multiplicity.

**Practice C (resized whole image).** One fixed rule for all datasets:
- downscale by a factor of 4 with the aspect ratio preserved;
- score;
- upsample the prediction back to native size;
- evaluate on the native mask, with any padding excluded.

**Fragmentation.** Tracked through connected components of the native mask:
- the retained fraction per component;
- whether it touches a tile border;
- components larger than 256 pixels in either side, reported separately because no tile can hold them.

A patch under 1% defect is not called a fragment unless its component was actually cut by a border.

## A4. Detector definitions (replaces the three names)

**1. PatchCore-style (primary).** Exactly as in `code/strong_patchcore.py`, Native class:
- Wide-ResNet-50 (ImageNet weights), layers 2 and 3, with layer-3 channel pairs averaged so that the concatenated feature is 1024-d;
- 128 sampled positions per training tile, 1,000 training tiles, so 128,000 candidates;
- greedy coreset of 10,240 vectors on a 128-d Gaussian projection (8% of candidates, about 1% of all lattice positions);
- k = 1 Euclidean nearest neighbour;
- bilinear upsampling, then Gaussian smoothing with σ = 4.

It is described as PatchCore-style, not standard PatchCore.

**2. Shared-Gaussian Mahalanobis baseline.**
- Same features as detector 1.
- One Gaussian fitted on all sampled normal vectors (pooled across positions) for each dataset, category or fold.
- No channel selection.
- Covariance regularised with 0.01 × identity.
- Same map processing as detector 1.

**3. DINOv2-token kNN.**
- `dinov2_vits14` from the official repository, last-layer patch tokens, L2-normalised, cosine distance to the nearest normal token, same map processing.
- Bank from the same normal images as detector 1, with the coreset rule as in detector 1.
- **Input alignment:** each input, tile or full frame, is reflect-padded on the bottom and right to the next multiple of 14 (256 becomes 266). The same policy applies under R1 and R2.
- **Map restoration:** the token-score map is upsampled to the padded input size, then cropped to the original height and width. Padding never enters evaluation. No boundary token is deleted before upsampling, because a partly padded token still describes real pixels.
- **Normal bank:** only tokens whose 14 × 14 patch lies entirely inside the real image enter the bank. Partly padded tokens are scored at test time but never stored as normal references.
- **Alignment test:** before any real run, a synthetic image whose height and width are not divisible by 14 (for example 257 × 263), with a known bright square, must produce a score map whose peak falls on the square to within one token, and whose size equals the original image.
- **CNN full-frame path:** R1 maps are restored to each frame's actual height and width (for example 1792 × 1024, 4096 × 256, 800 × 640), never to a fixed 256 × 256 size. The same alignment test applies.
- This is not a reproduction of AnomalyDINO; deviations from its official code are listed.

Detectors 1 and 2 share one feature family. The study therefore has three scoring pipelines over two feature families, and is described that way.

## A5. Statistics

| Dataset | Primary uncertainty | Notes |
|---|---|---|
| RAW-FABRID | Roll level. **Primary interval:** paired bootstrap of the seven roll-specific AP differences (resample rolls with replacement, equal roll weights, 2,000 resamples, percentile 95% interval). The exact sign-flip test over 7 rolls is reported alongside | Seven rolls carry limited uncertainty information, so this interval is reported with that caveat. Image-level bootstrap within rolls is secondary and labelled conditional on the 7 rolls and fitted banks. Raw scores from different folds are never pooled into one AP |
| AITEX | Bootstrap over whole strips, 2,000 resamples, shared draw indices for every comparison | Pooled bank across fabric types is primary (one score scale). Conclusions are conditional on the observed fabric types; per-type banks are a sensitivity check |
| VisA | AP within each category, then a macro average over 12 categories; bootstrap of whole images within category, shared draws, recompute the macro mean | Conditional on the 12 categories |
| TSFabrics | Cluster bootstrap over the independent recording clusters in the pilot's frozen table (reported there as 11 clusters, 8 condition groups; to be verified, A7). **Positive cluster:** a cluster containing at least one defect. **Negative cluster:** a normal-only cluster. Positive and negative clusters are resampled as two strata, each cluster taken whole, without separating its normal and defective frames, so every draw contains defects | Eligibility and replacement are decided by the gate in A7, before any detector scoring |

All intervals are percentile 95% intervals, bootstrap seed 13017. Fixed-bank intervals leave out coreset-seed variation. A 3-seed sensitivity check on AITEX shows how large that variation is; seeds are not treated as extra datasets.

## A6. Decision rules (replaces rules 1 and 2)

**Rule 1, prediction-layout claim (primary, directional).** Supported only if (R2 minus R1) whole-frame pixel AP for the primary detector:
- has the same sign on at least 3 of the 4 main datasets; and
- on each of those 3, has |effect| ≥ 0.01 and a 95% interval excluding zero.

As a secondary, non-directional summary, we also report how many datasets show |effect| ≥ 0.01 with an interval excluding zero, in either direction.

**What this rule establishes:** an effect that is not zero and whose point estimate is at least 0.01. It does not establish that the true effect is at least 0.01. For example, an estimate of +0.011 with interval [+0.001, +0.021] passes. Papers and summaries will describe a passing result in exactly those terms and will not claim that a minimum effect of 0.01 was demonstrated. For each dataset we also report whether the whole interval lies beyond ±0.01, as a descriptive statement only.

**Rule 2, detector ranking (exploratory only).**
- Rankings under R1 and R2 are reported with paired intervals from the same resamples.
- Any order change is labelled exploratory: with 3 detector pairs across 4 datasets there are 12 chances to see one.
- If there is no change, the wording is "a ranking change was not demonstrated", never "unchanged".

**Rule 3 (kept).** Every result that fails a rule is reported in the main text.

## A7. Gates before any detector run

**AITEX.**
- Confirm the cohort: the original 103-image cohort over six fabrics is primary.
- Exclusions, each recorded with its reason:
  - `0094_027_05` (3796 px wide);
  - `0100_025_08` (no mask);
  - `0106_010_03` (empty mask).
- Any extended cohort is secondary and reconciled with the primary one in a table.

**VisA.**
- Decoder checks pass.
- Overlays are saved.

**TSFabrics, run on the Kaggle copy without downloading.**
- Confirm:
  - frame size (800 × 640 per the publisher);
  - mask encoding;
  - image and mask alignment;
  - the defect and cutline annotation paths;
  - the scenario-to-cluster mapping;
  - the counts of eligible positive and negative clusters.
- Write a separate P5 manifest that refers to any pilot provenance. Study 1 files stay unchanged.

**TSFabrics cluster gate.** Before any P5 detector scoring, count the eligible independent positive test clusters. If there are fewer than five, TSFabrics is kept and reported descriptively, and MVTec AD 2 Fabric replaces it in the four-dataset primary panel. The resulting panel is written down, fingerprinted and frozen before any result exists.

**Fallback (general).** MVTec AD 2 Fabric replaces any main dataset only if that dataset fails a gate above (including the cluster gate) for a declared technical or data reason, recorded before any detector result. This resolves the frozen text's inconsistency between "real-fabric dataset" and "any main dataset". An unfavourable result is never a reason to substitute.

**Evaluation and training populations frozen before scoring.** For every dataset, a manifest lists, with file hashes:
- every test image, defective and clean (the 103-image AITEX cohort names only the defective part);
- every normal training reference used for a bank, and its sampled positions or seeds;
- every fold, category and fabric-type assignment;
- every exclusion and its reason.

The manifests are fingerprinted and frozen before any detector scoring, and the run stops if a file differs from its manifest.

**2 × 2 control budgets.** The tile-bank and full-frame-bank arms use the same normal parent images, the same number of sampled candidate vectors (128,000) and the same bank size (10,240). A larger pool of full-frame references would otherwise act as a second, unnoticed intervention.

**Pipeline tests before full runs.** Synthetic tests cover:
- AP invariance to pixel order;
- the duplicate-pixel effect;
- full coverage under R2;
- known gaps under practice A;
- correct unpadding for DINOv2.

Then timing and shape checks run on training data only. No setting is chosen using test-set metrics.

## A8. Wording corrections to earlier documents

| Document | Old wording | Corrected wording |
|---|---|---|
| `PLAN_PROTOCOL_5.md` | "the gain comes from coverage, not from extra second looks" | "coverage restoration adds value beyond the tested window-count and defect-pixel-count controls; both coverage and overlapping predictions contribute" |
| `PLAN_PROTOCOL_5.md` | impact factor ≈ 10.87 | removed until verified in Journal Citation Reports |
| `PLAN_PROTOCOL_5.md` | "strong Q1 paper if the rules pass" / "stays publishable" | removed; publication is not predicted by the decision rules |
| `RESULTS_CUT_AUDIT.md` | "defect-aware recentring left no unseen area" (VisA) | "no annotated foreground was omitted under this constructed rule; background still went uninspected (61,049,581 pixels in 772 of 1,200 frames, per the external review; to be recomputed)" |
| `RESULTS_CUT_AUDIT.md` | AITEX difference "most likely a mask threshold and rounding difference" | "threshold explains the foreground total (485,027 vs 489,335); floor versus `round()` crop placement explains the unseen count (150 vs 0)" |
| `RESULTS_P4.md` and summaries | label-free targeting "close to" dense overlap | "the tested two-window targeting policy did not meet its criterion; no equivalence to dense scanning was tested" |
| `LITERATURE_SURVEY.md` | "nobody has quantified the differences side by side" | removed; add Divide and Conquer (CVPRW 2024), HiAD (2025) and Beyond Academic Benchmarks (CVPRW 2025) as direct comparators, and code practices by dataset × paper × split |
