# Audit experiments: stronger detectors and a denominator-matched localization test

Written 20 November 2026, before any of these runs. Post hoc relative to the author's results (all RAW-FABRID results were already known), so every outcome below is exploratory and reported regardless of sign. Original project files are read, never modified.

## Questions

Q1. Is the original pipeline reproduced by an independent evaluation harness? (Sanity check.)
Q2. Do the paper's localization conclusions change when the spatial detector is configured like standard PatchCore (native 256 px input, 32 x 32 feature lattice, coreset of 10% of sampled candidates, bilinear upsampling plus Gaussian smoothing, sigma 4)?
Q3. The observed-area versus whole-frame contrast mixes two effects: (a) released windows leave 7.39% of marked pixels unseen (coverage), (b) moving a window changes the scores of the pixels it does see (placement). Separate them with a denominator-matched AP: every policy scored on exactly the pixel set S that the released layout observes.
Q4. What does it cost to repair coverage? "Gap-fill" = released windows plus every fixed-grid window that contains a pixel the released layout leaves unseen.
Q5. Are the paired roll-level differences distinguishable from zero under an exact sign-flip test over the seven rolls?

## Fixed design

- Folds: the seven saved leave-one-roll-out folds (baseline_seed_13017.json, held_out_<roll>), same ordered 1,000 clean training tiles per fold as the author's unseen-roll study.
- Test frames: the author's primary cohorts. Pixel metrics: 202 defective + 53 clean evaluation frames. Frame AUROC: those plus 48 calibration clean frames (303). Calibration frames never enter pixel or alarm denominators.
- Masks: intensity >= 128, 8-connected regions.
- Policies: released (28), fixed (28), overlap stride 128 (91), gap-fill (released + covering fixed windows).
- Detectors:
  - D0 orig_r18: the author's own fit_bank/tile_scores functions (ResNet-18, 128 px input, 256-vector coreset, nearest-cell maps). Used only to validate the harness against saved results.
  - D1 r18_native: ResNet-18 layers 2+3, native 256 px input, 384-d features, 3x3 local averaging, 256 random lattice positions per training tile (256,000 candidates), greedy coreset of 10% (25,600) on a 128-d Gaussian random projection, k=1 nearest distance, bilinear upsampling, Gaussian sigma 4.
  - D2 wrn50: as D1 but Wide-ResNet-50-2 (the PatchCore paper's backbone), 1536 channels averaged to 1024-d.
  - All seeds 13017.
- Score bins: log1p(score) x 5000 (relative resolution about 0.02%); unseen pixels rank below all observed pixels.
- Frame score: maximum of the fused map. Threshold per fold: empirical 95th percentile (method higher) of that roll's calibration clean frames, strict exceedance. Region hit: any pixel of the region above the threshold.

## Endpoints (per roll, then equal-roll mean)

1. Whole-frame AP, observed-area AP (released), denominator-matched AP on S for all policies.
2. Delta placement = AP_S(released) - AP_S(fixed). Delta whole = AP(released) - AP(fixed).
3. Frame AUROC per policy.
4. Region hits and clean false alarms at the calibrated threshold.
5. Exact two-sided sign-flip p-value over 7 rolls (128 sign patterns) for every mean paired difference; delete-one-roll ranges. With 7 rolls the smallest attainable two-sided p is 2/128 = 0.0156.

## Interpretation rules, fixed now

- If D1 and D2 both give a negative mean Delta whole and a sign-flip p <= 0.05, the coverage cost is a robust detector-independent finding. Otherwise the paper must not claim a localization ranking reversal.
- If Delta placement is near zero while Delta whole is negative, the localization penalty is due to coverage, not placement.

## Deviation log

- 20 Nov 2026, before any result: the smoke test crashed because PyTorch MPS cannot adaptive-average 1536 channels to 1024. D2 now averages adjacent layer-3 channel pairs (1024 -> 512) and concatenates them with the 512 layer-2 channels (1024-d). This mirrors the channel reduction the author already uses for ResNet-18. No outputs existed when this was changed.
- 20 Nov 2026, before any result: the Mac (8 GB RAM) swapped heavily during the D2 fit. For D1 and D2 the candidate pool is now 128 random lattice positions per training tile (128,000 candidates, stored in half precision) and the coreset is 10,240 vectors, i.e. 1% of all 1,024,000 training patch positions (the PatchCore-1% setting). Smoke test switched to D1. No outputs existed.
