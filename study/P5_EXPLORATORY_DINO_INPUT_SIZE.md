# Protocol 5, exploratory addendum E1: why does DINOv2-token kNN fail on native full frames?

Written 22 November 2026, 08:30 (Asia/Dhaka), after the Protocol 5 results were known and before any E1 score was computed. E1 is exploratory. It cannot change the frozen Protocol 5 decision, and the manuscript will label it as a post-hoc analysis.

## Observation that motivates E1

Across RAW-FABRID, TSFabrics and VisA, tiled prediction (R2, 256-pixel tiles) raised DINOv2-token kNN pixel AP by about 0.19 to 0.21 over native full-frame prediction (R1). The two WRN-50 detectors moved slightly the other way. In R1, DINOv2 sees a whole frame, but its memory bank was built from 256-pixel normal tiles.

## Two candidate explanations

- **H-context (fit-regime mismatch).** A ViT token depends on the whole input through attention. Tokens computed on a full frame are distributed differently from bank tokens computed on 256-pixel tiles, so normal full-frame tokens look anomalous. This predicts that a bank built from full normal frames largely restores full-frame performance.
- **H-size (input-size degradation).** DINOv2 features themselves degrade when the token grid is much larger than in training, for example through interpolated positional embeddings. This predicts that full-frame performance stays low even with a bank built at the same input size, and that matched-bank performance falls as tile size grows.

The two are not exclusive.

## Design

- **Data.** VisA, all 12 categories, official 1-class split, files checked against the frozen manifest. Test subset fixed before scoring: per category, the first 25 anomalous and the first 15 normal test images in sorted path order (480 images in total). Training images are used only to build banks.
- **Families.** DINOv2 ViT-S/14 (the frozen P5 feature extractor, padding and restoration) and, as a CNN control, the frozen WRN-50 extractor with PatchCore scoring.
- **Banks.** To keep cost equal across arms, every bank in E1 is a seeded random subsample of 10,240 valid feature vectors, not a greedy coreset. Source windows: up to 300 windows drawn with `default_rng(13018)` from all eligible windows of the category's normal training images, about 35 vectors per window drawn with `default_rng(13017)`, then exactly 10,240 kept with `default_rng(13019)`. A vector is valid if its 14 x 14 patch lies inside the real window (DINOv2) or always (WRN-50). Window types:
  - B256, B512, B768: edge-aligned grid tiles of that side.
  - Bfull: whole training frames.
  - B518: whole training frames resized so the long side is 518 pixels, the DINOv2 pretraining resolution.
- **Inference arms** (all graded on whole-frame pixel AP over unique native pixels, maps fused by maximum where windows overlap):
  - Mismatch sweep, DINOv2: bank B256; inference tiles T = 256, 512, 768 and the full frame.
  - Matched sweep, DINOv2: bank B_T with inference at T, for T = 512, 768, full (T = 256 is shared with the mismatch sweep).
  - Training-resolution arm, DINOv2: bank B518, whole frame resized to long side 518, score map resized back bilinearly before smoothing.
  - CNN control, WRN-50 PatchCore: B256 with T = 256 and full frame; Bfull with full frame.
- **Statistics.** Per-category AP from pooled histograms, macro mean over categories, 95% percentile intervals from 2,000 within-category image resamples with shared draws (seed 13017), as in Protocol 5 for VisA.

## Contrasts reported (all exploratory)

1. Context: DINOv2 Bfull/full minus B256/full. H-context predicts a large positive value.
2. Residual tiling advantage: DINOv2 B256/256 minus Bfull/full.
3. Size trend under matched banks: B512/512, B768/768, Bfull/full, each minus B256/256.
4. Size trend under the mismatched bank: B256 at 512, 768 and full, each minus B256/256.
5. Training resolution: B518/518 minus B256/256, and minus Bfull/full.
6. CNN control: WRN-50 Bfull/full minus B256/full, and B256/256 minus B256/full.

## Reading rule, fixed now

- If contrast 1 is at least +0.05 and contrast 2 lies within plus or minus 0.03, the report will say that the full-frame failure is mainly a bank/context mismatch (H-context).
- If contrast 1 is below +0.05, or matched-bank AP falls steadily with T (contrast 3 negative at every T and below -0.03 at full), the report will say that input size itself degrades DINOv2 tokens (H-size).
- Anything else is reported as mixed, with the numbers.
- Under any outcome, the manuscript will state that E1 was designed after the main results.
