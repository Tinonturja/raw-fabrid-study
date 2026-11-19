# Does a targeted second look improve localization at a fixed window budget?

This exploratory follow-up was specified after the main RAW-FABRID and AITEX results were known and before computing any targeted rescan outcomes. Use the existing primary split (13017) on both collections. The independent sample for a new generalization claim would be a future fabric collection.

## Decision rule

Refit the recorded spatial ResNet-18 PatchCore variant with exactly the original normal reference tiles and the original seed for each dataset. Begin every frame with a complete 256-pixel grid. Eligible extra windows are the other positions on the 128-pixel-stride dense grid. Score every candidate footprint using the mean of its four highest first-pass spatial cell scores. Use precisely seven extra windows on RAW-FABRID (28 + 7) and four on AITEX (16 + 4). Break ties by row and then column, allow neighboring windows, and use the same rule for clean and defective frames. The selector sees only the fixed-grid responses and the known grid geometry.

Compare four policies on each original image. Use the fixed grid alone, the grid with targeted extras, the grid with evenly spaced extras at the same budget, and the full dense grid. For the evenly spaced control, sort the eligible coordinates in row-major order and take the midpoint index of each of K equal intervals in that list, rounding down. This defines one reproducible placement control. Use maximum fusion for every policy. Record the chosen coordinates per frame. Window counts describe inference work. Timing a production scanner is outside this comparison.

## Outcomes

Primary contrast per dataset is targeted minus evenly spaced whole-frame pixel average precision (AP), using original masks, the recorded held-out evaluation frames, and the original 0.001 distance bins. Also report each policy's frame AUROC, clean-calibrated alarms, region hits and full-frame AP. Reuse the original split and the higher 95th percentile clean calibration threshold rule separately per policy. Recompute pooled pixel AP after omitting each roll or fabric once, with the fitted model and window rules held fixed. These deletions describe sensitivity to cohort composition.

Check each reconstructed fixed and dense control against the archived result. Record per-frame scores and the histogram evidence needed to verify the analysis. Report both datasets regardless of the direction of the contrast. No retuning on either collection.
