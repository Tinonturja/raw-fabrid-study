# Exploratory analysis E2: memory-bank origin crossed with inference layout (paired 2 × 2)

Status: exploratory. Designed on 22 November 2026, after the Protocol 5 results and the E1 analysis were known, and before
any E2 score was computed. It is new work, not part of the Protocol 5 pre-registration. This file is fingerprinted
(SHA-256) before the first E2 image is scored; any later change is logged as an amendment before the scores it affects.

## Why

E1 (six VisA categories) found that a DINOv2 bank fitted on whole frames made native full-frame scoring the best DINOv2
condition. An external review raised four limits, which E2 addresses:
1. E1 had three bank/layout cells, not four: it lacked a full-frame bank applied to tiles, so the interaction could not
   be estimated.
2. E1 drew tile and full-frame bank windows separately, so bank context was confounded with which reference pixels the
   bank saw.
3. E1 did not include RAW-FABRID, where Protocol 5 reported a detector-ranking reversal.
4. E1 used one bank seed.

## Design

**Units and test images.** RAW-FABRID: all 255 Protocol 5 evaluation frames (202 defective, 53 clean), evaluated per
held-out roll (7 rolls). VisA: all 12 categories, the first 25 anomalous and first 15 normal test images per category in
sorted order (480 images; the E1 rule extended to every category). All files are checked against the frozen manifests.

**Paired reference windows.** For each unit the two banks are built from the same reference windows, so they differ only
in the context in which features are computed.
- RAW-FABRID: the candidate windows are the unit's released reference tiles (1000 per held-out roll, all from the other
  rolls). Each tile name encodes its source frame and grid cell (row, column), and the tile is that frame's
  256 × 256 square at (256 × row, 256 × column); every tile is checked pixel for pixel against its source frame before use.
- VisA: the candidate windows are the 256-pixel edge-aligned grid tiles of the category's training images.
- Source images: 80 per unit, drawn once with seed 13018 (RAW-FABRID: 80 source frames among those that supply reference
  tiles; VisA: 80 training images). Windows are drawn from the tiles of these source images only.
- For each bank seed s, up to 300 windows are drawn from the candidates with seed s.

**Bank vectors.** For each window, features are computed (a) on the 256 × 256 tile alone, giving the tile bank B_tile,
and (b) on the whole source image, keeping only the feature cells whose centre lies inside the same tile, giving the
full-frame bank B_full. Only cells wholly inside the real image enter either bank (the Protocol 5 validity rule). The
same number of vectors is sampled per window for both banks (ceil(10240 × 1.05 / windows), seed s), and each bank is then
subsampled to 10,240 vectors (seed s). Both banks are random subsamples; neither uses a greedy coreset, so absolute AP is
not comparable with the Protocol 5 tables.

**Bank seeds.** s ∈ {13017, 13027, 13037}.

**Feature families.** DINOv2 ViT-S/14 token kNN and Wide-ResNet-50 PatchCore-style scoring, with the Protocol 5 feature
extractors, map restoration, smoothing and max fusion unchanged.

**Inference layouts.** R2: the 256-pixel edge-aligned grid, tile maps fused by maximum. R1: the native full frame.

**Arms.** family × bank (B_tile, B_full) × layout (R2, R1) × seed: 2 × 2 × 2 × 3 = 24 arms. Features of each test image
are computed once per family and layout and scored against every bank.

**Run order.** RAW-FABRID rolls first, then VisA categories in sorted order.

## Metrics

Primary: whole-frame pixel AP per unit (binned histograms as in Protocol 5), averaged over the three seeds within each
unit, then averaged over units with equal weights (rolls for RAW-FABRID, categories for VisA).

Secondary: pixel recall at the threshold where 1% of normal pixels in the unit score higher (normal-calibrated
operating point); wall-clock time and peak accelerator memory per image for feature extraction and scoring under each
layout and family, on the same machine.

## Estimands (per dataset and family)

- E_bank,full = AP(B_full, R1) − AP(B_tile, R1): bank context at native inference.
- E_bank,tile = AP(B_full, R2) − AP(B_tile, R2): bank context at tiled inference.
- Interaction = E_bank,full − E_bank,tile.
- E_layout,matched = AP(B_full, R1) − AP(B_tile, R2): native versus tiled, each with its matched bank.
- E_layout,tilebank = AP(B_tile, R2) − AP(B_tile, R1): the Protocol 5 contrast, reproduced under E2's banks.

Uncertainty: RAW-FABRID, paired bootstrap over rolls (2,000 draws) and exact sign-flip test over the seven rolls;
VisA, macro mean with images resampled within each category (2,000 draws, an independent stream per category, shared
across arms). Seed-level estimates are reported separately.

## Reading rule (fixed before scoring)

For DINOv2, the bank-context explanation is **supported** if, on both RAW-FABRID and VisA:
(a) E_bank,full ≥ +0.05 with a 95% interval excluding zero, and positive under each of the three seeds; and
(b) the upper end of the 95% interval of E_layout,matched is at least −0.01 (native scoring with a matched bank is not
clearly worse than tiled scoring with a matched bank).
It is **partly supported** if (a) and (b) hold on exactly one dataset, and **not supported** otherwise.
The interaction, the WRN-50 control, the RAW-FABRID detector order under matched banks (B_full/R1 versus B_tile/R2) and
the secondary metrics are reported descriptively, without a pass/fail rule.

## Smoke check

Before the full run, one unit per dataset is run on two defective and two clean images to check shapes, timing and
memory. Smoke outputs are stored separately and not analysed.
