# Protocol 5 implementation log (FROZEN 22 November 2026, 00:16 Asia/Dhaka)

Implements `PROTOCOL_5.md` as amended by `PROTOCOL_5_AMENDMENT_1.md` (SHA-256 `2b599707b556cd6c667129329c72154153ea6b03ea3e44b945b64d40c2e1b848`). It records only choices the protocol leaves open. It changes no endpoint, rule or estimand. Approval: the author authorised the overnight work at 23:55 on 21 November 2026 and was asleep when the log was frozen; this was taken as approval of the log, and the author can review every choice below. The log is fingerprinted in `PROTOCOL_5_IMPLEMENTATION.sha256`, and `code/p5_run.py` refuses a full run unless the file matches.

Smoke tests before freezing (outputs in `results_p5/*/smoke/`, never analysed as results): AITEX 3 + 3 strips and VisA candle 2 + 2 images ran end to end, with no R1 memory failures, about 8 s per image.

Code: `code/p5_core.py` (NumPy), `code/p5_models.py` (PyTorch), `code/p5_run.py` (runner), `code/tests_p5.py` (33 tests passed on 21 November 2026, including the AITEX real-mask check of 485,027 defect pixels and 150 unseen).

## I1. Memory banks
- **Tiles:** training tiles come from the edge-aligned 256 grid of each normal training image. If a unit has more than 1,000 tiles, 1,000 are drawn first, with `default_rng(13018)`.
- **Candidates:** candidate positions are then drawn exactly as in `strong_patchcore.Native.fit` (`default_rng(13017)`, per tile, without replacement), with `ceil(128,000 / number of tiles)` positions per tile.
  - AITEX: 60 clean strips give 960 tiles, 134 positions each, so 128,640 candidates.
  - VisA: 1,000 tiles per category, 128 each, so 128,000 candidates.
- **PatchCore-style and DINOv2-token kNN:** both use a 10,240-vector greedy coreset (128-d Gaussian projection, as in Protocols 1 to 4).
- **DINOv2 bank tokens:** only tokens fully inside the 256 tile enter the bank, which is 18 × 18 of 19 × 19.
- **Coreset device:** the selection loop uses the same formula as Protocols 1 to 4. Banks built during the smoke tests (AITEX and VisA candle, both families) ran it on the GPU (MPS). All later banks run it on the CPU, which is about 3.5 times faster. Results can differ only through floating-point rounding at near-ties.
- **Shared-Gaussian baseline:** fitted on all candidates (not the coreset): mean plus covariance + 0.01·I, inverted in float64.

## I2. Cohorts
- **AITEX:** the frozen manifest (`manifests/aitex_manifest.json`).
  - Bank: the 60 clean "train" strips.
  - Test: the 103 defective strips and 31 clean evaluation strips.
  - The 30 calibration strips are not used, because the primary comparison needs no threshold.
  - One pooled bank across fabric types; per-type banks come later, as a sensitivity check.
- **VisA:** the frozen manifest, official 1cls split. One bank per category from its normal training images. The test set is all 2,162 test images (1,200 anomalous, 962 normal).
- **Input format:** AITEX grayscale is copied into three channels. Both feature families use ImageNet normalisation.

## I3. Scores and maps
- **R2:** each 256 tile is scored on its own; its map is restored to 256 × 256 (bilinear, then Gaussian σ = 4 with reflect padding); maps are fused by maximum, one value per unique pixel.
- **R1:** the whole image is passed through the network once, and the map is restored to the image's own height and width with the same steps.
- **DINOv2 maps:** inputs are reflect-padded to multiples of 14; the token map is upsampled to the padded size, cropped to the original size, then smoothed (Amendment 1, A4).
- **Practice C:** the image is downscaled 4 times (bilinear, minimum side 64) and scored with the same tile-trained bank. The low-resolution map is restored at low resolution, then bilinearly upsampled to native size and graded on the native mask.
- **Practice A:** AITEX only in this run, with the original floor-division rule. AP is reported over the whole frame (unseen pixels rank last) and over observed pixels. VisA's constructed rule A comes in a later run, after its geometry audit is redone with whole-image coverage counts.

## I4. Metrics as computed
- **Histograms:** pixel scores are binned at 5,000 bins per unit of log1p(score). Never-scored pixels go in bin 0, the lowest rank. AP treats pixels in one bin as tied.
- **Per-image storage:** a sparse histogram is stored for each (scorer, layout). Layouts:
  - R1, R2;
  - R2occ (every tile occurrence counted);
  - A_all, A_obs;
  - C4.
- **Patch records:** for each R2 tile, the tile maximum score per scorer and the tile's defect-pixel count. The patch label is ≥1 defect pixel; the 1% foreground version is computed in analysis.
- **Parent records:** the frame maximum for each scorer and layout.
- **Resampling:** AITEX gives one AP over its whole cohort, resampled by strip. VisA gives an AP per category, resampled within category, then macro-averaged. Both use 2,000 shared draws with seed 13017.

## I5. Failures and order of work
- **R1 out of memory:** if R1 for a feature family exceeds the Mac's memory on some image, it is logged, and that family's R1 for the whole dataset is rerun on a Kaggle GPU with the same model. Its R2 is not rerun.
- **Smoke-test outputs:** written to `results_p5/<dataset>/smoke/` and never used in analysis. Banks are always built from the full training set, even in a smoke test, so the full run reuses them. Every output file records each bank's SHA-256.
- **Run order on the Mac:** AITEX (both families), then VisA with the WRN-50 family (PatchCore-style primary and shared Gaussian) for all 12 categories, then VisA with DINOv2. Each family's outputs are written to separate files per unit and merged in the analysis.
- **Later separate scripts, each logged before it runs:**
  - RAW-FABRID: reuses the Protocol 1 WRN-50 banks after a fingerprint check; Gaussian and DINOv2 banks are rebuilt with the identical sampling.
  - TSFabrics on Kaggle, after its cluster gate.
  - The 2 × 2 bank-origin control, fragmentation, VisA practice A, and the 3-seed AITEX check.
