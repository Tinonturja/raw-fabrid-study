# Protocol 5 run log

Dated record of what ran and of every change made after `PROTOCOL_5_IMPLEMENTATION.md` was frozen (SHA-256 `632abc47...87fe42`, 22 November 2026, 00:16 Asia/Dhaka). The frozen files are never edited.

## Model weights used
- Wide-ResNet-50-2, torchvision ImageNet weights: `wide_resnet50_2-95faca4d.pth`, SHA-256 `95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8`.
- DINOv2 ViT-S/14: `dinov2_vits14_pretrain.pth`, SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`. Loaded through `torch.hub` from `facebookresearch/dinov2` (main branch, downloaded 21 November 2026). The SHA-256 of the concatenated hub Python sources is `2883c5ee...a5869`. xFormers was not available, so standard attention was used.
- Environment: PyTorch 2.15.0, torchvision 0.29.2, Apple M-series GPU (MPS), macOS, 8 GB memory.

## 22 November 2026
- **00:16:** the implementation log was frozen. The full run (`run_p5.sh`) was started in Terminal. A first launch failed because the script had lost its execute permission when copied; it was relaunched with `/bin/bash`. No output was produced by the failed launch.
- **00:17:** the pre-run tests passed (33 of 33, including the AITEX real-mask check). The AITEX full run started, using the banks built during the smoke test.
- **About 00:30:** RAW-FABRID was added, as planned in the implementation log (I5), before any RAW-FABRID Protocol 5 output existed:
  - **Manifest:** `manifests/rawfabrid_manifest.json` (SHA-256 `ab08685a...93715`), built by `code/p5_rawfabrid_manifest.py`. It covers 255 frames across 7 rolls and reproduces 140,701 of 1,904,926 defect pixels (7.386%) unseen under the released crops.
  - **Runner:** `code/p5_run.py` gained a RAW-FABRID adapter. Practice A uses the released offsets.
  - **PatchCore-style bank:** the Protocol 1 WRN-50 bank of each roll is reused only if its SHA-256 matches the Protocol 1 record.
  - **Gaussian and DINOv2 banks:** the shared Gaussian is fitted on candidates re-sampled from the same 1,000 reference tiles with the Protocol 1 sampling procedure. The DINOv2 bank is built from the same tiles.
  - **Analysis:** `code/p5_analyze.py` gained the roll-level primary interval (paired bootstrap of the seven roll differences, plus the sign-flip test).
  - **Scheduling:** `run_p5_rawfabrid.sh` waits for the AITEX and VisA run to finish, then runs a smoke test (roll 3A, 2 + 2 frames), the full run and the analysis.
  - **Compatibility:** these code changes do not alter the AITEX or VisA paths. Output file names and the practice A rule for AITEX are unchanged.
- **About 00:45:** the TSFabrics gate could not run overnight: the gate script (`code/p5_tsfabrics_gate.py`) is ready, but Kaggle was not signed in in the browser used for the overnight run, and the author is asleep. The gate waits for the author.
- **About 00:50, before any MVTec AD 2 output:** the fallback dataset was prepared, so the four-dataset panel is not delayed whichever way the gate goes:
  - **Manifest:** `manifests/mvtec_fabric_manifest.json` (SHA-256 `42ba0ab4...a5d8b`), public test split only: 387 training, 90 anomalous and 66 normal test images from 26 physical samples.
  - **Scheduling:** `run_p5_mvtec.sh` runs MVTec AD 2 Fabric after RAW-FABRID.
  - **The panel rule is unchanged** (Amendment 1, A7). The TSFabrics cluster gate alone decides whether TSFabrics or MVTec AD 2 Fabric is the fourth primary dataset, and the gate depends only on data counts. If TSFabrics passes, the MVTec results are reported as secondary. Running MVTec early does not change this.
  - **MVTec resampling unit, decided now:** the test images are lighting and shift variants of the same physical samples, so the bootstrap resamples whole samples (object IDs), not single images. This follows the same reasoning as the TSFabrics cluster rule in A5.
- **About 04:35, TSFabrics gate (Amendment 1, A7), run on Kaggle in the author's account. No detector was run.**
  - **First version invalid.** The first run counted any non-empty mask as usable. The annotation folders are split into `cutline` (normal production marks) and `defect`, so that run could have counted cutline masks. Its result is not used.
  - **Corrected run.** `code/p5_tsfabrics_gate.py` (SHA-256 `59ae7b8c238c35a0c74646f4f2411b4e846c980240cea44484dffbdebad7cff2`) counts only non-cutline masks. Full log: `results_p5/tsfabrics/gate_v2_log.txt`.
  - **Result:** 7 independent positive clusters have non-empty defect masks (C1, C2, C4, C5, C6, C7 on texture T1; C9 on T2). Every frame is 800 x 640 grayscale.
  - **The gate passes**, and each positive cluster has normal clusters of the same texture for a leave-cluster-out bank.
  - **Primary panel, frozen now, before any TSFabrics or MVTec AD 2 detector output:** RAW-FABRID, AITEX, TSFabrics, VisA. MVTec AD 2 Fabric is reported as secondary.
  - **Open decoder question, to settle before TSFabrics scoring:** most masks are 0/255, but the C4 defect masks (T1_S174_I108_1) and some cutline masks contain intermediate values. The decoder threshold must be fixed from the mask files alone before any score exists.
- **About 04:50, TSFabrics run design, fixed before any TSFabrics detector output** (`code/p5_tsfabrics_run.py`, SHA-256 `c8ff30ad287c6770865a315d969ca85ec200138c54175ab0b135f7b7fe306cdc`; runs on a Kaggle GPU in the author's account):
  - **Test units:** the 7 positive clusters (C1, C2, C4, C5, C6, C7, C9).
  - **Frames per cluster:** every frame with a defect mask (capped at 300, uniform draw, seed 13017), an equal number of label-1 normal frames, and up to 50 label-2 cutline frames. Cutline frames count as normal for the defect metric, and their cutline masks are kept for a false-alarm analysis.
  - **Decoder:** a defect pixel is any value above 0 in the frame's defect masks. This equals the 128 threshold on 0/255 masks and keeps the grey-valued C4 masks. Fixed from the gate's mask-value summary, before any score.
  - **Bank, leave-cluster-out:** label-1 frames from every other cluster of the same texture. One random edge-aligned grid tile is taken from each of up to 1,000 frames (seeds 13018 and 13019), followed by the standard candidate sampling and the 10,240 coreset.
  - **Layouts:** R1, R2 (edge-aligned grid, 12 overlapping windows on 800 x 640), R2occ and C4. Practice A does not apply.
  - **Statistics:** each cluster has its own bank, so AP is computed within each cluster and averaged with equal cluster weights. The primary interval is a bootstrap over the 7 cluster differences, with the sign-flip test, the same treatment as RAW-FABRID rolls.
  - **Deviation from A5:** A5 described positive and negative clusters resampled as separate strata. Here the normal frames sit inside each positive test cluster, and the normal-only clusters supply banks rather than test units. T3 has no positive cluster and no second cluster for a texture-matched bank.

## 2026-11-22 05:05 TSFabrics runner input fixes (no change to scoring or statistics)
- First Kaggle submission failed before any frame was scored: `KeyError: 'FileName'` while reading the per-scenario CSV.
- Fix 1 (commit 4306933): read the CSV with `utf-8-sig` and strip header whitespace.
- Fix 2 (commit 80e05ea): the filename column is no longer assumed. The runner selects the `Label` column and the first column whose name contains file/name/image/frame/id, and prints the detected columns to the log. The gate script never read the filename column, so this name was an unverified assumption.
- From this run on, the Kaggle cell downloads `audit_experiments/code` at a pinned commit from the public GitHub repository instead of an embedded archive, so each Kaggle version maps to one commit.
- Neither fix touches tiling, scoring, decoding, AP computation or resampling. No TSFabrics score had been computed before these fixes.

## 2026-11-22 07:10 Interim status under the frozen decision rule (A6)
- RAW-FABRID: primary R2 - R1 = -0.0282 [-0.0422, -0.0185], 7/7 rolls negative, sign-flip p = 0.016. Meets the per-dataset criterion (negative).
- VisA: -0.0268 [-0.0311, -0.0230]. Meets the criterion (negative).
- AITEX: -0.0003 [-0.0028, +0.0010]. Does not meet the criterion.
- TSFabrics: running on Kaggle (versions 5 and 6, identical scoring code). The rule is met only if TSFabrics is negative with |est| >= 0.01 and a 95% interval excluding 0. Otherwise it fails. No interpretation is written until that result exists.
- MVTec AD 2 Fabric (secondary, outside the rule): -0.0023, object-clustered interval degenerate; reported descriptively only.
- No native full-frame (R1) run failed for memory on any dataset.

## 2026-11-22 08:05 TSFabrics complete; decision rule applied
- Version 5 (commit 4306933 state) and version 6 (pinned commit 80e05ea) both completed with 0 R1 failures. They agree to full printed precision on the primary block and to 6 decimals on every per-cluster AP checked.
- The CSV header was ['FileName', 'Label']: the first failure was the byte-order mark only, and fix 2 changed nothing in behaviour.
- TSFabrics primary: -0.0112 [-0.0199, -0.0011], 5/7 clusters negative, sign-flip p = 0.109.
- Rule 1 (A6): met. Datasets meeting the criterion, all negative: RAW-FABRID, VisA, TSFabrics. AITEX does not meet it. The TSFabrics margin is narrow and is reported as such.

## 2026-11-22 11:41 E1 (post-hoc, exploratory) complete; reading rule applied
- Run: six VisA categories per amendment 1 (candle, cashew, fryum, macaroni2, pcb2, pcb4), 40 test images each (25 anomalous, 15 normal), 240 in all. Design file hash unchanged (2a9c1a95...). Log: logs/e1_full.log; analysis: logs/e1_analysis.log, results_p5/e1_dino_size/full/analysis.json.
- Contrast 1 (context, DINOv2 Bfull/full - B256/full): +0.3854 [+0.3381, +0.4383], 6/6 positive.
- Contrast 2 (residual tiling, DINOv2 B256/256 - Bfull/full): -0.2214 [-0.2813, -0.1687], 0/6 positive.
- Contrast 3 (matched sweep vs B256/256): +0.073 (512), +0.117 (768), +0.221 (full); rising, not falling.
- Reading rule: H-context branch needs contrast 1 >= +0.05 AND contrast 2 within +/-0.03 -> not met (contrast 2 = -0.22). H-size branch needs contrast 1 < +0.05 OR matched sweep negative at every T and < -0.03 at full -> not met. Outcome by the rule: MIXED. Reported with the numbers; the data show a matched full-frame bank surpassing tiling, beyond what H-context predicted.
- Manuscript: Section 5.7 and Figure 12 added; Section 4.3 corrected (the pre-registered 2 x 2 bank-origin control was not run in Protocol 5); abstract, contribution 3, Sections 5.4 and 5.6, discussion, limitations and conclusion updated to the bank-mismatch explanation.

## 2026-11-22 15:30 Audit-driven corrections (after an external review of the manuscript and code)
- Code: exact_ap() now groups tied scores (matches scikit-learn; previously tie order depended on row order); the 1% patch threshold is 656 pixels (ceil(0.01 x 256^2)), not 655; bootstrap draws now use an independent, reproducible stream per unit (category, roll, cluster) via C.unit_seed(), still shared across the arms or layouts compared inside a unit. Tests: code/test_audit_fixes.py.
- These fixes change patch-level metrics and interval endpoints slightly. The affected analyses (p5_analyze.py for each dataset, p5_e1_dino_size.py --analyze) must be re-run on the machine that holds the per-image histograms, and the manuscript numbers updated from the new analysis files. The external reviewer's independent recomputation with independent category draws gave VisA primary -0.026799 [-0.030896, -0.023046], E1 bank effect +0.385377 [+0.337036, +0.440309], matched full minus tile +0.221374 [+0.169715, +0.277261]; point estimates are unchanged by construction.
- Known, not yet fixed: R2occ pools unmerged tile maps, so it changes scores as well as weights (described as such in the manuscript); E1 lacks the full-bank/tiled cell; TSFabrics runner has no per-file hashes and shares one RNG across clusters (restart hazard).
- Manuscript text, PDFs, Word and LaTeX files were removed from the public branch tip (kept privately); earlier commits still contain them.

## 2026-11-22 21:40 Re-analysis with the audit fixes
- Re-ran p5_analyze.py (AITEX, RAW-FABRID, MVTec AD 2 Fabric, VisA) and p5_e1_dino_size.py --analyze on the Mac from the saved per-image histograms. Pre-fix files kept in results_p5/_before_audit_fix/. A first attempt failed because the backups had been placed inside the result folders and were read as data units; they were moved out and the chain restarted. Field-by-field comparison: code/compare_analysis.py, logs/compare_analysis.txt.
- Result: every point estimate, per-unit AP, patch AUROC/AP and parent AUROC is unchanged (no field moved by more than 5e-5 other than interval endpoints). Only bootstrap interval endpoints moved, by at most 0.017 (MVTec AD 2, object bootstrap) and at most 0.004 for any interval quoted in the manuscript. RAW-FABRID roll-level intervals and the TSFabrics cluster-level intervals use roll/cluster resampling with the fixed seed and are unaffected.
- Updated: VisA primary -0.0268 [-0.0310, -0.0230] (was [-0.0311, -0.0230]); AITEX primary -0.0003 [-0.0025, +0.0010] (was [-0.0028, +0.0010]); E1 contrast 1 +0.3854 [+0.3374, +0.4377]; contrast 2 -0.2214 [-0.2778, -0.1687]. No conclusion or rule outcome changes. Manuscript Section 5.4, 5.6, 5.7, Supplementary Table S1 and Figures 7, 8, 11, 12 and the graphical abstract regenerated from the new files.
- The note at 15:30 that the fixes "change patch-level metrics" did not hold in practice: no saved patch score was tied and no patch had exactly 655 defect pixels.

## 2026-11-22 21:45 E2 designed and frozen (exploratory, new work)
- Design: P5_EXPLORATORY_E2_BANK_LAYOUT.md, SHA-256 20404d8b323d0d91a882cec861c47bed0842b5f1ed09077f5fcb44af704549f0, written after Protocol 5 and E1 results were known and before any E2 score. Code: code/p5_e2_bank_layout.py (checks the hash at start).
- Before writing the design, a probe confirmed on 22 reference tiles (Rollo1A, Rollo4A units) that each released RAW-FABRID reference tile is exactly the 256 x 256 square of its source frame at (256 x row, 256 x column) as encoded in its name; the runner re-checks every tile it uses pixel for pixel.
- Implementation detail within the design: each seed s is split with numpy SeedSequence(s).spawn(4) into the window pick, tile-bank sampling, full-frame-bank sampling and final pick; the final pick uses one stream for both banks of a seed, so their rows stay paired window by window.
- Caveat recorded before scoring: RAW-FABRID reference tiles are clean, but the rest of their source frames is not guaranteed clean, so full-frame bank vectors may carry defect context through the receptive field. Bank vectors are taken only from cells whose centre lies inside the clean tile.

## 2026-11-22 23:20 TSFabrics Kaggle version 6 notebook archived
- The saved notebook of Kaggle version 6 ("tsf run pinned 80e05ea") was downloaded by the author. Its printed output holds the full run log and the complete analysis.json; both are now archived as logs/p5_tsfabrics_kaggle_v6.log and results_p5/tsfabrics/full/analysis.json. The primary block matches the manuscript: -0.0112 [-0.0199, -0.0011], 5/7 clusters negative, sign-flip p = 0.109.
- The notebook does not contain the per-image histogram files, so per-file hashes and a re-analysis with the corrected code still need the version 6 Output files.
