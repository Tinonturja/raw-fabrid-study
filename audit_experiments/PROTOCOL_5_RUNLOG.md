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
