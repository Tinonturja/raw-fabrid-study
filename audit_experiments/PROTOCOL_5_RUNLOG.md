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
