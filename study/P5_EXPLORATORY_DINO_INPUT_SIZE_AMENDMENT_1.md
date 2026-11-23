# E1 amendment 1 (22 November 2026, 08:40 Asia/Dhaka), before any E1 image was scored

- **Reason:** bank construction for whole-frame DINOv2 windows took about 14 minutes per category on the Mac's GPU, so the full 12-category run would not finish today.
- **Change 1, scope:** E1 runs on every other VisA category in sorted order: candle, cashew, fryum, macaroni2, pcb2, pcb4 (`sorted(categories)[::2]`). The rule uses names only, not any Protocol 5 or E1 result. All other design choices (test subset, banks, arms, contrasts, reading rule) are unchanged.
- **Change 2, code only, no effect on values:** features are computed once per image and inference configuration and reused for every bank, instead of being recomputed per arm.
- **Status at the time of change:** banks for candle had been built and no test image had been scored (`results_p5/e1_dino_size/smoke/` was empty).
