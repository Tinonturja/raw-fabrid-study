# Protocol 4 (Step 3): can a label-free scanner recover the benefit?

Written 21 Nov 2026 (about 11:40 Dhaka), before any Protocol 4 output. Post hoc relative to Protocols 1 to 3.

## Question
Gap-fill needs the dataset makers' label-informed windows, so a real machine cannot use it. Does a label-free two-pass scan do better than the fixed grid at the same small window budget?

## Policies (applied identically to every frame, defective or clean, so frame AUROC and false alarms are fair)
- fixed: 28-window grid.
- targeted+k: fixed grid first; then add the k stride-128 windows (not already in the grid) whose area holds the highest first-pass score (maximum of the fused fixed-grid map inside the window). Uses only the detector's own output.
- random+k: fixed grid + k random stride-128 windows not in the grid (5 draws, seeded by frame name).
- overlap91: all 91 stride-128 windows (upper reference).
- k in {2, 7}: 2 matches gap-fill's average budget; 7 matches the author's earlier 25% rescan budget.
For context only (not compared statistically): released and gap-fill.

## Primary question (one)
Mean over 7 rolls of whole-frame AP(targeted+2) - AP(random+2), per detector, exact sign-flip test.
Rule: claim a useful label-free targeting effect only if positive with p <= 0.05 for both r18_native and wrn50.

## Secondary
targeted+2 - fixed; targeted+7 - random+7; targeted+7 - fixed; frame AUROC for each; overlap91 - targeted+7.

## Design
Same folds, cohorts, masks, detectors, verified cached banks, provenance hashes and stop-on-failure as Protocols 2 and 3.
