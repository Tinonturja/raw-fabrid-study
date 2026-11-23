# Protocol 2: is the gap-fill gain about coverage, or just extra looks?

Written 20 November 2026 (Dhaka evening), after Protocol 1 results and after an external review showed that gap-fill improves AP even on pixels the released layout already sees (all 7 rolls, all 3 detectors). That finding motivates this test, so the study is post hoc relative to Protocol 1. The question and the decision rule below are fixed before any Protocol 2 output exists.

## Problem
Gap-fill = released windows + the fixed-grid windows that touch pixels the released layout never sees (k extra windows per frame, k = 0 to 8). Overlapping windows are combined by taking the maximum score per pixel, so each extra window both (a) scores unseen pixels and (b) gives already-seen pixels a second look. Protocol 1 could not separate (a) from (b).

## Controls (all use exactly k extra windows on the same frame, so the window budget is matched)
- C1 nearby extra looks: released + k windows from the stride-128 grid that overlap a moved (off-grid) released window but contain no unseen pixel. Same neighbourhood as gap-fill extras, no coverage gain. If fewer than k candidates exist, all candidates are used and the shortfall is recorded.
- C2 random extra looks: released + k stride-128 windows anywhere that contain no unseen pixel.
- C3 label-free: fixed grid + k random stride-128 windows not already in the grid. Uses no label information at all.
- Each control is drawn 5 times per frame (seeded by frame name and draw number); AP is computed per draw and averaged.
Clean frames have k = 0, so all policies coincide there.

## Primary question (only one)
Mean over 7 rolls of whole-frame AP(gap-fill) - AP(C1), per detector, with the exact two-sided sign-flip test over rolls.
Decision rule: coverage restoration adds value beyond extra nearby looks only if this difference is positive with p <= 0.05 for BOTH stronger detectors (r18_native and wrn50). Otherwise the paper may not claim a coverage-specific benefit of gap-fill.

## Secondary and exploratory (reported, not used for the decision)
- gap-fill - C2, gap-fill - C3, C3 - fixed.
- Coverage-only hybrid: released scores on seen pixels S, gap-fill scores on unseen pixels U (from saved histograms).
- Region hits are not reported: per-roll clean calibration sets (1 to 16 frames) are too small for a meaningful threshold.

## Fixed design
Same folds, cohorts (255 pixel frames, 303 AUROC frames), masks (>= 128), detectors and settings as Protocol 1 final configuration: orig_r18 (author code, refitted); r18_native and wrn50 (128 sampled positions per training tile, 10,240-vector coreset = 8% of 128,000 candidates, Euclidean k=1, bilinear + Gaussian sigma 4, max fusion), using the Protocol 1 cached banks after hash verification against Protocol 1 records. Each output records code, bank and pretrained-weight hashes. The run stops on the first failure.
