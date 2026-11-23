# Protocol 3: does gap-fill still win against extra looks matched on defect proximity?

Written 21 Nov 2026 (about 01:45 Dhaka), after Protocol 2 results. Post hoc; the question and rule are fixed before any Protocol 3 output exists.

## Problem
Protocol 2 showed gap-fill beats budget-matched nearby extra looks (C1), but gap-fill's extra windows raise scores on already-seen pixels about four times more than C1's. Gap-fill windows may simply re-examine more defect pixels.

## Control C4 (proximity-matched extra looks)
For each frame: let k = number of gap-fill extra windows and, for any window w, f(w) = number of marked defect pixels inside w that the released layout already sees. Candidates: stride-128 windows not in the released layout that contain no unseen pixel. Taking gap-fill extras in decreasing f, assign to each the unused candidate with the closest f (random tie-break). C4 = released + these k windows. Five draws per frame (seeded by frame and draw). C4 adds second looks at as many seen defect pixels as gap-fill does, but no coverage.
Matching quality (mean f of gap-fill extras vs C4 extras) is recorded per frame and reported.

## Primary question
Mean over 7 rolls of whole-frame AP(gap-fill) - AP(C4), per detector, exact two-sided sign-flip test.
Rule: a coverage-specific benefit may be claimed only if this is positive with p <= 0.05 for both r18_native and wrn50, AND C4's matched seen-defect pixel total is at least 90% of gap-fill's (otherwise matching failed and the test is inconclusive).

## Secondary
Seen-pixel AP: gap-fill - C4 (expected near zero if matching worked); C4 - C1.

## Design
Identical to Protocol 2 (folds, cohorts, masks, detectors, verified cached banks, provenance hashes, stop on first failure). Detectors: r18_native and wrn50 only.
