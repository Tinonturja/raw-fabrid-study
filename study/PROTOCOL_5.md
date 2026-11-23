# Protocol 5 (FROZEN 21 Nov 2026): do cutting protocols change what anomaly benchmarks measure?

Status: approved by the author on 21 Nov 2026 (22:13, Asia/Dhaka) and frozen before any detector output on the new datasets. SHA-256 of this file is recorded in PROTOCOL_5.sha256. Any later change goes in a dated deviation log at the end, never by editing the text above it.

## Claim to test
How a benchmark cuts high-resolution test images (A defect-aware crops; B blind grid with per-patch labels; C whole image resized; reference: full-frame grid scan scored at native resolution) changes reported detection and localization scores, and can change which detector looks best.

## Datasets (decided 21 Nov 2026 with the author; parent images and pixel masks available, so every protocol runs on the same pixels)

| Group | Dataset | Frame size | Why it is in | Where it runs |
|---|---|---|---|---|
| Real fabric | RAW-FABRID | 1792 x 1024, line-scan | Released defect-aware crops exist; exact 7 x 4 tiling leaves gaps (7.39% of marked pixels unseen) | Mac |
| Real fabric | AITEX | 4096 x 256 strips | Woven fabric, released per-patch relabelling practice (B) | Mac |
| Real fabric | TSFabrics (Scientific Data 2026, 13:379) | 640 x 800 grayscale video frames (from the author's pilot code; confirm on first run) | Knitted fabric from circular knitting machines; pixel masks; production cutlines (label 2) are normal and act as defect look-alikes. 800/256 is not whole, so an edge-aligned grid overlaps: a geometry contrast to RAW-FABRID. No released crops, so its defect-aware layout is constructed by us | Kaggle (author's copy of the dataset; no local download) |
| General benchmark | VisA (CC BY-NC-SA 4.0) | high-resolution objects | Shows the effect is not fabric-specific | Mac |
| Backup | MVTec AD 2, Fabric | 2448 x 2048 | Used only if a real-fabric dataset fails; the author judged its printed, staged defects unrealistic. Private test folders not used | Mac |
| Survey only | ZJU-Leaper | 512 x 512 pre-cut images | Cited as an example of a patch-only release; no full frames, so no cutting can be studied. Not run | none |

Dropped from the earlier draft: MVTec AD 2 Sheet Metal (VisA already covers non-textile objects).

## Reference scans (approved 21 Nov 2026)
- Tile-free full-frame scan: the frozen backbone is run on the whole image at native resolution (fully convolutional), so there are no square borders, no gaps and no slivers. This is the "no cutting" ground truth every cutting practice is compared against. If memory does not allow Wide-ResNet-50 at full size on the Mac, it runs on ResNet-18 locally and Wide-ResNet-50 on a cloud GPU.
- Full-coverage grid scan at native resolution (edge-aligned last row and column where sizes do not divide by 256).

## Detectors (training-free, three different families; run on the Mac, and on Kaggle for TSFabrics)
- PatchCore, Wide-ResNet-50, native resolution, 1% coreset (as in Protocols 1 to 4).
- PaDiM-style Gaussian scorer on the same features (different scoring rule).
- AnomalyDINO-style DINOv2 patch kNN (foundation-model features, 2025).
Optional on Kaggle GPU: EfficientAD or Dinomaly (need training).

## Primary question (one)
Does the blind-grid patch-level score (practice B) differ from the score of the tile-free full-frame reference scan?

- Primary metric: pixel AP. B is scored on its released or constructed patches with per-patch labels; the reference is scored over every pixel of the full frame.
- Primary detector: PatchCore, Wide-ResNet-50. The other two detectors are used only for the ranking claim and as secondary results.
- Main datasets: RAW-FABRID, AITEX, TSFabrics, VisA. MVTec AD 2 Fabric enters only if one of these fails to run, and the substitution is logged.

Decision rules (approved 21 Nov 2026):
1. Measurement claim: "cutting changes what the benchmark measures" holds only if (B - reference) pixel AP has the same sign on at least 3 of the 4 main datasets, and the 95% confidence interval excludes zero on each of those 3 datasets.
2. Ranking claim: "cutting can change which detector looks best" holds only if the order of the three detectors by pixel AP differs between B and the reference on at least one dataset, and on that dataset the pairwise difference that swaps has a 95% interval excluding zero under both B and the reference. Otherwise report the ranking as unchanged.
3. Anything failing these rules is reported as not shown, in the main text, not dropped.

## Uncertainty
- RAW-FABRID: leave-one-roll-out, equal-roll means, exact sign-flip test over 7 rolls (as in Protocols 1 to 4), plus a whole-frame bootstrap for intervals comparable with the other datasets.
- AITEX and VisA: bootstrap over whole images, 2,000 resamples, percentile 95% intervals; detectors and memory banks fixed, only test images resampled.
- TSFabrics: bootstrap over whole video clips (cluster bootstrap), 2,000 resamples, because neighbouring frames are nearly identical.
- Seeds: bootstrap seed 13017; coreset seed fixed per dataset and recorded.

## Secondary
Practice A coverage cost (where crops exist or are constructed); practice C small-defect loss; sliver-patch share; second-look effect.
