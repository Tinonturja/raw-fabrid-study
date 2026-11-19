# Does the crop ranking survive a change of fabric roll?

Both spatial detectors scored every one of the 709 original RAW-FABRID frames, with the complete test roll excluded from fitting. The primary test uses the same frames as the previous analysis. Each roll contributes equally to the estimate.

- **ResNet-18, roll held out.** The released-minus-fixed AP difference was +0.0129 on inspected pixels and +0.00043 on the whole frame. This did not meet the prespecified reversal rule. 3 of seven rolls individually met it.
- **Swin-T, roll held out.** The released-minus-fixed AP difference was +0.0084 on inspected pixels and +0.00045 on the whole frame. This did not meet the prespecified reversal rule. 4 of seven rolls individually met it.

## What was fixed before this run

The question arose after the original RAW-FABRID and AITEX experiments. Commit 2d6b9fc fixed the models, fold membership, sampling seed, inputs, score transformation, policies, endpoints and direction of the test before any holdout score was computed. Each model received the same 1,000 clean tiles per fold from the six eligible rolls. Tile count is per fold, and different folds can reuse tiles. The publisher archive and both ImageNet checkpoints were hashed. The release carries the positions of its defect-guided crops. The fixed scan uses 28 disjoint windows over the whole frame. [Protocol](unseen_roll_protocol.md) · [Recorded fitting and scoring code](../code/unseen_roll.py)

ResNet-18 matches local features to a 256-vector reference bank. Swin-T instead compares 32 selected features with a normal distribution fitted separately at each of 256 tile positions. Both preprocess the same source pixels and use the same 16 × 16 score lattice. This is a comparison of two complete detector configurations. The architecture and the scoring model change together. [Swin Transformer](https://arxiv.org/abs/2103.14030) · [PaDiM](https://arxiv.org/abs/2011.08785)

## Primary comparison

Pixel AP uses 202 defective and 53 clean frames. Frame AUROC uses the same 202 defective and 101 clean frames as the original analysis. For each roll the analysis computes both policy scores before taking their difference. The tables report arithmetic means across the seven rolls. A positive difference favors the released positions. The fixed grid observes the whole frame, so its observed and whole-frame AP are identical.

| Model and fit | Released observed AP | Released whole AP | Fixed AP | Δ observed | Δ whole | Released AUROC | Fixed AUROC | Rolls with reversal / 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-18, original fit | 0.2306 | 0.2147 | 0.2190 | +0.0116 | −0.0043 | 0.9358 | 0.9419 | 5 |
| ResNet-18, roll held out | 0.1854 | 0.1729 | 0.1725 | +0.0129 | +0.00043 | 0.8910 | 0.8909 | 3 |
| Swin-T, roll held out | 0.1214 | 0.1134 | 0.1130 | +0.0084 | +0.00045 | 0.8445 | 0.8541 | 4 |

## Sensitivity to the number of clean frames

All 709 originals enter both endpoints, including 507 clean frames. Pixel AP depends on foreground prevalence. Adding clean frames can change its absolute value and the size or sign of the policy difference.

| Model and fit | Released observed AP | Released whole AP | Fixed AP | Δ observed | Δ whole | Released AUROC | Fixed AUROC | Rolls with reversal / 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-18, roll held out | 0.1554 | 0.1449 | 0.1426 | +0.0128 | +0.0023 | 0.8636 | 0.8632 | 3 |
| Swin-T, roll held out | 0.0880 | 0.0822 | 0.0809 | +0.0072 | +0.0014 | 0.8348 | 0.8419 | 2 |

## Where the result varied

Each pair of AP differences below gives the observed-area value first and the whole-frame value second. The last count in the population column includes all original frames, irrespective of label.

### Original matched populations

| Roll | Defective / all frames | ResNet-18, original fit | ResNet-18, roll held out | Swin-T, roll held out |
| --- | --- | --- | --- | --- |
| 1A | 21 / 87 | +0.0075 / −0.0056 | +0.0076 / −0.0042 | +0.0076 / −0.0007 |
| 2A | 27 / 82 | +0.0070 / −0.0049 | +0.0077 / −0.0044 | +0.0033 / −0.0019 |
| 3A | 7 / 20 | +0.0360 / +0.0161 | +0.0244 / +0.0106 | +0.0255 / +0.0146 |
| 4A | 78 / 170 | +0.0162 / −0.0018 | +0.0220 / +0.0042 | +0.0252 / +0.0139 |
| 5A | 15 / 33 | +0.0137 / −0.0018 | +0.0067 / −0.0061 | +0.0017 / −0.0092 |
| 6A | 28 / 124 | +0.0186 / −0.0001 | +0.0238 / +0.0059 | +0.0009 / −0.0077 |
| 7A | 26 / 193 | −0.0177 / −0.0319 | −0.0018 / −0.0030 | −0.0051 / −0.0059 |

### All originals

| Roll | Defective / all frames | ResNet-18, roll held out | Swin-T, roll held out |
| --- | --- | --- | --- |
| 1A | 21 / 87 | +0.0082 / −0.0016 | +0.0101 / +0.0043 |
| 2A | 27 / 82 | +0.0082 / −0.0012 | +0.0024 / −0.0007 |
| 3A | 7 / 20 | +0.0223 / +0.0112 | +0.0175 / +0.0106 |
| 4A | 78 / 170 | +0.0226 / +0.0067 | +0.0223 / +0.0137 |
| 5A | 15 / 33 | +0.0079 / −0.0037 | +0.0009 / −0.0077 |
| 6A | 28 / 124 | +0.0218 / +0.0061 | −0.0012 / −0.0085 |
| 7A | 26 / 193 | −0.0013 / −0.0016 | −0.0017 / −0.0019 |

### Omit-one-roll sensitivity

Each range is the minimum and maximum of the seven six-roll means. It measures dependence on a particular roll. The training sets overlap, and the ranges have no confidence-interval interpretation.

| Population | Model and fit | Δ observed AP | Δ whole AP | Δ frame AUROC |
| --- | --- | --- | --- | --- |
| primary | ResNet-18, original fit | +0.0075 to +0.0165 | −0.0077 to +0.0003 | −0.0101 to −0.0031 |
| primary | ResNet-18, roll held out | +0.0110 to +0.0154 | −0.0013 to +0.0015 | −0.0062 to +0.0036 |
| primary | Swin-T, roll held out | +0.0056 to +0.0107 | −0.0019 to +0.0021 | −0.0181 to +0.0046 |
| all | ResNet-18, roll held out | +0.0112 to +0.0151 | +0.0008 to +0.0033 | −0.0081 to +0.0045 |
| all | Swin-T, roll held out | +0.0047 to +0.0087 | −0.0007 to +0.0030 | −0.0156 to +0.0082 |

## Interpretation and reproducibility

The released windows omitted 140,701 of 1,904,926 marked native pixels (7.39%) for every detector. Coverage follows from the rectangles and masks. Both held-out mean whole-frame differences were below 0.0005 in magnitude on the primary cohort. Both signs occurred among the delete-one-roll means. The observed whole-frame estimates offer little basis for preferring either policy on that cohort. The sign of an AP difference also depends on how the detector ranks all scored foreground and background pixels. A small whole-frame difference can therefore change sign even when coverage is identical.

The earlier pooled ResNet-18 result was +0.0108 observed and −0.0042 whole-frame AP at seed 13017 (rounded). Reweighting those same saved predictions equally across rolls gives +0.0116 and −0.0043. Comparing that row with the held-out ResNet row holds the evaluation frames and group weighting constant. Removing the test roll also changes which clean tiles are available to construct the reference bank. No raw score ranks are compared across newly fitted folds.

The original archive SHA-256 is f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9. On an Apple M1, PyTorch used the GPU through MPS for network inference and distance scoring. Covariance estimation and count-based statistics ran on CPU. Only the clean reference distributions were fitted. Each output file records the device, library versions, weight hash, reference selection, fitted-parameter hashes, frame scores and pixel counts.

The analysis reproduces every AP two ways, using scikit-learn with bin counts as sample weights and a direct precision–recall calculation. It checks coverage against the source totals and verifies that the two detectors share an identical reference order, test membership and per-frame mask accounting within each fold. A separate check compares Gaussian scores with direct linear-system solutions and native overlap histograms with direct pixel maps. The worked image-based repeat uses the smallest roll, 3A. [Reproduction commands](../README.md#spatial-detectors-on-unseen-rolls) · [Full numeric results and fold records](../results/unseen_roll)

For a computer-vision paper, the supported contribution is the controlled evaluation of what guided cropping exposes and how conclusions change with the detector, split and evaluation area. Seven rolls of one fabric construction provide limited evidence across production settings. A prospective experiment on a different production line with known inspected length and a fixed false-alarm budget would address a different, operational claim.
