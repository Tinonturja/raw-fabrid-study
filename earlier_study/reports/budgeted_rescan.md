# Where should a fabric scanner take a second look?

On the primary RAW-FABRID split, a second pass aimed at suspicious pixels recovered 87% of the whole-frame localization improvement from a dense scan, using 35 windows per frame compared with 91. Applying the same rule to AITEX reduced localization performance. At each policy's clean-calibrated alarm threshold, evenly spaced extras found more distinct regions on both collections.

## What the scanner knew

Every frame first received a complete grid of disjoint windows. The detector measured how unusual each patch of fabric looked relative to clean training fabric. Each potential extra window received the average of its four most unusual first-pass cells. We rescanned the highest-ranked windows. Defect annotations were used afterward to measure performance. No annotation or defect status entered the selection rule.

We specified the budget, ranking rule, comparisons, endpoints and group deletions before computing the new outcomes. The earlier results on these same images were already known. The original primary split, seed 13017, was reused on each collection. Extra windows equalled 25% of the first-pass budget. The equal-budget control placed its extras at preset intervals in the list of possible positions. Both policies saw every pixel at least once. [Dated protocol in Git](rescan_protocol.md) · [Selection and scoring code](../code/budgeted_rescan.py)

## Paired outcomes

Pixel average precision (AP) measures how highly the detector ranks marked pixels across complete images. Frame AUROC measures how often it ranks a defective frame above a clean one. Each policy's alarm threshold was set using clean calibration frames. A region is found if any of its pixels exceeds that threshold.

| Dataset and policy | Windows per frame | Whole-frame pixel AP | Frame AUROC | Defective frames flagged | Distinct regions found | Clean false alarms |
|---|---:|---:|---:|---:|---:|---:|
| RAW-FABRID, fixed grid | 28 | 0.1965 | 0.9458 | 175/202 | 224/371 | 4/53 |
| RAW-FABRID, targeted extras | 35 | 0.2164 | 0.9474 | 169/202 | 216/371 | 1/53 |
| RAW-FABRID, evenly spaced extras | 35 | 0.1995 | 0.9471 | 176/202 | 225/371 | 4/53 |
| RAW-FABRID, dense grid | 91 | 0.2193 | 0.9537 | 172/202 | 219/371 | 1/53 |
| AITEX, fixed grid | 16 | 0.0920 | 0.9106 | 78/103 | 70/175 | 3/31 |
| AITEX, targeted extras | 20 | 0.0866 | 0.8367 | 63/103 | 41/175 | 6/31 |
| AITEX, evenly spaced extras | 20 | 0.0928 | 0.8827 | 66/103 | 48/175 | 1/31 |
| AITEX, dense grid | 31 | 0.0943 | 0.8357 | 64/103 | 41/175 | 6/31 |

The primary paired difference in whole-frame AP was **+0.0170 on RAW-FABRID** and **−0.0062 on AITEX**, targeted minus evenly spaced. For RAW-FABRID the difference stayed positive whenever one of seven rolls was omitted (+0.0129 to +0.0197). Five of six AITEX fabric deletions retained a negative difference. Omitting fabric 01 gave +0.00045, and omitting fabric 02 gave −0.02197. Each deletion rescored the pooled evaluation population with its original model. Repeating these computations does not create new independent fabrics. [RAW-FABRID scores and histograms](../results/budgeted_rescan_raw.json) · [AITEX scores and histograms](../results/budgeted_rescan_aitex.json)

The RAW-FABRID result illustrates why a localization score alone is insufficient to choose an alarm policy. Targeting improved pixel ranking, yet at its calibrated cutoff it flagged seven fewer defective frames and hit nine fewer distinct regions than the evenly spaced control. Its cutoff also rose from 1.4761 to 1.5290, and clean false alarms fell from four to one. The two AITEX policies produced different realized false-alarm rates too. These operating points cannot settle which policy detects more defects at an identical production false-alarm rate.

## Did the second look reach a real defect?

After observing the opposite AP results, we registered the saved extra-window positions against the original defect masks. For this diagnostic, count each marked pixel once even if several extra windows cover it. The first pass already saw the entire frame. These numbers count the marked fabric that received a second look.

| Dataset | Policy for extra windows | Defective images where an extra window touched a defect | Marked pixels revisited |
|---|---|---:|---:|
| RAW-FABRID | Targeted | 194/202 | 1,145,556/1,904,926 (60.14%) |
| RAW-FABRID | Evenly spaced | 98/202 | 342,973/1,904,926 (18.00%) |
| AITEX | Targeted | 75/103 | 175,823/485,027 (36.25%) |
| AITEX | Evenly spaced | 30/103 | 117,353/485,027 (24.20%) |

Targeting put more marked fabric inside the extra windows on both datasets. On AITEX that greater access still produced poorer pixel rankings, fewer region hits, and more clean false alarms. A simple explanation that the scanner merely looked in the wrong place cannot fully account for the AITEX result. A future experiment could measure how rescanning changes scores on specific types of defects and benign texture. This diagnostic alone cannot identify that mechanism. [Per-image mask counts](../results/rescan_coverage.json) · [Native-mask audit code](../code/audit_rescan_coverage.py)

## Hardware, model and reproducibility

The original global experiment and all three original RAW-FABRID spatial runs recorded PyTorch MPS. The original AITEX run and both new image-based rescan experiments also used the Apple M1 GPU. The new files record device `mps:0` and PyTorch 2.15.0. Image decoding, native mask measurement and the arithmetic verification used the CPU. The scanner uses 256 × 256 native tiles, resized to 128 × 128 for the network. The work count in the table is the number of network windows for each policy. Production latency and energy remain unmeasured.

Both deep detectors use an ImageNet-pretrained **ResNet-18** with frozen weights. The original global detector pools a tile into a 512-number vector and measures distance to the nearest clean reference. The follow-up uses the **spatial PatchCore variant**. It combines ResNet layers 2 and 3 into a 16 × 16 grid of 256-number vectors per tile. Each vector is compared to a compact bank of 256 representative normal vectors, fitted separately for each dataset. That preserves a location for the anomaly response. The pretrained network weights were held fixed throughout. [Implementation](../code/localization_study.py) · [Torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) · [PatchCore authors' paper](https://arxiv.org/abs/2106.08265) · [Apple GPU support in PyTorch](https://developer.apple.com/metal/pytorch/)

The refits selected exactly the same clean reference indices as the original spatial runs in both datasets. All reconstructed fixed- and dense-grid frame scores, whole-frame APs, frame AUROCs, region-hit counts and false-alarm counts matched the original controls. The standard-library verifier independently recomputes metrics from the saved predictions, checks the window counts, and recalculates the group deletions.

For a clean rerun we cloned code commit `2d142ce` into a temporary folder, removed all three follow-up result files from that copy, and supplied the same source archives, installed environment and pretrained weights. Both GPU runs and the mask audit regenerated all three JSON files byte for byte. A separate calculation intersected individual foreground pixel coordinates with each window and matched every saved revisit count across all 305 defective images.

From the project root, verify the stored evidence without downloading images:

```bash
python3 code/verify_results.py --external --rescan
```

To repeat inference on this Mac, use the installed project environment and the original archives in the locations described in the [project README](../README.md).

```bash
.venv/bin/python code/budgeted_rescan.py raw --device mps
.venv/bin/python code/budgeted_rescan.py aitex --device mps
.venv/bin/python code/audit_rescan_coverage.py
```

Omitting the device flag selects an available Apple GPU, NVIDIA CUDA device, or CPU in that order. The recorded numerical comparisons use the M1 environment. The experiment scores additional windows to evaluate the control policies alongside the targeted scanner.

## Publication scope

This is evidence for a supplementary experiment on the same two cohorts. One deterministic evenly spaced control and one targeted rule were tested. A practical next test would freeze both policies and the model, collect consecutive images from previously unseen rolls, and compare region detection at a common false-alarm budget with synchronized wall-clock timing. The existing data support a paper about how fabric inspection is evaluated. A claim about factory throughput or production false alarms requires that additional evidence.
