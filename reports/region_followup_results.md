# Region overlap under a low false-positive limit

This analysis replays all seven saved RAW-FABRID held-out rolls. The protocol was written after the earlier experiments and before these results. It uses 202 defective and 53 clean primary pixel-evaluation frames and one score distribution per roll.

At each score threshold, compute the fraction of each eight-connected mask component that lies above threshold. Average those fractions equally across components within a roll. The false-positive rate uses all non-foreground pixels in that roll. Integrate to 0.05 by trapezoidal interpolation and divide by 0.05. Uncovered pixels receive the lowest score, and thresholds traverse tied bins together.

| Model | Released mean | Fixed mean | Released minus fixed |
|---|---:|---:|---:|
| resnet18 | 0.5627 | 0.5573 | +0.0053 |
| swin_t | 0.4614 | 0.4700 | -0.0086 |

| Roll | ResNet released | ResNet fixed | Swin-T released | Swin-T fixed |
|---|---:|---:|---:|---:|
| Rollo1A | 0.6884 | 0.6684 | 0.5127 | 0.5224 |
| Rollo2A | 0.5632 | 0.5279 | 0.3972 | 0.3787 |
| Rollo3A | 0.5603 | 0.5689 | 0.5257 | 0.5679 |
| Rollo4A | 0.5500 | 0.5608 | 0.4784 | 0.4811 |
| Rollo5A | 0.6241 | 0.6660 | 0.5834 | 0.5847 |
| Rollo6A | 0.7589 | 0.7228 | 0.5789 | 0.5517 |
| Rollo7A | 0.1937 | 0.1867 | 0.1538 | 0.2037 |

Each per-roll result includes its saved pixel-background and region-mass bins. Replayed foreground and background bins match the earlier pixel-AP histograms exactly. A synthetic perfect-ranking case, reversed-ranking case and tied-score case yielded 1.0, 0.0 and 0.025 respectively. The repeated folds share training material, so no independence-based statistical test is implied. The metric is related to AU-PRO at 5% FPR, with the treatment of unobserved pixels specified in this study.

[Protocol](region_analysis_protocol.md) · [Saved per-roll bins and summary](../results/region_followup) · [MVTec AD 2](https://doi.org/10.1007/s11263-026-02743-0)
