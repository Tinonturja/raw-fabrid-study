# Exploratory direct transfer to FabriD v3

Both previously fitted RAW-FABRID models were applied directly to the FabriD v3 test partition. The experiment scored 358 images, including 270 with polygon-derived foreground and 88 with empty labels. The polygons produced 373 eight-connected foreground regions. Both models used the same RAW-FABRID reference tiles from the saved held-out-Rollo3A fold and were not refitted on FabriD.

| Detector | Windows | Frame AUROC | Whole-frame AP | Observed-area AP | Region overlap at 5% FPR |
|---|---|---:|---:|---:|---:|
| ResNet-18 spatial | Fixed | 0.4521 | 0.0297 | 0.0297 | 0.0533 |
| ResNet-18 spatial | Mask-guided | 0.4325 | 0.0343 | 0.0343 | 0.0538 |
| Swin-T Gaussian | Fixed | 0.4621 | 0.0280 | 0.0280 | 0.0967 |
| Swin-T Gaussian | Mask-guided | 0.4516 | 0.0366 | 0.0366 | 0.1265 |

The mask-guided positions use target labels and serve as an oracle comparison. They omitted 181 of 2,013,654 polygon-derived marked pixels (0.0090%). The fixed layout covered every frame. The direct-transfer frame AUROC point estimates were below 0.5 for both layouts and detectors. Their uncertainty was not estimated in this exploratory run. Pixel AP depends on the target dataset's defect prevalence and annotation practice.

The released ZIP contained 14 source identifiers appearing in more than one published split, using the filename before the Roboflow .rf. suffix. All 358 test images had different source identifiers. No FabriD training or validation images were used for fitting. Physical scene, roll, manufacturer and acquisition grouping remain unverified, so images should not be treated as independent production rolls. The publisher describes a different image collection with polygon labels, but this does not prove independent factories or rule out reuse of source material.

The publisher ZIP hash matched 4944cec7f54ce3f9120ed243ea8c33f3ff3a3d4c39556fd06e7cfe3a5b3a05c3. The verifier recomputed AP from the saved score bins with scikit-learn, frame AUROC from saved scores and region overlap using a separately implemented trapezoidal integration. The supplied JSON contains the full frame scores and per-policy histograms. The protocol was recorded after the dataset description, archive file list, and several polygon-label entries were inspected but before image decoding and scoring.

To reproduce, place the original publisher ZIP at data/fabrid_v3.zip and retain data/RAW_FABRID.zip, then run:

    .venv/bin/python code/fabrid_external.py
    .venv/bin/python code/verify_fabrid_external.py

[Protocol](fabrid_external_protocol.md) · [Full saved results](../results/fabrid_external.json) · [FabriD v3 publisher record](https://doi.org/10.17632/vncdwps24r.3)
