# Crop placement in fabric anomaly detection

Tinon Turja Majumder, United International University, Bangladesh, maintains this project. The [software citation](CITATION.cff) records the preferred attribution. Read the [manuscript PDF](manuscript/manuscript.pdf), edit the [Word document](manuscript/manuscript.docx), or browse the [Markdown source](manuscript/manuscript.md).

This research repository compares what a detector sees when defect annotations guide its test windows and when it scans the complete fabric frame. The Markdown manuscript, source code, figures and scored results are included. Large image archives, pretrained weights, feature caches and the local Python environment are kept on the research machine.

**Start with the [draft manuscript](manuscript/manuscript.md).** On RAW-FABRID v1, guided crop placement improved a globally pooled detector's frame ranking. A spatial PatchCore variant favored a blind fixed grid on all three matched splits. In the original within-roll splits, the pixel-localization ranking changed when the evaluation included marked fabric outside the released windows. With entire rolls held out, both ResNet-18 and Swin-T retained an observed-area gain, while the mean whole-frame differences were each below 0.0005 on the matched cohort. The [publication assessment](reports/publication_scope.md) documents the literature search and the exploratory work preceding the spatial experiments.

**For a first reading, open the [25-page learning guide](output/pdf/fabric_inspection_research_guide.pdf).** The [editable learning notes](reports/learning_notes.md) cover the same lessons. Read pages 3–10 for the fabric and metrics, pages 11–17 for code and practical labs, and pages 19–24 for the full methodology diagram, grouped statistics, second-look experiment and questions with answers. Page 25 explains the unseen-roll test and how to reproduce it. The [Q1 submission assessment](reports/q1_submission_assessment.docx) explains venue fit and the limits of the evidence.

The [follow-up on targeted rescanning](reports/budgeted_rescan.md) tests how to spend 25% more inspection windows after the first pass. It includes the GPU and model details, paired results on both collections, and instructions to repeat the experiment.

## Check the results without image data

From the project root, run the following command with Python 3. It uses the standard library and the committed scores.

```bash
python3 code/verify_results.py
```

The check independently recalculates matched frame AUROCs, calibration thresholds, false alarms, region hits and pixel average precision, and compares native coverage records across files. A [clean RAW-FABRID rebuild](reports/clean_rebuild.json) reproduced the five primary evidence files exactly apart from elapsed times. It started from the original ZIP with empty feature and result directories. Public-ready outputs live in [results](results). The primary spatial analysis is [seed 13017](results/localization_study.json). The [protocol](reports/localization_protocol.md) and [secondary analysis decision](reports/localization_repeats.md) document the sequence of decisions.

## Rebuild RAW-FABRID from its original images

The reference runs used Python 3.13, the packages pinned in [requirements-audit.txt](requirements-audit.txt), an Apple M1 and PyTorch MPS. CPU execution is supported by the code. Numeric differences can occur across platforms. The raw ZIP is about 4.13 GB. The scripts read images directly from it. The package install and model weights require additional disk space.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-audit.txt
.venv/bin/python code/reproduce.py --download
```

If the archive is already at `data/RAW_FABRID.zip`, omit `--download`. For a manual download, use the publisher's version 1 ZIP. Its expected SHA-256 is `f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9`. The preparation step verifies the file and reads its four metadata files. The workflow checks member CRCs, audits image and mask pairs, extracts both feature sets, evaluates the three splits, runs all crop controls and spatial analyses, and regenerates the research figures. A previous completed global feature cache can accelerate reruns. Use a fresh checkout for a clean rebuild. [RAW-FABRID publisher record, CC BY 4.0](https://data.mendeley.com/datasets/db6g85xsyg/1)

## Main files

| File | Purpose |
|---|---|
| [Manuscript](manuscript/manuscript.md) | Research question, methods, results, limitations and references |
| [Full RAW-FABRID workflow](code/reproduce.py) | Original archive to audited metrics |
| [Pixel coverage check](code/check_release_coverage.py) | Pixel equality of all 19,852 patches, native mask coverage and component losses |
| [Crop reassignment experiment](code/check_crop_publication_scope.py) | Released, fixed, same-roll donor and overlapping controls |
| [Spatial localization experiment](code/localization_study.py) | Coreset fit, matched inference, native map fusion, coverage-sensitive scoring |
| [Score verification](code/verify_results.py) | Arithmetic checks from shareable result files |
| [Initial audit](reports/research_update.md) | Dataset provenance and first global and texture baselines |

The machine-readable results include source filenames, model scores and aggregate pixel counts. The mask example in the manuscript is adapted from RAW-FABRID v1 and credited to Rubén Pérez-Llorens, Teresa Albero-Albero and Javier Silvestre-Blanes under CC BY 4.0. The `research_sha256.txt` report records the earlier exploratory snapshot, including historical code digests. Source code for the analyses is under `code/`.

## Reproduce the external AITEX test

The [external protocol](reports/aitex_protocol.md) specifies the constructed guided layout, fixed and donor controls, training split and evaluation denominators. The [result file](results/aitex_external.json) contains all eligible filenames, image hashes, actual guided coordinates and outcomes. The tested publisher archive fingerprints are listed in [aitex_archives.json](reports/aitex_archives.json).

With the same Python environment and the `bsdtar` program (included on macOS, provided by libarchive-tools on many Linux distributions), run:

```bash
.venv/bin/python code/external_aitex.py --download
python3 code/verify_results.py --external
```

Alternatively, place the three original publisher archives in `data/aitex/` and omit the download flag. The analysis reads temporary extracted copies. The public repository contains derived measurements. Original AITEX files remain available from the [publisher](https://www.aitex.es/afid/).

## Grouped statistics and publication figures

The primary-split analysis resamples entire rolls or fabrics with paired predictions and a fixed reference bank. The [estimates, percentile intervals and group deletions](results/cluster_statistics.json) and [new figures in PNG and vector PDF](manuscript/figures) accompany the manuscript. From the project root, reproduce them from the committed counts and scores with the pinned environment.

```bash
.venv/bin/python code/cluster_statistics.py
.venv/bin/python code/plot_study_design.py
```

To rebuild the guided counts from the original archives, run `.venv/bin/python code/rebuild_group_histograms.py` first. The program checks every pooled pixel-score bin against the original saved run. The GPU reconstruction and the grouped analysis use primary seed 13017.

## Spatial detectors on unseen rolls

The [holdout protocol](reports/unseen_roll_protocol.md) fixes the two models, training references, crop policies and sign rule. The [results and interpretation](reports/unseen_roll_results.md) include both cohorts and all seven rolls. The original article and overview can be read without downloading the image data.

Using the pinned environment, recompute both AP denominators and frame AUROC from the [committed per-roll scores and counts](results/unseen_roll), then redraw the figure.

```bash
.venv/bin/python code/analyze_unseen_roll.py
.venv/bin/python code/plot_unseen_roll.py
```

To rebuild both detectors on all seven rolls from the publisher archive, run the first command below. The next two commands demonstrate a separate rebuild of the smallest roll, including comparison of all scores and count bins. The first run requires the torchvision ImageNet-1K V1 checkpoints for ResNet-18 and Swin-T. PyTorch uses the Apple GPU, a CUDA GPU or the CPU according to availability. Different hardware or library versions can change floating-point scores.

```bash
.venv/bin/python code/unseen_roll.py
.venv/bin/python code/unseen_roll.py --roll Rollo3A --output-dir data/unseen_roll_repeat
.venv/bin/python code/check_unseen_roll.py --repeat-dir data/unseen_roll_repeat
```
