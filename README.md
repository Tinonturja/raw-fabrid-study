# Crop placement in fabric anomaly detection

This research repository compares what a detector sees when defect annotations guide its test windows and when it scans the complete fabric frame. The Markdown manuscript, source code, figures and scored results are included. Large image archives, pretrained weights, feature caches and the local Python environment are kept on the research machine.

**Start with the [draft manuscript](manuscript/manuscript.md).** On RAW-FABRID v1, guided crop placement improved a globally pooled detector's frame ranking. A spatial PatchCore variant favored a blind fixed grid on all three matched splits. The ranking for pixel localization changed when the evaluation included marked fabric outside the released windows. The [publication assessment](reports/publication_scope.md) documents the literature search and the exploratory work preceding the spatial experiments.

The [illustrated research guide and code tutorial](output/pdf/fabric_inspection_research_guide.pdf) explains the fabrics, the two datasets and how to interpret the results. Four worked examples take you from checking the published scores to tracing a fabric tile through the detector.

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
