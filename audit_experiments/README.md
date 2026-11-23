# Audit experiments (Protocols 1 to 5)

Independent re-analysis of the RAW-FABRID crop-placement study, plus the multi-dataset Protocol 5.

- `PROTOCOL*.md`: frozen protocols, each written before its results. `PROTOCOL_5.md` and `PROTOCOL_5_AMENDMENT_1.md` are frozen; their SHA-256 values are in the `.sha256` files.
- `RESULTS*.md`: results of Protocols 1 to 4 and the label-only cut audit.
- `code/`: experiment and analysis code. Detectors run in the project's Python environment (`../Project files/.venv` on the author's machine).
- `manifests/`: frozen Protocol 5 cohort manifests (RAW-FABRID, AITEX, VisA, MVTec AD 2 Fabric) with file hashes, and the TSFabrics run-output hashes.
- `P5_EXPLORATORY_*.md`: designs of the follow-up analyses E1 and E2, fingerprinted before scoring.
- `PROTOCOL_5_RUNLOG.md`: dated record of every run and correction after the freeze.
- `results*/`: analysis files. Per-image histograms and records are in the v1.0.0 release archive (see the top-level README). Memory banks (`*.pt`) are not tracked; they are rebuilt by the code.

Datasets are not included. See the protocols for sources and licences.
