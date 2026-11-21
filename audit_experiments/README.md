# Audit experiments (Protocols 1 to 5)

Independent re-analysis of the RAW-FABRID crop-placement study, plus the multi-dataset Protocol 5.

- `PROTOCOL*.md`: frozen protocols, each written before its results. `PROTOCOL_5.md` and `PROTOCOL_5_AMENDMENT_1.md` are frozen; their SHA-256 values are in the `.sha256` files.
- `RESULTS*.md`: results of Protocols 1 to 4 and the label-only cut audit.
- `code/`: experiment and analysis code. Detectors run in the project's Python environment (`../Project files/.venv` on the author's machine).
- `manifests/`: frozen Protocol 5 cohort manifests (AITEX, VisA) with file hashes.
- `results*/`: per-fold JSON outputs. Memory banks (`*.pt`) are not tracked; they are rebuilt by the code.

Datasets are not included. See the protocols for sources and licences.
