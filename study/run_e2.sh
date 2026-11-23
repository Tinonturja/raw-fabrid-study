#!/bin/sh
# E2 full run, then analysis. Started 22 Nov 2026.
cd "$(dirname "$0")"
PY="../Project files/.venv/bin/python"
RAW="<data_root>/data/RAW_FABRID.zip"
VISA="<datasets>/VisA"
"$PY" code/p5_e2_bank_layout.py --rawfabrid "$RAW" --visa "$VISA" >> logs/e2_full.log 2>&1
"$PY" code/p5_e2_bank_layout.py --analyze > logs/e2_analysis.log 2>&1
echo E2_DONE > logs/e2_done.flag
