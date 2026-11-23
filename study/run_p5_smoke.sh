#!/bin/bash
# Set the dataset paths below (<data_root>, <datasets>) and PY to your environment before running.
# Protocol 5 smoke test: 3+3 AITEX strips, 2+2 VisA candle images, smoke analysis. Not used in results.
set -e
cd "$(dirname "$0")"
PY="../Project files/.venv/bin/python"
AITEX="<data_root>/data/aitex"
VISA="<datasets>/VisA"
rm -f logs/p5_smoke_done.flag
caffeinate -dimsu -w $$ &
"$PY" code/p5_run.py --dataset aitex --data "$AITEX" --limit 3 > logs/p5_aitex_smoke.log 2>&1
"$PY" code/p5_analyze.py --dataset aitex --stage smoke > logs/p5_aitex_smoke_analysis.log 2>&1
"$PY" code/p5_run.py --dataset visa --data "$VISA" --categories candle --limit 2 > logs/p5_visa_smoke.log 2>&1
"$PY" code/p5_analyze.py --dataset visa --stage smoke > logs/p5_visa_smoke_analysis.log 2>&1
echo SMOKE_DONE > logs/p5_smoke_done.flag
