#!/bin/bash
# Protocol 5 overnight run on the Mac: tests, AITEX, VisA, then analysis. Stops at the first error.
set -e
cd "$(dirname "$0")"
PY="../Project files/.venv/bin/python"
AITEX="<data_root>/data/aitex"
VISA="<datasets>/VisA"
rm -f logs/p5_done.flag
caffeinate -dimsu -w $$ &
"$PY" code/tests_p5.py --aitex "$AITEX" --torch > logs/p5_tests_full.log 2>&1
"$PY" code/p5_run.py --dataset aitex --data "$AITEX" > logs/p5_aitex_full.log 2>&1
"$PY" code/p5_analyze.py --dataset aitex > logs/p5_aitex_analysis.log 2>&1
"$PY" code/p5_run.py --dataset visa --data "$VISA" > logs/p5_visa_full.log 2>&1
"$PY" code/p5_analyze.py --dataset visa > logs/p5_visa_analysis.log 2>&1
echo ALL_DONE > logs/p5_done.flag
