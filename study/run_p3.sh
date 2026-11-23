#!/bin/bash
set -e
cd "$(dirname "$0")"
PY="../Project files/.venv/bin/python"
rm -f logs/p3_done.flag
caffeinate -dimsu -w $$ &
"$PY" code/coverage_control_c4.py --model r18_native --rolls Rollo3A --limit 2 > logs/p3_smoke.log 2>&1
for m in r18_native wrn50; do
  "$PY" code/coverage_control_c4.py --model $m > logs/p3_$m.log 2>&1
done
echo ALL_DONE > logs/p3_done.flag
