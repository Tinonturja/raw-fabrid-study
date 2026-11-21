#!/bin/bash
set -e
cd "$(dirname "$0")"
PY="../Project files/.venv/bin/python"
rm -f logs/p4_done.flag
caffeinate -dimsu -w $$ &
"$PY" code/labelfree_scan.py --model r18_native --rolls Rollo3A --limit 2 > logs/p4_smoke.log 2>&1
for m in r18_native wrn50; do
  "$PY" code/labelfree_scan.py --model $m > logs/p4_$m.log 2>&1
done
echo ALL_DONE > logs/p4_done.flag
