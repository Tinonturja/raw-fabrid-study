#!/bin/bash
# Protocol 2 runs. Stops at the first failure; done flag only on full success.
set -e
cd "$(dirname "$0")"
PY="../Project files/.venv/bin/python"
rm -f logs/p2_done.flag
caffeinate -dimsu -w $$ &
"$PY" code/coverage_control.py --model r18_native --rolls Rollo3A --limit 2 > logs/p2_smoke.log 2>&1
for m in orig_r18 r18_native wrn50; do
  "$PY" code/coverage_control.py --model $m > logs/p2_$m.log 2>&1
done
echo ALL_DONE > logs/p2_done.flag
