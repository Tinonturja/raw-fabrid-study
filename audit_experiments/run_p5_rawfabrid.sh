#!/bin/bash
# Protocol 5, RAW-FABRID: waits for the AITEX/VisA run to finish, then smoke test, full run and analysis.
set -e
cd "$(dirname "$0")"
PY="../Project files/.venv/bin/python"
ZIP="<data_root>/data/RAW_FABRID.zip"
caffeinate -dimsu -w $$ &
while [ ! -f logs/p5_done.flag ]; do
  if ! pgrep -f "run_p5.sh" > /dev/null; then echo "main run stopped without finishing" >> logs/p5_rawfabrid_full.log; exit 1; fi
  sleep 60
done
"$PY" code/p5_run.py --dataset rawfabrid --data "$ZIP" --categories Rollo3A --limit 2 --families wrn dino > logs/p5_rawfabrid_smoke.log 2>&1
"$PY" code/p5_run.py --dataset rawfabrid --data "$ZIP" --families wrn dino >> logs/p5_rawfabrid_full.log 2>&1
"$PY" code/p5_analyze.py --dataset rawfabrid > logs/p5_rawfabrid_analysis.log 2>&1
echo DONE > logs/p5_rawfabrid_done.flag
