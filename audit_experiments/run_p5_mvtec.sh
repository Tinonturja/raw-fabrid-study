#!/bin/bash
# Protocol 5 fallback dataset, MVTec AD 2 Fabric: runs after RAW-FABRID finishes. Stops at the first error.
set -e
cd "$(dirname "$0")"
PY="../Project files/.venv/bin/python"
DATA="<datasets>/fabric/fabric"
caffeinate -dimsu -w $$ &
while [ ! -f logs/p5_rawfabrid_done.flag ]; do
  if ! pgrep -f "run_p5_rawfabrid" > /dev/null; then echo "RAW-FABRID run stopped without finishing" >> logs/p5_mvtec_full.log; exit 1; fi
  sleep 60
done
"$PY" code/p5_run.py --dataset mvtec_fabric --data "$DATA" --limit 2 --families wrn dino > logs/p5_mvtec_smoke.log 2>&1
"$PY" code/p5_run.py --dataset mvtec_fabric --data "$DATA" --families wrn dino >> logs/p5_mvtec_full.log 2>&1
"$PY" code/p5_analyze.py --dataset mvtec_fabric > logs/p5_mvtec_analysis.log 2>&1
echo DONE > logs/p5_mvtec_done.flag
