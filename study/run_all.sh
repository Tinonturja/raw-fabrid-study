#!/bin/bash
# Runs all audit experiments in sequence. Safe to re-run: finished rolls are skipped.
cd "$(dirname "$0")"
PY="../Project files/.venv/bin/python"
"$PY" -c "import torch;print('torch',torch.__version__,'mps',torch.backends.mps.is_available())" > logs/env.log 2>&1
caffeinate -dimsu -w $$ &
"$PY" code/strong_patchcore.py --model r18_native --rolls Rollo3A --limit 3 > logs/smoke.log 2>&1 || { echo FAILED >> logs/smoke.log; exit 1; }
for m in orig_r18 r18_native wrn50; do
  "$PY" code/strong_patchcore.py --model $m > logs/$m.log 2>&1 || { echo FAILED >> logs/$m.log; exit 1; }
done
echo ALL_DONE > logs/done.flag
