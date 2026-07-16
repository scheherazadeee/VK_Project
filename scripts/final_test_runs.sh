#!/bin/bash
# Финальные прогоны на test. Стохастические модели: 3 сида, детерминированные: 1.
set -u

DET_CONFIGS="popularity knn_d64 ease_small"
STOCH_CONFIGS="ials bpr hybrid_d64 tt_id_only tt_emb_d64 tt_full ials_small"

for c in $DET_CONFIGS; do
  echo "=== $c (test) ==="
  python3 -m src.train --config configs/$c.yaml --eval-split test \
    2>&1 | grep -E "saved|Traceback|Error" || echo "FAILED: $c"
done

for c in $STOCH_CONFIGS; do
  for seed in 42 43 44; do
    echo "=== $c seed=$seed (test) ==="
    python3 -m src.train --config configs/$c.yaml --eval-split test --seed $seed \
      2>&1 | grep -E "saved|Traceback|Error" || echo "FAILED: $c seed=$seed"
  done
done
echo "FINAL TEST RUNS DONE"
