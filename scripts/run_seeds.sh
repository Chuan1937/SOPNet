#!/usr/bin/env bash
# Sequential 3-seed SOPNet training + unified-test evaluation.
# Usage: bash scripts/run_seeds.sh [BATCH_SIZE] [EPOCHS]
# Logs:  outputs/runs/pipeline.log
set -e
cd "$(dirname "$0")/.."

BATCH_SIZE="${1:-512}"
EPOCHS="${2:-50}"

for seed in 36 2026 3407; do
  echo "=== SOPNet seed ${seed} start $(date) ==="
  python scripts/train.py \
    --config configs/experiments/sopnet_full.yaml \
    --cache-dir outputs/cache_v1 \
    --output "outputs/runs/sopnet_full_${seed}" \
    --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --seed "${seed}"
  python scripts/evaluate.py \
    --checkpoint "outputs/runs/sopnet_full_${seed}/best.pt" \
    --cache-dir outputs/cache_v1 \
    --split test --auto-threshold --examples 12 \
    --output "outputs/runs/sopnet_full_${seed}"
  echo "=== SOPNet seed ${seed} done $(date) ==="
done
