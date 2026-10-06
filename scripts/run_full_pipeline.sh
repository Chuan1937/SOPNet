#!/usr/bin/env bash
# Full paper pipeline: baselines -> ablation -> re-evaluation -> robustness -> bootstrap -> tables.
#
# Usage:
#   nohup bash scripts/run_full_pipeline.sh [BASELINE_BATCH] [MAIN_BATCH] [BASELINE_EPOCHS] [ABLATION_EPOCHS] \
#       > outputs/runs/pipeline2.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

BASELINE_BATCH=${1:-512}
MAIN_BATCH=${2:-1024}
BASELINE_EPOCHS=${3:-50}
ABLATION_EPOCHS=${4:-50}

log() { echo "[$(date '+%F %T')] $*"; }

run_step() {
  local name=$1
  shift
  log "=== START ${name} ==="
  if "$@"; then
    log "=== DONE ${name} ==="
  else
    local status=$?
    log "=== FAILED ${name} (exit ${status}) ==="
  fi
}

run_step baselines python scripts/train_baselines.py \
  --cache-dir outputs/cache_v1 \
  --baselines ross rpnet eqpolarity cfm diting_motion \
  --epochs "${BASELINE_EPOCHS}" --batch-size "${BASELINE_BATCH}" --seed 36

run_step ablation python scripts/run_ablation.py \
  --cache-dir outputs/cache_v1 \
  --experiments A_cls B_field C_jitter D_polarity \
  --epochs "${ABLATION_EPOCHS}" --batch-size "${MAIN_BATCH}" --seed 36

for seed in 36 2026 3407; do
  run_step "eval_seed_${seed}" python scripts/evaluate.py \
    --checkpoint "outputs/runs/sopnet_full_${seed}/best.pt" \
    --cache-dir outputs/cache_v1 --split test --auto-threshold --examples 12
done

for experiment in A_cls B_field C_jitter D_polarity; do
  run_step "eval_${experiment}" python scripts/evaluate.py \
    --checkpoint "outputs/ablation/${experiment}/best.pt" \
    --cache-dir outputs/cache_v1 --split test --auto-threshold
done

run_step robustness python scripts/run_robustness.py \
  --cache-dir outputs/cache_v1 --output-dir outputs/paper \
  --checkpoint sopnet=outputs/runs/sopnet_full_36/best.pt

run_step bootstrap python scripts/run_bootstrap.py \
  --checkpoint outputs/runs/sopnet_full_36/best.pt --cache-dir outputs/cache_v1

run_step tables python scripts/make_paper_tables.py \
  --runs-root outputs/runs outputs/ablation

log "=== PIPELINE FINISHED ==="
