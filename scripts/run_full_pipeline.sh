#!/usr/bin/env bash
# Full paper pipeline. STRICT mode: the first failed step aborts the pipeline,
# so a partially failed run can never produce a seemingly complete report.
#
# Usage:
#   nohup setsid bash scripts/run_full_pipeline.sh > outputs/runs/pipeline_full.log 2>&1 &
set -u
set -e
set -o pipefail
cd "$(dirname "$0")/.."

log() { echo "[$(date '+%F %T')] $*"; }

run_step() {
  local name=$1
  shift
  log "=== START ${name} ==="
  if "$@"; then
    log "=== DONE ${name} ==="
  else
    local status=$?
    log "=== FAILED ${name} (exit ${status}) - aborting pipeline ==="
    return "$status"
  fi
}

run_step baselines python scripts/train_baselines.py \
  --cache-dir outputs/cache_v1 \
  --baselines ross rpnet eqpolarity cfm diting_motion \
  --epochs 50 --batch-size 512 --window-length 600 --seed 36

run_step ablation python scripts/run_ablation.py \
  --cache-dir outputs/cache_v1 \
  --experiments A_cls_ud B_field C_polarity --epochs 50 --batch-size 1024

run_step eval_sopnet python scripts/evaluate.py \
  --checkpoint outputs/runs/sopnet_nojitter_36/best.pt \
  --cache-dir outputs/cache_v1 --split test --auto-threshold --examples 12

for experiment in A_cls_ud B_field C_polarity; do
  run_step "eval_${experiment}" python scripts/evaluate.py \
    --checkpoint "outputs/ablation/${experiment}/best.pt" \
    --cache-dir outputs/cache_v1 --split test
done

run_step robustness python scripts/run_robustness.py \
  --cache-dir outputs/cache_v1 --output-dir outputs/paper \
  --subset-size 30000 --subset-seed 20261004 \
  --models sopnet=outputs/runs/sopnet_nojitter_36/best.pt \
           ross=outputs/runs/baseline_ross/best.pt \
           rpnet=outputs/runs/baseline_rpnet/best.pt \
           eqpolarity=outputs/runs/baseline_eqpolarity/best.pt \
           cfm=outputs/runs/baseline_cfm/best.pt \
           diting_motion=outputs/runs/baseline_diting_motion/best.pt

run_step bootstrap python scripts/run_bootstrap.py \
  --checkpoint outputs/runs/sopnet_nojitter_36/best.pt --cache-dir outputs/cache_v1

run_step tables python scripts/make_paper_tables.py --runs-root outputs/runs outputs/ablation

log "=== PIPELINE FINISHED ==="
