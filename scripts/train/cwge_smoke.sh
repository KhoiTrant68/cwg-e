#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# CWG-E smoke launcher — imagenet-mini on Kaggle T4/T4×2 or a single dev GPU.
#
# Prereqs (see docs/proposal.md §Kaggle smoke and README §Path Configuration):
#   - `utils/env.py` patched so IMAGENET_CACHE_PATH points to a small latent
#     cache built via `python -m dataset.latent` on an imagenet-mini folder.
#   - MAE + SD-VAE downloaded via `python misc/download_pretrained.py`.
#
# What this verifies:
#   1. Data loader reads latent cache correctly.
#   2. Sticky refresh in cluster_stats.py fires at step
#      `sticky_warmup_steps` (log line prefixed "CWG-E sticky refresh").
#   3. Per-cluster Sinkhorn + outer Γ path in drift_loss_ot runs without
#      dimension errors.
#   4. DDP path (NGPU>=2) has bit-identical centroids across ranks.
#
# NOT tuned for quality — FID/loss values here are meaningless.
# ============================================================================

NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NGPU=${NGPU:-1}

MASTER_PORT=${MASTER_PORT:-6667}

CONFIG=${CONFIG:-configs/gen/cwge_smoke.yaml}
EXP_NAME=${EXP_NAME:-cwge_smoke}
WORKDIR=${WORKDIR:-./workdir/${EXP_NAME}}

# wandb OFF by default (config has use_wandb: false)
# DRIFT_COMPILE=0 — Tesla T4 lacks native bf16 compile; also cuts ~60s startup.
NCCL_DEBUG=${NCCL_DEBUG:-WARN} \
DRIFT_COMPILE=${DRIFT_COMPILE:-0} \
torchrun \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    --workdir "$WORKDIR"

echo "finished!"
