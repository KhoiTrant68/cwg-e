set -euo pipefail

# ============================================================================
# WARNING — NOT runnable on Kaggle out of the box.
#
# This launches full ImageNet-256 training (~200 GB SD-VAE latent cache +
# pretrained MAE/VAE checkpoints). It REQUIRES, on the local filesystem:
#   - IMAGENET_CACHE_PATH/train/train_moments.npy      (SD-VAE latent cache)
#   - HF_ROOT/{mae_latent_256, ...}                    (feature extractors)
#   - VAE_HF_PATH                                      (SD-VAE decoder)
# all configured in `utils/env.py` (or the WFLOW_HF_ROOT env override).
#
# See the upstream W-Flow README sections "Path Configuration" and
# "Dataset and Latent Cache" for how to prepare this. None of those caches
# are available on a fresh Kaggle GPU instance — for Kaggle, run
# `experiments/run_all.py` instead (2D theory + Gate 2).
# ============================================================================

NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NGPU=${NGPU:-1}

MASTER_PORT=${MASTER_PORT:-6667}

CONFIG=configs/gen/cwge_ablation_1node.yaml
EXP_NAME=cwge_ablation_1node

WORKDIR=/path/to/workdir/$EXP_NAME
WANDB_PROJECT=YOUR_WANDB_PROJECT
WANDB_NAME=$EXP_NAME

DRIFT_COMPILE=1 \
NCCL_DEBUG=WARN \
WANDB_PROJECT=$WANDB_PROJECT \
WANDB_NAME=$WANDB_NAME \
torchrun \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    --workdir "$WORKDIR"


echo "finished!"
