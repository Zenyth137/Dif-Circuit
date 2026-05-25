#!/bin/bash
# ============================================================
# Full Pipeline: SA Trajectories → Imitation → PPO Fine-tune
# ============================================================
set -e

PYTHON=/mnt/fuso/wynshue/conda_envs/vlsi_placement/bin/python
PROJECT=/mnt/fuso/DC
cd "$PROJECT"

LOG_DIR=vlsi_placement/log
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "PIPELINE START: $(date)"
echo "============================================================"

# ── Step 1: Generate SA expert trajectories ──
echo ""
echo "[1/3] Generating SA trajectories..."
$PYTHON vlsi_placement/scripts/generate_sa_trajectories.py \
    --num-trajectories 2000 \
    --output vlsi_placement/data/sa_trajectories \
    2>&1 | tee "$LOG_DIR/generate_sa.log"
echo "[1/3] DONE: $(date)"

# ── Step 2: Imitation Learning ──
echo ""
echo "[2/3] Imitation Learning (Behavior Cloning)..."
$PYTHON vlsi_placement/scripts/train_imitation.py \
    --trajectory-dir vlsi_placement/data/sa_trajectories \
    --save-path vlsi_placement/checkpoints/mdp_policy_imitation.pt \
    --num-epochs 50 \
    --batch-size 64 \
    --config vlsi_placement/configs/mdp_train.yaml \
    2>&1 | tee "$LOG_DIR/train_imitation.log"
echo "[2/3] DONE: $(date)"

# ── Step 3: PPO Fine-tuning ──
echo ""
echo "[3/3] PPO Fine-tuning..."
$PYTHON vlsi_placement/scripts/train_mdp.py \
    --config vlsi_placement/configs/mdp_train.yaml \
    --save-path vlsi_placement/checkpoints/mdp_policy_finetuned.pt \
    --pretrained vlsi_placement/checkpoints/mdp_policy_imitation.pt \
    2>&1 | tee "$LOG_DIR/train_finetune.log"
echo "[3/3] DONE: $(date)"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE: $(date)"
echo "============================================================"
