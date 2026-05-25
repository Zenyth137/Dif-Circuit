#!/bin/bash
# ============================================================
# Full Pipeline v2: B*-tree Trajectories → Imitation → PPO → Evaluate + Viz
# ============================================================
set -e
PYTHON=/mnt/fuso/wynshue/conda_envs/vlsi_placement/bin/python
PROJECT=/mnt/fuso/DC
cd "$PROJECT"
LOG=vlsi_placement/log
mkdir -p "$LOG" vlsi_placement/results/visualizations

echo "=== PIPELINE START: $(date) ==="

# ── Step 1: Generate B*-tree expert trajectories ──
echo "[1/4] Generating B*-tree trajectories..."
$PYTHON vlsi_placement/scripts/generate_sa_trajectories.py \
    --num-trajectories 2000 \
    --output vlsi_placement/data/sa_trajectories \
    2>&1 | tee "$LOG/generate_trajectories.log"
echo "[1/4] DONE: $(date)"

# ── Step 2: Imitation Learning ──
echo "[2/4] Imitation Learning..."
$PYTHON vlsi_placement/scripts/train_imitation.py \
    --trajectory-dir vlsi_placement/data/sa_trajectories \
    --save-path vlsi_placement/checkpoints/mdp_policy_imitation.pt \
    --num-epochs 50 --batch-size 64 \
    --config vlsi_placement/configs/mdp_train.yaml \
    2>&1 | tee "$LOG/train_imitation.log"
echo "[2/4] DONE: $(date)"

# ── Step 3: PPO Fine-tuning ──
echo "[3/4] PPO Fine-tuning..."
$PYTHON vlsi_placement/scripts/train_mdp.py \
    --config vlsi_placement/configs/mdp_train.yaml \
    --save-path vlsi_placement/checkpoints/mdp_policy_finetuned.pt \
    --pretrained vlsi_placement/checkpoints/mdp_policy_imitation.pt \
    2>&1 | tee "$LOG/train_finetune.log"
echo "[3/4] DONE: $(date)"

# ── Step 4: Evaluate + Visualize ──
echo "[4/4] Full evaluation..."
$PYTHON vlsi_placement/scripts/evaluate.py \
    --mdp-checkpoint vlsi_placement/checkpoints/mdp_policy_finetuned.pt \
    --diffusion-checkpoint vlsi_placement/checkpoints/diffusion.pt \
    --data-dir vlsi_placement/data/test_netlists \
    --num-netlists 20 --prefix netlist \
    --output vlsi_placement/results/eval_final \
    --config vlsi_placement/configs/mdp_train.yaml \
    2>&1 | tee "$LOG/evaluate_final.log"

# Generate training curve visualization
$PYTHON -c "
import sys; sys.path.insert(0,'vlsi_placement')
from src.utils.visualize import plot_training_curves
plot_training_curves('$LOG/train_finetune.log',
    title='PPO Fine-tuning Progress',
    save_path='vlsi_placement/results/visualizations/training_curves.png')
print('Training curves saved')
"

echo "[4/4] DONE: $(date)"
echo "=== PIPELINE COMPLETE: $(date) ==="
