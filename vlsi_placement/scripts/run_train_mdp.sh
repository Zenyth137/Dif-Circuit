#!/bin/bash
# Wrapper to redirect training output to log file
/mnt/fuso/wynshue/conda_envs/vlsi_placement/bin/python \
  "$(dirname "$0")/train_mdp.py" \
  --config "$(dirname "$0")/../configs/mdp_train.yaml" \
  --save-path "$(dirname "$0")/../checkpoints/mdp_policy_curiosity.pt" \
  > "$(dirname "$0")/../log/train_mdp.log" 2>&1
