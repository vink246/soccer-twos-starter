#!/usr/bin/env bash
set -euo pipefail

# Submit PPO training requesting one Tesla V100 GPU (no partition set — scheduler default).
#
# Usage:
#   bash scripts/submit_ppo_v100.sh [CONFIG_PATH] [RUN_TAG]
#
# Examples:
#   bash scripts/submit_ppo_v100.sh soccer_rl/algorithms/ppo/configs/config.yaml my_exp_v100
#   bash scripts/submit_ppo_v100.sh soccer_rl/algorithms/ppo/configs/ppo_baseline_multiagent_player_sparse.yaml sparse_baseline

CONFIG_PATH="${1:-soccer_rl/algorithms/ppo/configs/config.yaml}"
RUN_TAG="${2:-ppo_v100}"
CONDA_ENV="${CONDA_ENV:-soccertwos}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/slurm_logs}"

mkdir -p "${LOG_DIR}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${RUN_TAG}
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err

set -euo pipefail
source ~/.bashrc
conda activate ${CONDA_ENV}
export PACE_NUM_GPUS=1

cd /home/vineet/Documents/DRL/soccer-twos-starter
python soccer_rl/algorithms/ppo/training/train_ppo_team.py --config ${CONFIG_PATH} --output-dir ${OUTPUT_DIR}
EOF
