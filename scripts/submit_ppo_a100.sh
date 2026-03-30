#!/usr/bin/env bash
set -euo pipefail

# Submit PPO training on A100.
#
# Usage:
#   bash scripts/submit_ppo_a100.sh PPO/configs/config.yaml my_exp_a100

CONFIG_PATH="${1:-PPO/configs/config.yaml}"
RUN_TAG="${2:-ppo_a100}"
CONDA_ENV="${CONDA_ENV:-soccertwos}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/slurm_logs}"

mkdir -p "${LOG_DIR}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${RUN_TAG}
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
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
python PPO/training/train_ppo_team.py --config ${CONFIG_PATH} --output-dir ${OUTPUT_DIR}
EOF
