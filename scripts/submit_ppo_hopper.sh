#!/usr/bin/env bash
set -euo pipefail

# Submit PPO training on HOPPER with either h100 or h200 GPUs.
#
# Usage:
#   bash scripts/submit_ppo_hopper.sh h100 soccer_rl/algorithms/ppo/configs/config.yaml my_exp_h100
#   bash scripts/submit_ppo_hopper.sh h200 soccer_rl/algorithms/ppo/configs/config.yaml my_exp_h200

GPU_TYPE="${1:-h100}"          # h100 | h200
CONFIG_PATH="${2:-soccer_rl/algorithms/ppo/configs/config.yaml}"
RUN_TAG="${3:-ppo_hopper_${GPU_TYPE}}"
CONDA_ENV="${CONDA_ENV:-soccertwos}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/slurm_logs}"

if [[ "${GPU_TYPE}" != "h100" && "${GPU_TYPE}" != "h200" ]]; then
  echo "Invalid GPU_TYPE '${GPU_TYPE}'. Expected: h100 or h200."
  exit 1
fi

MEMORY_GB="64G"
if [[ "${GPU_TYPE}" == "h200" ]]; then
  MEMORY_GB="80G"
fi

mkdir -p "${LOG_DIR}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${RUN_TAG}
#SBATCH --gres=gpu:${GPU_TYPE}:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=${MEMORY_GB}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err

set -euo pipefail
source ~/.bashrc
conda activate ${CONDA_ENV}
export PACE_NUM_GPUS=1

cd /home/vineet/Documents/DRL/soccer-twos-starter
python soccer_rl/algorithms/ppo/training/train_ppo_team.py --config ${CONFIG_PATH} --output-dir ${OUTPUT_DIR}
EOF
