#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: run_offline_eval.sh <model_path> <prompt_parquet> <responses_out_parquet> <n_samples> <n_gpus_per_node> [hydra_overrides...]"
  exit 1
fi

MODEL_PATH="$1"
PROMPT_PARQUET="$2"
RESPONSES_OUT="$3"
N_SAMPLES="$4"
N_GPUS_PER_NODE="$5"
shift 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERL_DIR="${ROOT_DIR}/verl"
REWARD_PATH="${ROOT_DIR}/zh_structured_verl/rewards/structured_reward.py"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"

cd "${VERL_DIR}"

python3 -m verl.trainer.main_generation_server \
  data.train_files="${PROMPT_PARQUET}" \
  data.prompt_key=prompt \
  data.output_path="${RESPONSES_OUT}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.response_length=256 \
  actor_rollout_ref.rollout.n="${N_SAMPLES}" \
  actor_rollout_ref.rollout.temperature=0.0 \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  "$@"

python3 -m verl.trainer.main_eval \
  data.path="${RESPONSES_OUT}" \
  data.response_key=responses \
  data.data_source_key=data_source \
  data.reward_model_key=reward_model \
  custom_reward_function.path="${REWARD_PATH}" \
  custom_reward_function.name=compute_score
