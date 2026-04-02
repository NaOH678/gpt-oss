#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: run_baseline_eval.sh <model_path> <prompt_parquet> <responses_out_parquet> <metrics_out_dir> <n_gpus_per_node> [hydra_overrides...]"
  exit 1
fi

MODEL_PATH="$1"
PROMPT_PARQUET="$2"
RESPONSES_OUT="$3"
METRICS_OUT_DIR="$4"
N_GPUS_PER_NODE="$5"
shift 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERL_DIR="${ROOT_DIR}/verl"
EVAL_SCRIPT="${ROOT_DIR}/zh_structured_verl/scripts/evaluate_structured_calls.py"

# Baseline defaults; can be overridden by environment variables when needed.
BASELINE_N_SAMPLES="${BASELINE_N_SAMPLES:-1}"
BASELINE_TEMPERATURE="${BASELINE_TEMPERATURE:-0.0}"
BASELINE_RESPONSE_LENGTH="${BASELINE_RESPONSE_LENGTH:-256}"
BASELINE_RESPONSE_INDEX="${BASELINE_RESPONSE_INDEX:-0}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"

mkdir -p "$(dirname "${RESPONSES_OUT}")"
mkdir -p "${METRICS_OUT_DIR}"

cd "${VERL_DIR}"

python3 -m verl.trainer.main_generation_server \
  data.train_files="${PROMPT_PARQUET}" \
  data.prompt_key=prompt \
  data.output_path="${RESPONSES_OUT}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.response_length="${BASELINE_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.n="${BASELINE_N_SAMPLES}" \
  actor_rollout_ref.rollout.temperature="${BASELINE_TEMPERATURE}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  "$@"

python3 "${EVAL_SCRIPT}" \
  --responses_parquet "${RESPONSES_OUT}" \
  --output_dir "${METRICS_OUT_DIR}" \
  --response_index "${BASELINE_RESPONSE_INDEX}" \
  --prompt_key prompt \
  --responses_key responses \
  --reward_model_key reward_model
