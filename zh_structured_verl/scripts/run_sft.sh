#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: run_sft.sh <nproc_per_node> <model_path> <train_parquet> <val_parquet> <save_dir> [hydra_overrides...]"
  exit 1
fi

NPROC_PER_NODE="$1"
MODEL_PATH="$2"
TRAIN_PARQUET="$3"
VAL_PARQUET="$4"
SAVE_DIR="$5"
shift 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERL_DIR="${ROOT_DIR}/verl"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"

cd "${VERL_DIR}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.messages_key=messages \
  data.micro_batch_size_per_gpu=1 \
  data.max_token_len_per_gpu=4096 \
  data.pad_mode=no_padding \
  data.truncation=error \
  model.path="${MODEL_PATH}" \
  model.use_remove_padding=true \
  model.enable_gradient_checkpointing=true \
  engine=fsdp \
  optim.lr=5e-6 \
  trainer.default_local_dir="${SAVE_DIR}" \
  trainer.project_name=zh_structured \
  trainer.experiment_name=sft \
  trainer.logger='["console"]' \
  trainer.total_epochs=1 \
  trainer.save_freq=200 \
  trainer.test_freq=50 \
  trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
  trainer.nnodes=1 \
  "$@"
