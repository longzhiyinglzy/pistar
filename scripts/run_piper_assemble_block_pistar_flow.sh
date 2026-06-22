#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "${ROOT}/venv/bin/python" ]]; then
  DEFAULT_PYTHON="${ROOT}/venv/bin/python"
else
  DEFAULT_PYTHON="python"
fi

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

SRC_DATASET="${SRC_DATASET:-/home/user/.cache/huggingface/lerobot/piper/assemble_block1_v21}"
DATASET_PARENT="${DATASET_PARENT:-/home/user/.cache/huggingface/lerobot/piper}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-assemble_block1_pistar_10hz_3view}"
TRAIN_REPO_ID="${TRAIN_REPO_ID:-piper/${TRAIN_DATASET_NAME}}"
TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-${DATASET_PARENT}/${TRAIN_DATASET_NAME}}"
ROLLOUT_DATASET_NAME="${ROLLOUT_DATASET_NAME:-assemble_block1_pistar_rollout_r1}"
ROLLOUT_REPO_ID="${ROLLOUT_REPO_ID:-${ROLLOUT_DATASET_NAME}}"
ROLLOUT_DATASET_DIR="${ROLLOUT_DATASET_DIR:-${DATASET_PARENT}/${ROLLOUT_DATASET_NAME}}"
MERGED_DATASET_NAME="${MERGED_DATASET_NAME:-assemble_block1_pistar_10hz_3view_r1}"
MERGED_REPO_ID="${MERGED_REPO_ID:-piper/${MERGED_DATASET_NAME}}"
MERGED_DATASET_DIR="${MERGED_DATASET_DIR:-${DATASET_PARENT}/${MERGED_DATASET_NAME}}"

CONFIG_NAME="${CONFIG_NAME:-pi05_star_assemble_blocks}"
INFER_CONFIG_NAME="${INFER_CONFIG_NAME:-pi05_star_assemble_blocks_infer}"
EXP_NAME="${EXP_NAME:-assemble_block1_3view_stage0}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT}/checkpoints}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${ROOT}/assets}"
PI05_BASE_PARAMS="${PI05_BASE_PARAMS:-gs://openpi-assets/checkpoints/pi05_base/params}"
POLICY_NUM_STEPS="${POLICY_NUM_STEPS:-30000}"
POLICY_SAVE_INTERVAL="${POLICY_SAVE_INTERVAL:-1000}"
POLICY_KEEP_PERIOD="${POLICY_KEEP_PERIOD:-1000}"
POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-32}"
POLICY_NUM_WORKERS="${POLICY_NUM_WORKERS:-8}"
POLICY_STEP="${POLICY_STEP:-29999}"
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}/${POLICY_STEP}}"
POLICY_ADV_IND="${POLICY_ADV_IND:-positive}"

VALUE_CHECKPOINT_DIR="${VALUE_CHECKPOINT_DIR:-${CHECKPOINT_BASE_DIR}/value/assemble_block1_3view}"
VALUE_NUM_STEPS="${VALUE_NUM_STEPS:-20000}"
VALUE_SAVE_INTERVAL="${VALUE_SAVE_INTERVAL:-1000}"
VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-16}"
VALUE_NUM_WORKERS="${VALUE_NUM_WORKERS:-0}"
VALUE_FREEZE_MODE="${VALUE_FREEZE_MODE:-all_backbones}"
VALUE_LOAD_PRETRAINED="${VALUE_LOAD_PRETRAINED:-1}"
GEMMA_TOKENIZER_PATH="${GEMMA_TOKENIZER_PATH:-/home/user/hf_models/google/gemma-3-270m/tokenizer.model}"
SIGLIP_CHECKPOINT_PATH="${SIGLIP_CHECKPOINT_PATH:-${OPENPI_VALUE_SIGLIP_PATH:-}}"
GEMMA_CHECKPOINT_DIR="${GEMMA_CHECKPOINT_DIR:-${OPENPI_VALUE_GEMMA_CKPT_DIR:-}}"

TASK_NAME="${TASK_NAME:-Pick up the block1 and assemble it.}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-50}"
ROLLOUT_FPS="${ROLLOUT_FPS:-10}"
ROLLOUT_MAX_STEP="${ROLLOUT_MAX_STEP:-300}"
ARM_CAN="${ARM_CAN:-can0}"

OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"

main_py_path() {
  export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
}

robot_py_path() {
  export PYTHONPATH="${ROOT}/control_your_robot:${ROOT}/control_your_robot/src:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
}

usage() {
  cat <<EOF
Usage:
  $(basename "$0") <command>

Commands:
  show              Print the resolved paths and config.
  convert           Convert LeRobot v2.1 AV1 video data to flat PiStar 10 Hz, 3-view data.
  stats             Compute policy normalization stats for TRAIN_REPO_ID.
  train-policy      Train PiStar/pi0.5 policy on TRAIN_REPO_ID.
  train-value       Train the 3-view value model on TRAIN_DATASET_DIR.
  label-advantage   Run value inference and write adv_ind back into TRAIN_DATASET_DIR.
  rollout           Roll out policy on the real PiPER arm with SpaceMouse intervention.
  merge-rollout     Merge TRAIN_DATASET_DIR and ROLLOUT_DATASET_DIR into MERGED_DATASET_DIR.

Common overrides:
  OVERWRITE=1                         allow overwriting generated datasets/checkpoints
  EXP_NAME=assemble_block1_3view_r1    policy experiment name
  TRAIN_DATASET_NAME=assemble_block1_pistar_10hz_3view_r1
  TRAIN_REPO_ID=piper/assemble_block1_pistar_10hz_3view_r1
  POLICY_MODEL_PATH=/path/to/checkpoint_step
  PI05_BASE_PARAMS=/path/or/gs/to/pi05_base/params
  VALUE_LOAD_PRETRAINED=1
  SIGLIP_CHECKPOINT_PATH=/path/to/siglip2_so400m14_224.npz
  GEMMA_CHECKPOINT_DIR=/path/to/gemma-3-270m-orbax
  GEMMA_TOKENIZER_PATH=/path/to/tokenizer.model

Recommended first pass:
  $0 convert
  $0 stats
  $0 train-policy
  $0 rollout
  $0 merge-rollout
  TRAIN_DATASET_NAME=assemble_block1_pistar_10hz_3view_r1 \\
  TRAIN_REPO_ID=piper/assemble_block1_pistar_10hz_3view_r1 \\
  EXP_NAME=assemble_block1_3view_r1 $0 stats
  TRAIN_DATASET_NAME=assemble_block1_pistar_10hz_3view_r1 \\
  TRAIN_REPO_ID=piper/assemble_block1_pistar_10hz_3view_r1 \\
  EXP_NAME=assemble_block1_3view_r1 $0 train-value
  TRAIN_DATASET_NAME=assemble_block1_pistar_10hz_3view_r1 \\
  TRAIN_REPO_ID=piper/assemble_block1_pistar_10hz_3view_r1 \\
  EXP_NAME=assemble_block1_3view_r1 $0 label-advantage
  TRAIN_DATASET_NAME=assemble_block1_pistar_10hz_3view_r1 \\
  TRAIN_REPO_ID=piper/assemble_block1_pistar_10hz_3view_r1 \\
  EXP_NAME=assemble_block1_3view_r1 $0 train-policy
EOF
}

show_config() {
  cat <<EOF
ROOT=${ROOT}
PYTHON_BIN=${PYTHON_BIN}
SRC_DATASET=${SRC_DATASET}
TRAIN_DATASET_DIR=${TRAIN_DATASET_DIR}
TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME}
TRAIN_REPO_ID=${TRAIN_REPO_ID}
ROLLOUT_DATASET_DIR=${ROLLOUT_DATASET_DIR}
MERGED_DATASET_DIR=${MERGED_DATASET_DIR}
MERGED_REPO_ID=${MERGED_REPO_ID}
CONFIG_NAME=${CONFIG_NAME}
INFER_CONFIG_NAME=${INFER_CONFIG_NAME}
EXP_NAME=${EXP_NAME}
POLICY_MODEL_PATH=${POLICY_MODEL_PATH}
VALUE_CHECKPOINT_DIR=${VALUE_CHECKPOINT_DIR}
TASK_NAME=${TASK_NAME}
OVERWRITE=${OVERWRITE}
RESUME=${RESUME}
EOF
}

convert_dataset() {
  main_py_path
  args=(
    "${PYTHON_BIN}" "${ROOT}/scripts/convert_lerobot_v21_to_pistar_flat.py"
    --source "${SRC_DATASET}"
    --output "${TRAIN_DATASET_DIR}"
    --target-fps 10
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  "${args[@]}"
}

compute_stats() {
  main_py_path
  "${PYTHON_BIN}" "${ROOT}/scripts/compute_norm_stats.py" \
    --config-name "${CONFIG_NAME}" \
    --repo-id "${TRAIN_REPO_ID}"
}

train_policy() {
  if [[ "${OVERWRITE}" == "1" && "${RESUME}" == "1" ]]; then
    echo "OVERWRITE and RESUME cannot both be 1" >&2
    exit 2
  fi
  main_py_path
  args=(
    "${PYTHON_BIN}" "${ROOT}/scripts/train.py" "${CONFIG_NAME}"
    --exp-name "${EXP_NAME}"
    --data.repo-id "${TRAIN_REPO_ID}"
    --assets-base-dir "${ASSETS_BASE_DIR}"
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
    --weight-loader.params-path "${PI05_BASE_PARAMS}"
    --num-train-steps "${POLICY_NUM_STEPS}"
    --save-interval "${POLICY_SAVE_INTERVAL}"
    --keep-period "${POLICY_KEEP_PERIOD}"
    --batch-size "${POLICY_BATCH_SIZE}"
    --num-workers "${POLICY_NUM_WORKERS}"
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  if [[ "${RESUME}" == "1" ]]; then
    args+=(--resume)
  fi
  if [[ "${WANDB_ENABLED}" != "1" ]]; then
    args+=(--no-wandb-enabled)
  fi
  "${args[@]}"
}

train_value() {
  main_py_path
  args=(
    "${PYTHON_BIN}" "${ROOT}/scripts/train_value.py"
    --data_dir "${TRAIN_DATASET_DIR}"
    --checkpoint_dir "${VALUE_CHECKPOINT_DIR}"
    --batch_size "${VALUE_BATCH_SIZE}"
    --num_train_steps "${VALUE_NUM_STEPS}"
    --save_interval "${VALUE_SAVE_INTERVAL}"
    --num_workers "${VALUE_NUM_WORKERS}"
    --wandb_mode disabled
    --tokenizer_path "${GEMMA_TOKENIZER_PATH}"
    --freeze_mode "${VALUE_FREEZE_MODE}"
  )
  if [[ "${VALUE_LOAD_PRETRAINED}" == "1" ]]; then
    args+=(--load_pretrained)
  fi
  if [[ -n "${SIGLIP_CHECKPOINT_PATH}" ]]; then
    args+=(--siglip_checkpoint_path "${SIGLIP_CHECKPOINT_PATH}")
  fi
  if [[ -n "${GEMMA_CHECKPOINT_DIR}" ]]; then
    args+=(--gemma_checkpoint_dir "${GEMMA_CHECKPOINT_DIR}")
  fi
  "${args[@]}"
}

label_advantage() {
  main_py_path
  "${PYTHON_BIN}" "${ROOT}/scripts/label_advantage_from_vlm.py" \
    --data_dir "${TRAIN_DATASET_DIR}" \
    --checkpoint_dir "${VALUE_CHECKPOINT_DIR}" \
    --lookahead 50 \
    --top_percent 30 \
    --batch_size 32 \
    --num_workers 0 \
    --tokenizer_path "${GEMMA_TOKENIZER_PATH}" \
    --right_wrist_image_col side_image
}

rollout() {
  robot_py_path
  "${PYTHON_BIN}" "${ROOT}/control_your_robot/example/deploy/piper_spacemouse_dagger_on_PI0.py" \
    --model-path "${POLICY_MODEL_PATH}" \
    --train-config "${INFER_CONFIG_NAME}" \
    --task-name "${TASK_NAME}" \
    --adv-ind "${POLICY_ADV_IND}" \
    --repo-id "${ROLLOUT_REPO_ID}" \
    --output-dir "${DATASET_PARENT}" \
    --fps "${ROLLOUT_FPS}" \
    --num-episode "${ROLLOUT_EPISODES}" \
    --max-step "${ROLLOUT_MAX_STEP}" \
    --arm-can "${ARM_CAN}" \
    --save-adv-ind none
}

merge_rollout() {
  main_py_path
  args=(
    "${PYTHON_BIN}" "${ROOT}/scripts/merge_datasets.py"
    --sources "${TRAIN_DATASET_DIR}" "${ROLLOUT_DATASET_DIR}"
    --output "${MERGED_DATASET_DIR}"
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  "${args[@]}"
  echo "Merged repo id for the next round: ${MERGED_REPO_ID}"
}

cmd="${1:-}"
case "${cmd}" in
  show) show_config ;;
  convert) convert_dataset ;;
  stats) compute_stats ;;
  train-policy) train_policy ;;
  train-value) train_value ;;
  label-advantage) label_advantage ;;
  rollout) rollout ;;
  merge-rollout) merge_rollout ;;
  ""|-h|--help|help) usage ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
