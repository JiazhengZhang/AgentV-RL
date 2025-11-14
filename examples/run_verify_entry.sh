#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4,5,6,7"}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

NUM_WORKERS=4

TASK_NAME="qwen2.5-7b-math-bon128-gaokao-2023-eval"
EXP_NAME="qwen2.5-7b-eval-1"

CONFIG="/root/workspace/agent-rm/Agent-Verifier/config/distrubute_verify.yaml"
MODEL_PATH="/root/workspace/agent-rm/models/Qwen-2.5-7B-Instruct"

INPUT="/root/workspace/agent-rm/datasets/gaokao2023/bon/${TASK_NAME}.jsonl"
OUTPUT_DIR="/root/workspace/agent-rm/datasets/gaokao2023/1021"
LOG_DIR="/root/workspace/agent-rm/log"

START_IDX=0
APPEND=1      

ENABLE_THINKING=0   

EXTRA_PY_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [options] [-- extra_python_args...]

Options:
  --task-name NAME            task name (default: ${TASK_NAME})
  --exp-name NAME             exp name (default: ${EXP_NAME})
  --num-workers N             number of workers (default: ${NUM_WORKERS})
  --config PATH               distrubute_verify.yaml 路径 (default: ${CONFIG})
  --model-path PATH           hf model path (default: ${MODEL_PATH})
  --input PATH                input jsonl (default: ${INPUT})
  --output-dir DIR            output directory (default: ${OUTPUT_DIR})
  --log-dir DIR               log directory (default: ${LOG_DIR})
  --start-idx N               start_idx (default: ${START_IDX})
  --no-append                 disable append mode for output
  --enable-thinking           use enable_thinking for qwen3
  -h, --help                  show help

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-name)
      TASK_NAME="$2"; shift 2;;
    --exp-name)
      EXP_NAME="$2"; shift 2;;
    --num-workers)
      NUM_WORKERS="$2"; shift 2;;
    --config)
      CONFIG="$2"; shift 2;;
    --model-path)
      MODEL_PATH="$2"; shift 2;;
    --input)
      INPUT="$2"; shift 2;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    --log-dir)
      LOG_DIR="$2"; shift 2;;
    --start-idx)
      START_IDX="$2"; shift 2;;
    --no-append)
      APPEND=0; shift 1;;
    --enable-thinking)
      ENABLE_THINKING=1; shift 1;;
    -h|--help)
      usage; exit 0;;
    --)
      shift
      EXTRA_PY_ARGS=("$@")
      break;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1;;
  esac
done

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

OUTPUT="${OUTPUT_DIR}/${TASK_NAME}-${EXP_NAME}.jsonl"
LOG_FILE="${LOG_DIR}/${TASK_NAME}_${EXP_NAME}.log"

APPEND_FLAG=()
if [[ "${APPEND}" -eq 1 ]]; then
  APPEND_FLAG+=(--append)
fi

ENABLE_THINKING_ARG=()
if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
  ENABLE_THINKING_ARG+=(--enable-thinking)
fi

echo "===> Running verify:"
echo "  TASK_NAME         = ${TASK_NAME}"
echo "  EXP_NAME          = ${EXP_NAME}"
echo "  NUM_WORKERS       = ${NUM_WORKERS}"
echo "  INPUT             = ${INPUT}"
echo "  OUTPUT            = ${OUTPUT}"
echo "  MODEL_PATH        = ${MODEL_PATH}"
echo "  CONFIG            = ${CONFIG}"
echo "  START_IDX         = ${START_IDX}"
echo "  APPEND            = ${APPEND}"
echo "  ENABLE_THINKING   = ${ENABLE_THINKING}  # 1=on, 0=off"
echo "  LOG_FILE          = ${LOG_FILE}"
echo "  EXTRA_PY_ARGS     = ${EXTRA_PY_ARGS[@]:-<none>}"
echo

python /root/workspace/agent-rm/Agent-Verifier/src/run_verify_multihead.py \
  --config "${CONFIG}" \
  --input  "${INPUT}" \
  --output "${OUTPUT}" \
  --model_path "${MODEL_PATH}" \
  --record-batch-size 1 \
  --include_full_meta \
  --start_idx "${START_IDX}" \
  --num-workers "${NUM_WORKERS}" \
  --max-inflight-batches "${NUM_WORKERS}" \
  "${APPEND_FLAG[@]}" \
  "${ENABLE_THINKING_ARG[@]}" \
  "${EXTRA_PY_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
