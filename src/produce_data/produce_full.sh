export CUDA_VISIBLE_DEVICES=0,1,2,3

INPUT_DATA=/root/workspace/agent-rm/datasets/polaris/sft-1001/polaris-all-filterd.jsonl

OUTPUT_BASE_DIR="/root/workspace/agent-rm/datasets/polaris/sft-1001"

EXP_NAME="polaris-qwen2.5-7b-idx0-1000-exp3"

BACKEND_TYPE="vllm"

CONFIG_PATH="/root/workspace/agent-rm/Agent-Verifier/config/data_produce_local_model.yaml"

BATCH_SIZE=128

python /root/workspace/agent-rm/Agent-Verifier/src/produce_data/produce_plan.py \
    --input_path ${INPUT_DATA} \
    --output_path ${OUTPUT_BASE_DIR}/${EXP_NAME}-plan.jsonl \
    --backend_type $BACKEND_TYPE \
    --config_path $CONFIG_PATH \
    --batch_size $BATCH_SIZE \
    --start_idx 0 \


python /root/workspace/agent-rm/Agent-Verifier/src/produce_data/produce_subtask_report.py \
    --input_path ${OUTPUT_BASE_DIR}/${EXP_NAME}-plan.jsonl \
    --output_path ${OUTPUT_BASE_DIR}/${EXP_NAME}-subtask.jsonl \
    --backend_type $BACKEND_TYPE \
    --config_path $CONFIG_PATH \
    --batch_size $BATCH_SIZE \
    --start_idx 0 \

python /root/workspace/agent-rm/Agent-Verifier/src/produce_data/produce_integration.py \
    --input_path ${OUTPUT_BASE_DIR}/${EXP_NAME}-subtask.jsonl \
    --output_path ${OUTPUT_BASE_DIR}/${EXP_NAME}-integration.jsonl \
    --backend_type $BACKEND_TYPE \
    --config_path $CONFIG_PATH \
    --batch_size $BATCH_SIZE \
    --start_idx 0 \