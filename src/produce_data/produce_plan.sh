export CUDA_VISIBLE_DEVICES=0,1,2,3

python /root/workspace/agent-rm/Agent-Verifier/src/produce_data/produce_plan.py \
    --input_path "/root/workspace/agent-rm/datasets/polaris/raw-0928/polaris-1.jsonl" \
    --output_path "/root/workspace/agent-rm/datasets/test/data_produce_0927/test_plan.jsonl" \
    --backend_type "vllm" \
    --config_path "/root/workspace/agent-rm/Agent-Verifier/config/data_produce_local_model.yaml" \
    --batch_size 32 \
    --start_idx 0 \


# /root/workspace/agent-rm/Agent-Verifier/config/data_produce_local_model.yaml
# /root/workspace/agent-rm/Agent-Verifier/config/data_produce.yaml