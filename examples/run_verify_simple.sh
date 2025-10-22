export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOKENIZERS_PARALLELISM=false

NUM_WORKERS=4

TASK_NAME=qwen2.5-7b-math-bon128-gaokao-2023-eval

EXP_NAME=qwen2.5-7b-eval-simple-1

# ray stop

python /root/workspace/agent-rm/Agent-Verifier/src/run_verify_simple.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/distrubute_verify_simple.yaml \
  --input  /root/workspace/agent-rm/datasets/gaokao2023/bon/qwen2.5-7b-math-bon128-gaokao-2023.jsonl \
  --output /root/workspace/agent-rm/datasets/gaokao2023/1021/${TASK_NAME}-${EXP_NAME}.jsonl \
  --model_path /root/workspace/agent-rm/models/Qwen-2.5-7B-Instruct \
  --record-batch-size 1 \
  --include_full_meta \
  --start_idx 0 \
  --append \
  --num-workers $NUM_WORKERS \
  --max-inflight-batches $NUM_WORKERS \
  2>&1 | tee -a /root/workspace/agent-rm/log/${TASK_NAME}_${EXP_NAME}.log

  # --judge-system-file /root/workspace/agent-rm/prompts/judge_system.txt \
  # --judge-user-file   /root/workspace/agent-rm/prompts/judge_user.txt
