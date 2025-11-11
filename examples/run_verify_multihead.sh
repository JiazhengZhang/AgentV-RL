export CUDA_VISIBLE_DEVICES=5,6
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


NUM_WORKERS=2

TASK_NAME=qwen2.5-7b-math-bon128-math500-100samp-eval

EXP_NAME=qwen2.5-7b-base-eval-multihead-1104-1

# ray stop

python /root/workspace/agent-rm/Agent-Verifier/src/run_verify_multihead.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/distrubute_verify_local.yaml \
  --input  /root/workspace/agent-rm/datasets/math500/bon/qwen2.5-7b-math-bon128-math500-eval-100.jsonl \
  --output /root/workspace/agent-rm/datasets/math500/1031/${TASK_NAME}-${EXP_NAME}.jsonl \
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
