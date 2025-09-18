export CUDA_VISIBLE_DEVICES=0,1

python /root/workspace/agent-rm/Agent-Verifier/src/score_vanilla_infer.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/score_vanilla.yaml \
  --input  /root/workspace/agent-rm/datasets/math500/qwen3-4b_math-500_on-128.jsonl \
  --output /root/workspace/agent-rm/datasets/math500/0918/qwen3_4b_math500-bon1b28_vanilla_score1_by_qwen2.5-7b.jsonl \
  --record-batch-size 1 \
  # --judge-system-file /root/workspace/agent-rm/prompts/judge_system.txt \
  # --judge-user-file   /root/workspace/agent-rm/prompts/judge_user.txt
