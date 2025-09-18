export CUDA_VISIBLE_DEVICES=0,3

python /root/workspace/agent-rm/Agent-Verifier/src/score_vanilla_infer.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/score_vanilla.yaml \
  --input  /root/workspace/agent-rm/datasets/aime2025/bon/qwen3_4b_aime2025-bon128.jsonl \
  --output /root/workspace/agent-rm/datasets/aime2025/0918/qwen3_4b_aime2025-bon128_vanilla_score1_by_qwen2.5-7b.jsonl \
  --record-batch-size 1 \
  # --judge-system-file /root/workspace/agent-rm/prompts/judge_system.txt \
  # --judge-user-file   /root/workspace/agent-rm/prompts/judge_user.txt
