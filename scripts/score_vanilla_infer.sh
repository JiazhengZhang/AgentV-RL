export CUDA_VISIBLE_DEVICES=4

python /root/workspace/agent-rm/Agent-Verifier/src/score_vanilla_infer.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/score_vanilla.yaml \
  --input  /root/workspace/agent-rm/datasets/math500/bon/qwen2.5-7b-math-bon128-math500.jsonl \
  --output /root/workspace/agent-rm/datasets/math500/0919/qwen2.5-7b-math-bon128-math500_vanilla_score2_by_qwen2.5-7b.jsonl \
  --record-batch-size 1 \
  --append \
  # --judge-system-file /root/workspace/agent-rm/prompts/judge_system.txt \
  # --judge-user-file   /root/workspace/agent-rm/prompts/judge_user.txt
