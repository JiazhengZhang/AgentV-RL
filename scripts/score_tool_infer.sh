export CUDA_VISIBLE_DEVICES=5

python /root/workspace/agent-rm/Agent-Verifier/src/score_tool_agent.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/score_tool.yaml \
  --input  /root/workspace/agent-rm/datasets/math500/bon/qwen2.5-7b-math-bon128-math500.jsonl \
  --output /root/workspace/agent-rm/datasets/math500/0919/qwen2.5-7b-math-bon128-math500_tool_score3_by_qwen2.5-7b.jsonl \
  --record-batch-size 1 \
  --include_full_meta \
  --append \
  --start_idx 0 \
  # --judge-system-file /root/workspace/agent-rm/prompts/judge_system.txt \
  # --judge-user-file   /root/workspace/agent-rm/prompts/judge_user.txt
