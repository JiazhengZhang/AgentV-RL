export CUDA_VISIBLE_DEVICES=4,5

python /root/workspace/agent-rm/Agent-Verifier/src/score_tool_agent_with_search.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/score_tool.yaml \
  --input  /root/workspace/agent-rm/datasets/aime2025/bon/qwen3_4b_aime2025-bon128.jsonl \
  --output /root/workspace/agent-rm/datasets/aime2025/0919/qwen3_4b_aime2025-bon128_tool_search_score3_by_qwen3-4b.jsonl \
  --record-batch-size 1 \
  --include_full_meta \
  --append \
  # --judge-system-file /root/workspace/agent-rm/prompts/judge_system.txt \
  # --judge-user-file   /root/workspace/agent-rm/prompts/judge_user.txt
