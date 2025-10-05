export CUDA_VISIBLE_DEVICES=3,4

python /root/workspace/agent-rm/Agent-Verifier/src/score_tool_complex_agent.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/score_tool_complex.yaml \
  --input  /root/workspace/agent-rm/datasets/math500/bon/qwen2.5-7b-math-bon128-math500.jsonl \
  --output /root/workspace/agent-rm/datasets/math500/1005/qwen2.5-7b-math-bon128-math500-eval_complex_agent_score1_by_qwen3-4b-no-think.jsonl \
  --record-batch-size 1 \
  --include_full_meta \
  --start_idx 0 \
  # --append \

  # --judge-system-file /root/workspace/agent-rm/prompts/judge_system.txt \
  # --judge-user-file   /root/workspace/agent-rm/prompts/judge_user.txt
