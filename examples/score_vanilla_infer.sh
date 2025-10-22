export CUDA_VISIBLE_DEVICES=0,1

python /root/workspace/agent-rm/Agent-Verifier/src/score_vanilla_infer.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/score_vanilla.yaml \
  --input  /root/workspace/agent-rm/datasets/gaokao2023/bon/qwen2.5-7b-math-bon128-gaokao-2023.jsonl \
  --output /root/workspace/agent-rm/datasets/gaokao2023/1009/qwen2.5-7b-math-bon128-gaokao-2023-eval_vanilla_score1_by_qwen3-4b-think-1.jsonl \
  --record-batch-size 1 \
  --append \
  # --judge-system-file /root/workspace/agent-rm/prompts/judge_system.txt \
  # --judge-user-file   /root/workspace/agent-rm/prompts/judge_user.txt
