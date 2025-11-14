export CUDA_VISIBLE_DEVICES=1,2,3,4

bash /root/workspace/agent-rm/Agent-Verifier/examples/run_verify_entry.sh \
  --task-name qwen2.5-7b-math-bon128-gaokao-2023-eval \
  --exp-name qwen3-4b-no-think-8 \
  --num-workers 4 \
  --config /root/workspace/agent-rm/Agent-Verifier/config/distrubute_verify.yaml \
  --model-path /root/workspace/agent-rm/models/qwen3-4b \
  --input /root/workspace/agent-rm/datasets/gaokao2023/bon/qwen2.5-7b-math-bon128-gaokao-2023.jsonl \
  --output-dir /root/workspace/agent-rm/datasets/gaokao2023/1114 \
  --log-dir /root/workspace/agent-rm/log \
  --start-idx 0 \
  --no-append \
  # --enable-thinking \