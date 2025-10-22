# run_verify_ray.py
from __future__ import annotations
import os
import json
import argparse
import time
import heapq
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import ray
import torch

from agentflow.config import load_config
from agentflow.backend.vllm_logits import VllmChoiceLogitsBackend
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.agent.plan import PlanSubtaskSingleAgent
from agentflow.utils.json_util import JsonUtil
from agentflow.utils.log_util import get_logger

SYSTEM_PROMPT = """you control a 3-stage xml-based verification workflow. every assistant message must be valid xml and fully closed.  
no markdown, no prose outside xml. lowercase all tags.

=== global rules ===
- each round outputs one complete top-level block:
  • stage 1 → <plan>...</plan>  
  • stage 2 → <subtask>...</subtask>  
  • stage 3 → <review>...</review><answer>...</answer>  
- never leave tags open across turns.
- reasoning and tool calls happen only inside <think> or <python>.
- total flow: plan → multiple subtasks → review+answer (termination).

=== stage 1 : planning ===
role: verification planner.
goal: decompose the verification problem into 4–7 atomic, falsifiable validation units.
each unit = one binary-check step.

<plan> inner json schema:
{
 "problem_statement": "formal restatement",
 "target_quantity": "value or entity under verification",
 "required_assumptions": ["..."],
 "verification_units": [
   {
     "id": "u1",
     "title": "concise name",
     "category": "intent_validation | premise_verification | evidence_alignment | numeric_validation | final_consistency | ...",
     "tool_spec": {"python": false, "search": false, "max_calls": 1},
     "stop_on_failure": true
   }
 ],
 "termination_conditions": ["must_pass:u2","must_pass:u3","must_pass:final"]
}
output only <plan>…</plan> with valid json inside. nothing else.

=== stage 2 : subtask execution ===
role: tool-augmented verifier.
for each unit, solve and judge it end-to-end in one <subtask>…</subtask> block.

allowed tags inside <subtask>…</subtask>:
- <rubric>…</rubric> : 2–4 axes for evaluation.
- <think>…</think> : reasoning step (may appear multiple times).
- <python>…</python> : code, print(...) only, math/sympy allowed, ≤3 per subtask.
- <verify>…</verify> : ≤300 words reviewing the sub-goal only.
- <answer>true|false</answer> : exactly once, final tag.

interaction:
1. begin with <rubric>.
2. perform one or more <think> (with optional <python> blocks).
3. conclude with <verify><answer>.
4. keep each <subtask> standalone and fully closed.

example format:
<subtask>
<rubric>…</rubric><think>…</think><python>…</python>(optional, may not appear)
<think>…</think><verify>…</verify><answer>true|false</answer>
</subtask>

=== stage 3 : final judgment ===
role: strict verifier-judge.
given the full rollout (plan + subtasks), output only:
<review>
- consistency: list candidate results for asked quantity; note if rollout proves them equivalent.
- bridge: confirm reasoning chain from premises → final claim; note missing links.
- type/form: check final claim vs asked_quantity.
- scope: hidden assumptions not in required_assumptions.
</review>
<answer>true|false</answer>
"""

USER_PROMPT = """The problem, proposed solution, and prior rollout are below as {sequence}.

your task:
- first output <plan>…</plan> (stage 1).
- then, for each validation unit, output one <subtask>…</subtask> per turn until all are done.
- finally, output <review>…</review><answer>…</answer> (stage 3).
{sequence}
"""

DEFAULT_SEQ_TEMPLATE = """
### Problem ###
{problem}

### Solution ###
{solution}
"""

logger = get_logger(name = __name__)

def ensure_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def read_jsonl_stream(path: str, *, max_records: Optional[int] = None):
    with open(path, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield count, json.loads(line)
            count += 1
            if max_records is not None and count >= max_records:
                break

def to_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("text", "output", "answer", "content", "message"):
            if k in x:
                v = x[k]
                return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return json.dumps(x, ensure_ascii=False)
    return str(x)

def build_sequences_for_block(block: Dict[str, Any], *, join_template: str) -> List[str]:
    prompt_text = block.get("question", "")
    samples = block.get("samples", []) or []
    return [join_template.format(problem=prompt_text, solution=to_text(s)) for s in samples]

@ray.remote(num_gpus=1, max_restarts=6, max_task_retries=1)
class JudgeWorker:
    def __init__(self, config: Dict[str, Any], system_prompt: str, user_prompt: str):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        backend = VllmChoiceLogitsBackend(config)
        backend.set_chat_template_defaults(enable_thinking=False)
        reg = ToolRegistry()
        reg.register(PythonExecutionTool(max_rounds=15))
        self.agent = PlanSubtaskSingleAgent(
            backend=backend,
            prob_calculator=backend,
            tool_registry=reg,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_rounds=36,
        )

    def score_batch(self, payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in payload:
            seqs: List[str] = item["sequences"]
            scores, metas = self.agent.score(seqs)
            metas = JsonUtil.json_sanitize(metas)  
            out.append({"scores": scores, "metas": metas, "count": len(seqs)})
        return out

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("LLM-as-Judge with Ray (batch-by-batch, low-memory)")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--model_path", default=None, type=str)
    p.add_argument("--record-batch-size", type=int, default=16)
    p.add_argument("--append", action="store_true")
    p.add_argument("--join-template", type=str, default=DEFAULT_SEQ_TEMPLATE)
    p.add_argument("--judge-system-file", type=str, default=None)
    p.add_argument("--judge-user-file", type=str, default=None)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--include_full_meta", action="store_true")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-inflight-batches", type=int, default=2,
                   help="At most this many batches submitted to workers at once.")
    p.add_argument("--ray-address", type=str, default=None,
                   help="Ray cluster address, e.g. 'auto'. Leave empty for local.")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_parent_dir(args.output)

    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT
    if args.judge_system_file:
        with open(args.judge_system_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    if args.judge_user_file:
        with open(args.judge_user_file, "r", encoding="utf-8") as f:
            user_prompt = f.read()

    write_mode = "a" if args.append else "w"
    if not args.append:
        JsonUtil.write_jsonlines(args.output, [], mode="w")

    config = load_config(args.config)
    if args.model_path:
        config["backend"]["model_path"] = args.model_path

    ray.init(include_dashboard=False)  
    logger.info("Ray initialized.")

    num_workers = max(1, args.num_workers)
    workers = [JudgeWorker.remote(config, system_prompt, user_prompt) for _ in range(num_workers)]
    logger.info(f"Spawned {num_workers} Ray workers.")

    window = max(1, int(args.max_inflight_batches))
    total_records = 0
    reader = read_jsonl_stream(args.input, max_records=args.max_records)

    def build_one_payload_group() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """返回 (payload, original_blocks)。payload: List[{'sequences': [...] }]*record_batch_size"""
        cur_blocks: List[Dict[str, Any]] = []
        cur_payload: List[Dict[str, Any]] = []
        while len(cur_blocks) < args.record_batch_size:
            idx_block = next(reader, None)
            if idx_block is None:
                break
            idx, block = idx_block
            if idx < args.start_idx:
                continue
            block["idx"] = idx
            seqs = build_sequences_for_block(block, join_template=args.join_template)
            cur_blocks.append(block)
            cur_payload.append({"sequences": seqs})
        return cur_payload, cur_blocks
    
    def pick_worker(inflight: List[Tuple[int, "ray.ObjectRef", List[Dict[str, Any]]]],
                    num_workers: int) -> int:
        cnt = Counter(w for (w, _, _) in inflight)
        return min(range(num_workers), key=lambda w: cnt.get(w, 0))

    def submit_one() -> bool:
        payload, blocks = build_one_payload_group()
        if not payload:
            return False
        wid = pick_worker(inflight, num_workers)
        obj = workers[wid].score_batch.remote(payload)
        inflight.append((wid, obj, blocks))
        logger.info(f"Submitted batch to worker#{wid}, size={len(payload)}")
        return True

    inflight: List[Tuple[int, "ray.ObjectRef", List[Dict[str, Any]]]] = []  # (wid, obj, blocks)
    write_buffer: List[Dict[str, Any]] = []

    pending_heap = []  # 存 (idx, blk_out)
    next_to_write = args.start_idx
    
    def flush_ready(max_flush=10_000):
        nonlocal next_to_write, write_mode
        flushed = 0
        batch = []
        while pending_heap and pending_heap[0][0] == next_to_write and flushed < max_flush:
            _, item = heapq.heappop(pending_heap)
            batch.append(item)
            next_to_write += 1
            flushed += 1
        if batch:
            JsonUtil.write_jsonlines(args.output, batch, mode=write_mode)
            if write_mode == "w":
                write_mode = "a"       
            return True
        return False
    
    try:

        while len(inflight) < window:
            payload, blocks = build_one_payload_group()
            if not payload:
                break
            wid = len(inflight) % num_workers
            obj = workers[wid].score_batch.remote(payload)
            inflight.append((wid, obj, blocks))
            logger.info(f"Submitted batch to worker#{wid}, size={len(payload)}")

        while inflight:
            ready, rest = ray.wait([obj for (_, obj, _) in inflight], num_returns=1, timeout=8)
            if not ready:
                while len(inflight) < window and submit_one():
                    pass
                continue
            obj_done = ready[0]
            i = next(k for k, (_, o, _) in enumerate(inflight) if o == obj_done)
            wid, _, blocks = inflight.pop(i)

            items = ray.get(obj_done)  # List[{"scores":..., "metas":..., "count":...}]
            for blk, item in zip(blocks, items):
                L = int(item["count"])
                scores = item["scores"]
                metas = item["metas"]
                blk_out = dict(blk)
                if "idx" not in blk_out.keys():
                    blk_out["idx"]=10000

                evaluations: List[Dict[str, Any]] = blk_out.get("evaluations") or [dict() for _ in range(L)]
                for eva, meta, score in zip(evaluations, metas, scores):
                    eva["judge"] = meta.get("judge")
                    eva["score"] = score
                blk_out["evaluations"] = evaluations

                if args.include_full_meta:
                    blk_out["metas"] = metas

                if scores:
                    bi = max(range(len(scores)), key=lambda i: scores[i])
                    blk_out["best_index"] = bi
                    blk_out["best_score"] = float(scores[bi])
                    try:
                        blk_out["best_sample"] = to_text(blk_out["samples"][bi])
                    except Exception:
                        pass
                else:
                    blk_out["best_index"] = None
                    blk_out["best_score"] = None
                heapq.heappush(pending_heap, (blk_out["idx"], JsonUtil.json_sanitize(blk_out)))
                # write_buffer.append(JsonUtil.json_sanitize(blk_out))
                total_records += 1

            # if len(write_buffer) >= num_workers:
            #     write_buffer = sorted(write_buffer, key = lambda b: b.get("idx",1000))
            #     JsonUtil.write_jsonlines(args.output, write_buffer, mode=write_mode)
            #     if write_mode == "w":
            #         write_mode = "a"
            #     write_buffer.clear()
                
            while len(pending_heap) >= num_workers:
                if not flush_ready():
                    break

            while len(inflight) < window and submit_one():
                pass

    finally:
        while pending_heap:
            flush_ready()
        # if write_buffer:
        #     write_buffer = sorted(write_buffer, key = lambda b: b.get("idx",1000))
        #     JsonUtil.write_jsonlines(args.output, write_buffer, mode=write_mode)
        logger.info(f"[DONE] Judged {total_records} records → {args.output}")
        ray.shutdown()

if __name__ == "__main__":
    main()
