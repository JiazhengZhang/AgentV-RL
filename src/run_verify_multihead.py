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
from agentflow.agent.plan import MultiturnPlanSubtaskAgent, BackwardVerifyAgent
from agentflow.inference.scorers.base import BoolLogitsScorer
from agentflow.utils.json_util import JsonUtil
from agentflow.utils.log_util import get_logger
from agentflow.utils.tag_util import find_tags
from agentflow.utils.vllm import free_vllm_mem, SupportVllm


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

def _to_bool(text: str) -> Optional[bool]:
    if text is None:
        return None
    s = str(text).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None

@ray.remote(num_gpus=1, max_restarts=8, max_task_retries=8)
class JudgeWorker:
    def __init__(self, config: Dict[str, Any], system_prompt: str, user_prompt: str):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        backend = VllmChoiceLogitsBackend(config)
        backend.set_chat_template_defaults(enable_thinking=True)
        self.backend = backend
        reg = ToolRegistry()
        reg.register(PythonExecutionTool())
        self.agent = MultiturnPlanSubtaskAgent(
            backend=backend,
            tool_registry=reg,
        )
        
        self.backward_agent = BackwardVerifyAgent(
            backend=backend,
            tool_registry=reg,
        )
        
        self.scorer = BoolLogitsScorer(backend)

    def score_batch(self, payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in payload:
            questions: List[str] = item["questions"]
            answers: List[str] = item["answers"]
            # forwaard score
            msgs, gen_metas = self.agent.generate(questions, answers)
            forward_verdicts: list[Optional[bool]] = []
            for msg in msgs:
                if not msg:
                    forward_verdicts.append(None)
                    continue
                ans_tags = find_tags(msg[-1].content,["answer"])
                if not ans_tags:
                    forward_verdicts.append(False)
                    continue
                forward_verdicts.append(_to_bool(ans_tags[-1].body))
            if isinstance(self.backend, SupportVllm):
                with free_vllm_mem(self.backend):
                    forward_scores, score_metas = self.agent.score(msgs, self.scorer)
            else:
                forward_scores, score_metas = self.agent.score(msgs, self.scorer)
                
            extras = []
            final_verdicts = []
            final_scores = forward_scores.copy()
            backward_samples: List[int] = []
            
            # forward merge
            for idx, (fmsg, fscore, verdict) in enumerate(zip(msgs, forward_scores, forward_verdicts)):
                final_verdicts.append(verdict)
                if not verdict:
                    forward_scores[idx] = 0
                    final_scores[idx] = 0
                    backward_samples.append(idx)
                else:
                    backward_samples.append(idx)
                tool_counts = 0
                for message in fmsg:
                    message.dict_data = None
                    if message.role == "tool":
                        tool_counts += 1
                extras.append({
                    "forward":{"tool_counter":tool_counts, "process": fmsg, "score":fscore, "verdict":verdict},
                    "backward":{},
                    "verdict": verdict,
                    "score": fscore,
                    })
                
            # backward score
            if backward_samples:
                
                backward_questions = [questions[idx] for idx in backward_samples]
                backward_answers = [answers[idx] for idx in backward_samples]
                
                b_msgs, bgen_metas = self.backward_agent.generate(backward_questions, backward_answers)
                backward_verdicts: list[Optional[bool]] = []
                for bmsg in b_msgs:
                    if not bmsg:
                        backward_verdicts.append(None)
                        continue
                    ans_tags = find_tags(bmsg[-1].content,["answer"])
                    if not ans_tags:
                        backward_verdicts.append(None)
                        continue
                    backward_verdicts.append(_to_bool(ans_tags[-1].body))
                
                if isinstance(self.backend, SupportVllm):
                    with free_vllm_mem(self.backend):
                        backward_scores, score_metas = self.backward_agent.score(b_msgs, self.scorer)
                else:
                    backward_scores, score_metas = self.backward_agent.score(b_msgs, self.scorer)
                    
                for local_indice, (idx, bmsg, bscore, bverdict) in enumerate(zip(backward_samples, b_msgs, backward_scores, backward_verdicts)):
                    forward_score = forward_scores[idx]
                    forward_verdict = forward_verdicts[idx]
                    if not bverdict:
                        backward_scores[local_indice] = 0
                        bscore = 0
                    final_scores[idx] = (forward_score + bscore) / 2
                    
                    if (forward_verdict is True and bverdict is True):
                        final_verdicts[idx] = True
                    else:
                        final_verdicts[idx] = False
                        
                    tool_counts = 0
                    for message in bmsg:
                        message.dict_data = None
                        if message.role == "tool":
                            tool_counts += 1
                        
                    extras[idx]["verdict"] = final_verdicts[idx]
                    extras[idx]["score"] = final_scores[idx]
                    extras[idx]["backward"] =  {"tool_counter":tool_counts, "process": bmsg, "score":bscore, "verdict":bverdict}
                
            
            metas = [{"score": score, "verdict": verdict, "id": idx,"extra": e} for idx, (score, verdict, e) in enumerate(zip(final_scores, final_verdicts, extras))]
            metas = JsonUtil.json_sanitize(metas)  
            out.append({"scores": final_scores, "metas": metas, "count": len(questions)})
        return out




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("LLM-as-Judge with Ray (batch-by-batch, low-memory)")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--model_path", default=None, type=str)
    p.add_argument("--record-batch-size", type=int, default=16)
    p.add_argument("--append", action="store_true")
    p.add_argument("--join-template", type=str, default="")
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
    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    args = parse_args()
    ensure_parent_dir(args.output)

    system_prompt = ""
    user_prompt = ""
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
            question: str = block.get("question", "")
            samples = block.get("samples", []) or []
            questions = [question for _ in range(len(samples))]
            cur_blocks.append(block)
            cur_payload.append({"questions": questions, "answers": samples})
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
        idxs = []
        for block in blocks:
            if "idx" in block.keys():
                idxs.append(block.get("idx"))
        logger.info(f"Submitted batch to worker#{wid}, size={len(payload)}, idxs={idxs}")
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
            idxs = []
            for block in blocks:
                if "idx" in block.keys():
                    idxs.append(block.get("idx"))
            logger.info(f"Submitted batch to worker#{wid}, size={len(payload)}, idxs={idxs}")

        while inflight:
            ready, rest = ray.wait([obj for (_, obj, _) in inflight], num_returns=1, timeout=8)
            if not ready:
                while len(inflight) < window and submit_one():
                    pass
                continue
            obj_done = ready[0]
            i = next(k for k, (curr_wid, o, _) in enumerate(inflight) if o == obj_done)
            wid, _, blocks = inflight.pop(i)
            try:
                items = ray.get(obj_done)  # List[{"scores":..., "metas":..., "count":...}]
            except Exception as e:
                logger.error(f"Worker#{wid} failed while getting result. Error: {e}")
            for blk, item in zip(blocks, items):
                L = int(item["count"])
                scores = item["scores"]
                metas = item["metas"]
                blk_out = dict(blk)
                if "idx" not in blk_out.keys():
                    blk_out["idx"]=10000
                idx = blk_out.get("idx",10000)
                logger.info(f"Worker#{wid} finished task with idx {idx}")
                evaluations: List[Dict[str, Any]] = blk_out.get("evaluations") or [dict() for _ in range(L)]
                for eva, meta, score in zip(evaluations, metas, scores):
                    eva["judge"] = meta.get("verdict")
                    eva["score"] = score
                    meta["eval"] = eva
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
                
            while len(pending_heap) >= 1:
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
