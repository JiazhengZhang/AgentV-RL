# llm_judge_scoring.py
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from agentflow.config import load_config
from agentflow.utils.json_util import JsonUtil
from agentflow.utils.log_util import log_config, get_logger, print_args

from agentflow.backend.vllm_logits import VllmChoiceLogitsBackend
from agentflow.agent.summary.simple import GeneratorSummarizer
from agentflow.agent.basic import ToolDrivenAgent, AgentContext
from agentflow.tools.caller import ToolCaller
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.parser import TagToolParser
from agentflow.tools.search.base_search import AsyncSearchTool
from agentflow.tools.search.backend.searxng import SearxngBackend
from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.inference.scorers.generative_scorer import BoolLogitsGenerativeScorer
from agentflow.utils.tag_util import find_tags

SYSTEM_PROMPT_TOOL="""
You are a teacher. Your task is to grade the solution, verifying correctness. Use Expected Answer to find any erroneous step in the Solution.

## GOAL
Given a problem and a solution, you must:
1) write a concise verification text that inspects the given solution step-by-step (do not re-solve unless a single local derivation is trivially needed);
2) decide whether the solution correctly solves the problem;
3) output a boolean verdict.

You must prioritize checking the original soluiton via paper checks: legality of algebraic steps, substitution mentally for proposed roots, domain and edge cases, theorem prerequisites, and consistency of the final statement with intermediate steps. Do NOT produce a fresh full solution when the provided reasoning is wrong or incomplete.

You may perform **multi-turn reasoning**. In each round:
- Always start with <rubric>…</rubric> (once at the very beginning).
- Then include exactly one <think>…</think>, with your private reasoning (state micro-goal, decisive axes, known/unknown, next step).
- After <think>, you must choose one action:
  - **Tool-call**: 
    - <python>…</python> with safe left-aligned code using only print(...) (numpy, sympy, math pre-imported).
    - <search>…</search> with a short, precise web query string (one query).
  - **Final answer**: output <verify>…</verify> (concise inspection text) and <answer>true|false</answer> (exactly once, session ends).

## ALLOWED TAGS
* <rubric> … </rubric> – required once at the very beginning, list 2–4 decisive axes you’ll check.
* <think> … </think> – required exactly once per round; state reasoning progress.
* <python> … </python> – optional, can be used at most three times across rounds, but only one per round. Left-aligned code, safe subset.
* <search> … </search> – optional, can be used once across rounds;keep query short and precise.
* <verify> … </verify> – required in the final round only, ≈60–160 words or 5–10 bullets.
* <answer> … </answer> – required in the final round only, exactly once.

### Important
- Do NOT fully re-solve the problem when the given reasoning is wrong; focus on checking legality of steps.
- Do NOT output <verify>/<answer> in the same round as <python> or <search>.
- The dialogue ends once you give <verify> and <answer>.
"""




USER_PROMPT="""
{sequence}

### Verification Task Reminder ###
- Start with <rubric>…</rubric>.
- Include exactly one <think>…</think>.
- After <think>, you must choose:
  - Call tool:
    - <python>…</python> (safe, left-aligned code using print(...); numpy, sympy, math already imported).
    - <search>…</search> (one short, precise web query).
  - Or finalize: 
    <verify>…</verify>
    <answer>true|false</answer>
- Do not output <verify>/<answer> in the same round as <python> or <search>.
- End only when <verify> and <answer> are given.

"""

DEFAULT_SEQ_TEMPLATE="""
### Problem ###
{problem}

### Solution ###
{solution}
"""

def ensure_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl_stream(path: str, *, max_records: Optional[int] = None):
    """逐行读取 JSONL，生成 (idx, obj)。"""
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
    """兼容 sample 既可能是 str 也可能是 dict 的情况."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("text", "output", "answer", "content", "message"):
            if k in x:
                v = x[k]
                return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def build_sequences_for_block(
    block: Dict[str, Any],
    *,
    join_template: str,
) -> List[str]:
    """
    把一条记录 (含 prompt 与 samples[*]) 转为若干 judge 'sequence' 文本：
    默认格式： "User: {problem}\nAssistant: {solution}"
    """
    prompt_text = block.get("question", "")
    samples = block.get("samples", []) or []
    seqs: List[str] = []
    for samp in samples:
        resp = to_text(samp)
        seq = join_template.format(problem=prompt_text, solution=resp)
        if len(seq) > 20000:
            seq = seq[(len(seq)-20000):]
        seqs.append(seq)
    return seqs


def flush_batch_and_write(
    scorer: BoolLogitsGenerativeScorer,
    batch_blocks: List[Dict[str, Any]],
    batch_sequences_per_block: List[List[str]],
    output_path: str,
    *,
    first_write_mode: str,
    include_full_meta: bool = True,
) -> str:
    """
    将一个“记录批次”的所有序列拉平，调用 scorer.score，然后按块拆回并写出。
    返回下一次写入应使用的文件模式（一般切到 'a'）。
    """
    # 1) 拉平
    flat_sequences: List[str] = []
    lens: List[int] = []
    for seqs in batch_sequences_per_block:
        lens.append(len(seqs))
        flat_sequences.extend(seqs)

    # 2) 打分
    scores: List[float] = []
    extras = [{"mission":seq} for seq in flat_sequences]
    if flat_sequences:
        scores, metas = scorer.score(flat_sequences,extras)

    # 3) 回切并写出
    offset = 0
    for block, L in zip(batch_blocks, lens):
        block_out = dict(block)  # 不污染原对象
        evaluations: List[Dict] = block_out.get("evaluations",None)
        if not evaluations:
            evaluations = [{}] * L
        if L == 0:
            block_scores: List[float] = []
            block_metas = []
        else:
            block_scores = scores[offset : offset + L]
            block_metas = metas[offset : offset + L]
        offset += L
        
        for eva, meta, score in zip(evaluations,block_metas,block_scores):
            raw_text=meta["raw_text"]
            eva["verify_text"]=raw_text
            answer_tags = find_tags(raw_text,["answer"])
            verify_tags = find_tags(raw_text,["verify"])
            if answer_tags:
                eva["judge"]=answer_tags[-1].body
            else:
                eva["judge"]=None
            if verify_tags:
                eva["verification"]=verify_tags[-1].body
            else:
                eva["verification"]=None
            eva["score"]=score

        block_out["evaluations"] = evaluations
        if include_full_meta:
            block_out["metas"] = block_metas


        if block_scores:
            best_idx = max(range(len(block_scores)), key=lambda i: block_scores[i])
            block_out["best_index"] = best_idx
            block_out["best_score"] = float(block_scores[best_idx])
            try:
                block_out["best_sample"] = to_text(block_out["samples"][best_idx])
            except Exception:
                pass
        else:
            block_out["best_index"] = None
            block_out["best_score"] = None
        write_out = JsonUtil.json_sanitize(block_out)
        JsonUtil.write_jsonlines(output_path, write_out, mode=first_write_mode)
        if first_write_mode == "w":
            first_write_mode = "a"

    return "a"


def score_streaming(
    config_path: str,
    input_path: str,
    output_path: str,
    *,
    record_batch_size: int,
    append: bool,
    join_template: str,
    judge_system_path: Optional[str],
    judge_user_path: Optional[str],
    max_records: Optional[int],
    include_full_meta: bool,
    start_idx: int = 0,
):
    ensure_parent_dir(output_path)

    # 1) 初始化后端（同时可用于生成与 choice_probs）
    config = load_config(config_path)
    logger = get_logger(config, __name__)
    log_config(logger,__name__)
    backend = VllmChoiceLogitsBackend(config)
    registry = ToolRegistry()

    summarizer = GeneratorSummarizer(
        generator=backend, prompt_template="""
You are a summarizer specialized in processing web search results.  
Your role is to extract and organize information that may help a verifier check whether a given answer to a specific question is correct.  

Given qeestion and answer: 

{mission}  

Instructions:
- Do NOT attempt to directly solve or answer the mission yourself.  
- Only summarize or highlight relevant facts, definitions, formulas, examples, or reasoning patterns from the search results that could be useful for verification.  
- If the search results contain conflicting information, point out the differences clearly.  
- Be concise, neutral, and faithful to the source texts.  
- Your output should serve as supporting evidence for a separate verifier, not as a final judgment.  

Output format:
- **Key Facts:** bullet points of extracted facts
- **Potential Conflicts or Variations:** (if any)
- **Useful References:** (optional, if sources mention notable names, theorems, datasets, etc.)

Raw search results:
{content}

"""
    )
    search_tool = AsyncSearchTool(SearxngBackend("http://127.0.0.1:8888"),enable_summarize=True,summarize_engine=summarizer)
    py_tool = PythonExecutionTool()
    py_tool.register_helpers_from_code(
    """
import math
import numpy as np
import sympy
    """
    )
    registry.register(search_tool)
    registry.register(py_tool)
    parser = TagToolParser()
    caller = ToolCaller(registry,parser)
    def _finish_gen(context: AgentContext):
        msg = context.last_message()
        tags = find_tags(msg.content,["answer"])
        if tags:
            return True
        return False
    
    def _error_gen(context: AgentContext):
        msg = context.last_message()
        tags = find_tags(msg.content,["answer","python","next"])
        if tags:
            return False
        return True
    agent = ToolDrivenAgent(
        backend=backend,
        tool_caller=caller,
        finish_fn=_finish_gen,
        max_rounds=12,
        error_fn=_error_gen,
    )

    # 2) 读取 judge 模板；若未提供则用 BoolLogitsGenerativeScorer 默认
    system_prompt = SYSTEM_PROMPT_TOOL
    user_prompt = USER_PROMPT
    if judge_system_path:
        with open(judge_system_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    if judge_user_path:
        with open(judge_user_path, "r", encoding="utf-8") as f:
            user_prompt = f.read()

    # 3) 初始化打分器（依赖注入同一个 backend）
    scorer = BoolLogitsGenerativeScorer(
        generator=agent,
        prob_calculator=backend,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # 4) 写入模式：是否先清空
    write_mode = "a" if append else "w"
    if not append:
        JsonUtil.write_jsonlines(output_path, [], mode="w")  # 清空

    # 5) 流式读入与打分
    batch_blocks: List[Dict[str, Any]] = []
    batch_sequences_per_block: List[List[str]] = []
    total = 0

    for idx, block in read_jsonl_stream(input_path, max_records=max_records):
        if idx < start_idx:
            continue
        seqs = build_sequences_for_block(block, join_template=join_template)
        batch_blocks.append(block)
        batch_sequences_per_block.append(seqs)

        if len(batch_blocks) >= record_batch_size:
            write_mode = flush_batch_and_write(
                scorer,
                batch_blocks,
                batch_sequences_per_block,
                output_path,
                first_write_mode=write_mode,
                include_full_meta=include_full_meta,
            )
            total += len(batch_blocks)
            batch_blocks.clear()
            batch_sequences_per_block.clear()

    # 尾批
    if batch_blocks:
        write_mode = flush_batch_and_write(
            scorer,
            batch_blocks,
            batch_sequences_per_block,
            output_path,
            first_write_mode=write_mode,
            include_full_meta=include_full_meta,
        )
        total += len(batch_blocks)

    print(f"[DONE] Judged {total} records → {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("LLM-as-Judge scoring (streaming JSONL)")
    p.add_argument("--config", required=True, type=str, help="Path to backend config for VllmChoiceLogitsBackend")
    p.add_argument("--input", required=True, type=str, help="Input JSONL produced by sampling (with 'prompt' and 'samples')")
    p.add_argument("--output", required=True, type=str, help="Output JSONL with scores")
    p.add_argument("--record-batch-size", type=int, default=16, help="How many records per scoring batch")
    p.add_argument("--append", action="store_true", help="Append to output instead of overwrite")
    p.add_argument("--join-template", type=str, default=DEFAULT_SEQ_TEMPLATE,
                   help="How to form judge 'sequence' from prompt + response")
    p.add_argument("--judge-system-file", type=str, default=None, help="Optional system prompt file for the judge")
    p.add_argument("--judge-user-file", type=str, default=None, help="Optional user prompt file for the judge")
    p.add_argument("--max-records", type=int, default=None, help="Only process first N records")
    p.add_argument("--include_full_meta", action="store_true", help="Write all inference meta to result")
    p.add_argument("--start_idx",type=int,default=0,help="Start idx of records")
    return p.parse_args()


def main():
    args = parse_args()
    print_args(args)
    score_streaming(
        config_path=args.config,
        input_path=args.input,
        output_path=args.output,
        record_batch_size=max(1, int(args.record_batch_size)),
        append=bool(args.append),
        join_template=args.join_template,
        judge_system_path=args.judge_system_file,
        judge_user_path=args.judge_user_file,
        max_records=args.max_records,
        include_full_meta=args.include_full_meta,
        start_idx=args.start_idx,
    )


if __name__ == "__main__":
    main()
