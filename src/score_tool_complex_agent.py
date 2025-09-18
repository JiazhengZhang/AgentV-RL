# llm_judge_scoring.py
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from agentflow.config import load_config
from agentflow.utils.json_util import JsonUtil

from agentflow.backend.vllm_logits import VllmChoiceLogitsBackend
from agentflow.agent.basic import ToolDrivenAgent, AgentContext
from agentflow.tools.caller import ToolCaller
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.parser import TagToolParser
from agentflow.tools.search.base_search import AsyncSearchTool
from agentflow.tools.search.backend.searxng import SearxngBackend
from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.inference.scorers.generative_scorer import BoolLogitsGenerativeScorer
from agentflow.utils.tag_util import find_tags




SYSTEM_PROMPT_TOOL_NO_SEARCH = """
You are a tool-augmented math verifier.

GOAL
From a chat-like SEQUENCE (QUESTION + ASSISTANT’S REASONING), verify step-by-step whether the reasoning solves the QUESTION. Prefer minimal checks; do NOT re-solve unless a tiny subcheck is strictly required.

TOOLS
You may emit at most 3 <python>…</python> blocks total. Use ONLY for calculations with math/sympy/standard lib (no numpy, no input/os/system/loops).
Preloaded helpers callable inside <python>:

* is_int(x), is_rational(x), simplify_str(s)
* nCk(n,k), nPk(n,k), factors(n)->dict, prime_list(N)
* mod_inv(a,m), crt_pair(a1,m1,a2,m2)
* eval_poly(coeffs,x), roots_quadratic(a,b,c)
* check_equal_expr(lhs,rhs), sample_check(expr, [{'x':1}, …])

ALLOWED TAGS

* <rubric>…</rubric> — round 1 only; list 2–4 decisive axes.
* <step>…</step> — exactly one per round: state micro-goal, two key axes, and a brief Known/Unknown ledger; be non-repetitive.
* <python>…</python> — optional tool call for THIS round.
* <verify>…</verify> — 60–140 words; start with the intent check; then 2–4 minimal checks.
* <answer>true|false</answer> — final verdict, exactly once in the whole session.
* <next>…</next> — “continue to the next round”.

INTERACTION RULES

* Use ≤ 4 rounds. Round 1 must contain <rubric> and one <step>. Rounds 2..K: one <step>.
* After </step>, output EXACTLY ONE of: <python>…</python> OR <next/> OR (<verify>…</verify><answer>…</answer>).
* Progress rule: each <step> must add NEW evidence or tighten the verdict; no repetition.
* Failure policy: if WHAT≠RESULT (object/domain/form) or the result violates the required format/range, or an introduced premise is essential, immediately output <verify>…</verify><answer>false</answer>.
* Never output incomplete tags.

ROUND PLAN (follow unless early termination)

• Step 1 — Intent & object check:
Extract WHAT (exact requested quantity/constraints) and RESULT (what the assistant actually computed/claimed).
Decide MATCH: yes/no and FORMAT: ok/bad (mod range, interval, integer/radical form, probability in [0,1], etc.).
If mismatch or bad format → conclude false.
• Step 2 — Premises & modeling audit:
List Given[…] vs Introduced[…]; map any variable transforms and intervals (track endpoints/special cases).
If an introduced premise is used critically → conclude false.
• Step 3 — Critical-step verification (minimal compute):
Pick ONE decisive equation/identity/inequality from the reasoning; validate via simplify/ substitution/ tiny sampling.
• Step 4 — Edges & finalize:
Check boundaries/branches/monotonicity claims that affect correctness. If all pass, produce <verify> then <answer>.

MANDATORY INTENT CHECK (FIRST lines inside <verify>)
WHAT: …
RESULT: …
MATCH: yes/no
FORMAT: ok/bad
PREMISES: Given=[…]; Introduced=[…]

"""

USER_PROMPT="""
The sequence for judge:
{sequence}

Your judgement:
- Start with <rubric>…</rubric>.
- Include exactly one <think>…</think>.
- If you need a calculation to verify a step, after </think> output ONLY one <python>…</python> block (and nothing else) in this round.
- Otherwise (or after the tool round), output:
  <verify>…</verify>
  <answer>true|false</answer>
"""

HELPERS_MATH = r"""
import math
import sympy as sp
from sympy import Integer, Rational, symbols, Eq, simplify, factorint

def is_int(x):
    try:
        return bool(Integer(x) == int(x))
    except Exception:
        try:
            return bool(sp.Integer(sp.nsimplify(x)) == int(sp.nsimplify(x)))
        except Exception:
            return False

def is_rational(x):
    try:
        sp.nsimplify(x)
        return True
    except Exception:
        return False

def nCk(n, k):
    n, k = int(n), int(k)
    if k < 0 or k > n: return 0
    return math.comb(n, k)

def nPk(n, k):
    n, k = int(n), int(k)
    if k < 0 or k > n: return 0
    out = 1
    for i in range(k):
        out *= (n - i)
    return out

def factors(n: int):
    return dict(factorint(int(n)))

def prime_list(nmax: int):
    return list(sp.primerange(2, int(nmax)+1))

def mod_inv(a: int, m: int):
    a, m = int(a), int(m)
    g, x, y = sp.gcdex(a, m)
    if g != 1:
        raise ValueError("No modular inverse")
    return int(x % m)

def crt_pair(a1, m1, a2, m2):
    sol = sp.ntheory.modular.crt([int(m1), int(m2)], [int(a1)%int(m1), int(a2)%int(m2)])
    if sol is None: 
        raise ValueError("CRT infeasible")
    return int(sol[0]), int(sol[1])

def eval_poly(coeffs, x):
    # coeffs: highest degree first
    val = 0
    for c in coeffs:
        val = val * x + c
    return val

def roots_quadratic(a,b,c):
    a,b,c = map(sp.nsimplify, (a,b,c))
    x = sp.symbols('x')
    sol = sp.solve(sp.Eq(a*x**2 + b*x + c, 0), x)
    return sol

def simplify_str(expr_str: str) -> str:
    try:
        e = sp.sympify(expr_str)
        return str(sp.simplify(e))
    except Exception:
        return expr_str

def check_equal_expr(lhs_str: str, rhs_str: str) -> bool:
    try:
        lhs = sp.simplify(sp.sympify(lhs_str))
        rhs = sp.simplify(sp.sympify(rhs_str))
        return bool(sp.simplify(lhs - rhs) == 0)
    except Exception:
        return False

def sample_check(expr_str: str, subs_list):
    # subs_list: [{'x':1,'y':2}, ...]
    try:
        e = sp.sympify(expr_str)
        out = []
        for subs in subs_list:
            val = e.subs({sp.symbols(k): v for k, v in subs.items()})
            out.append(sp.N(val))
        return out
    except Exception as ex:
        return [f'ERR:{ex}']
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
    默认格式： "User: {prompt}\nAssistant: {response}"
    """
    prompt_text = block.get("question", "")
    samples = block.get("samples", []) or []
    seqs: List[str] = []
    for samp in samples:
        resp = to_text(samp)
        seq = join_template.format(prompt=prompt_text, response=resp)
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
    if flat_sequences:
        scores, metas = scorer.score(flat_sequences)

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
):
    ensure_parent_dir(output_path)

    # 1) 初始化后端（同时可用于生成与 choice_probs）
    config = load_config(config_path)
    backend = VllmChoiceLogitsBackend(config)
    registry = ToolRegistry()
    # search_tool = AsyncSearchTool(SearxngBackend("http://127.0.0.1:8888"))
    py_tool = PythonExecutionTool()
    py_tool.register_helpers_from_code(HELPERS_MATH)
    # registry.register(search_tool)
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
        error_fn=_error_gen
    )

    # 2) 读取 judge 模板；若未提供则用 BoolLogitsGenerativeScorer 默认
    system_prompt = SYSTEM_PROMPT_TOOL_NO_SEARCH
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

    for _, block in read_jsonl_stream(input_path, max_records=max_records):
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
    p.add_argument("--join-template", type=str, default="User: {prompt}\nAssistant: {response}",
                   help="How to form judge 'sequence' from prompt + response")
    p.add_argument("--judge-system-file", type=str, default=None, help="Optional system prompt file for the judge")
    p.add_argument("--judge-user-file", type=str, default=None, help="Optional user prompt file for the judge")
    p.add_argument("--max-records", type=int, default=None, help="Only process first N records")
    p.add_argument("--include_full_meta", action="store_true", help="Write all inference meta to result")
    return p.parse_args()


def main():
    args = parse_args()
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
    )


if __name__ == "__main__":
    main()
