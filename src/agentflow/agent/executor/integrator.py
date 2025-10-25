# agentflow/agent/executor/integrator.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import json
import html

from agentflow.agent.planner.interfaces import Plan, Subtask
from agentflow.agent.executor.interfaces import ExecutionReport, VerificationSubtaskReport
from agentflow.core.interfaces import CanChoiceProbs, CanGenerate
from agentflow.inference.scorers.generative_scorer import BoolLogitsGenerativeScorer
from agentflow.utils.tag_util import find_tags

@dataclass
class FinalPrediction:
    sequence_id: str
    verdict: bool
    score: Optional[float]                
    evidence: Dict[str, Any] = field(default_factory=dict)
    rollout_text: str = ""                 



def _json_compact(d: Any) -> str:
    try:
        return json.dumps(d, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(d)

def _esc(s: Any) -> str:
    if s is None: return ""
    return html.escape(str(s), quote=False)

def _soft_trunc(s: str, limit: int) -> str:
    if s is None: return ""
    if len(s) <= limit: return s
    return s[:max(0, limit-3)] + "..."


def stats_and_has_fail(report: ExecutionReport) -> Tuple[Dict[str, int], bool]:
    passed = failed = uncertain = 0
    for r in report.subtask_reports:
        if not isinstance(r, VerificationSubtaskReport):
            continue
        if r.verdict is True:
            passed += 1
        elif r.verdict is False:
            failed += 1
        else:
            uncertain += 1
    return {"passed": passed, "failed": failed, "uncertain": uncertain}, (failed > 0 or uncertain > 0)



def build_rollout_for_model(
    *,
    sequence: str,
    plan: "Plan",
    report: "ExecutionReport",
    max_chars_per_subtask: int = 4096
) -> str:
    """
    Build a rollout of format
    \<plan>
    {} # JSON format plan
    \</plan>
    \<subtasks>
      \<subtask>\</subtask>
    \</subtasks>
    """
    def _esc_attr(s: Any) -> str:
        if s is None:
            return ""
        return html.escape(str(s), quote=True)

    def _json_compact(obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return json.dumps(str(obj), ensure_ascii=False)
        
    def _indent_block(block: str, levels: int = 2, unit: str = "  ") -> str:
        if not block:
            return ""
        prefix = unit * levels
        return "\n".join(prefix + line for line in block.splitlines())

    def _coerce_list(x, default: Optional[List[str]] = None) -> List[str]:
        if x is None:
            return default or []
        if isinstance(x, list):
            return x
        return [str(x)]

    plan_dict: Dict[str, Any] = {
        "problem_brief": plan.problem_brief,
        "asked_quantity": plan.asked_quantity,
        "assumptions_required": _coerce_list(plan.assumptions_required),
    }
    if getattr(plan, "reasoning", None):
        plan_dict["reasoning"] = plan.reasoning
    if getattr(plan, "meta", None):
        if plan.meta:
            plan_dict["meta"] = plan.meta

    st_items: List[Dict[str, Any]] = []
    for st in plan.subtasks:
        item: Dict[str, Any] = {
            "id": st.id,
            "title": st.title,
            "rationale": st.rationale or "",
            "category": st.category or "",
        }
        if getattr(st, "tool_hint", None):
            if st.tool_hint:
                item["tool_hint"] = st.tool_hint
        st_items.append(item)

    plan_dict["subtasks"] = st_items

    out_lines: List[str] = []
    out_lines.append("<plan>")
    out_lines.append(_json_compact(plan_dict))
    out_lines.append("</plan>")

    id2report: Dict = {}
    for r in report.subtask_reports or []:
        sid = r.subtask_id
        if sid:
            id2report[sid] = r

    PRIORITY = ["answer", "verify", "rubric"]

    def _smart_tail_with_priority(text: str, limit: int) -> str:

        if not text or len(text) <= limit:
            return text or ""

        end = len(text)
        ans_m = find_tags(text, allowed_tags=["answer"])
        ver_m = find_tags(text, allowed_tags=["verify"])
        rub_m = find_tags(text, allowed_tags=["rubric"])

        last_ans = ans_m[-1] if ans_m else None
        last_ver = ver_m[-1] if ver_m else None
        last_rub = rub_m[-1] if rub_m else None 

        start = None

        def _span_fits(s: int, e: int) -> bool:
            return (e - s) <= limit
        if last_ans and last_ver:
            both_start = min(last_ans.start, last_ver.start)
            if _span_fits(both_start, end):
                start = both_start
        if start is None and last_ans:
            if _span_fits(last_ans.start, end):
                start = last_ans.start
        if start is None and last_ver:
            if _span_fits(last_ver.start, end):
                start = last_ver.start
        if start is None:
            return text[-limit:]
        curr_len = end - start
        budget_left = limit - curr_len
        if last_rub and last_rub.end <= start and budget_left > 0:
            needed = (end - last_rub.start)
            if needed <= limit:
                start = last_rub.start
                curr_len = end - start
                budget_left = limit - curr_len

        segment = text[start:end]
        if len(segment) > limit:
            segment = segment[-limit:]
        return segment

    out_lines.append("<subtasks>")
    for st in plan.subtasks or []:
        sid = st.id
        cat = st.category
        title = st.title

        rep = id2report.get(sid)
        raw_trace = getattr(rep, "raw_trace", "") if rep is not None else ""
        clipped = _smart_tail_with_priority(raw_trace, max_chars_per_subtask)
        clipped = _indent_block(clipped)

        out_lines.append(f'  <subtask id="{_esc_attr(sid)}" category="{_esc_attr(cat)}" title="{_esc_attr(title)}">')
        out_lines.append(clipped)
        out_lines.append("  </subtask>")
    out_lines.append("</subtasks>")

    return "\n".join(out_lines)


def _to_bool(text: str) -> Optional[bool]:
    if text is None:
        return None
    s = str(text).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None

def _to_float(text: str) -> float:
    try:
        return float(text)
    except:
        return 0


def integrate_and_predict(
    *,
    sequences: List[str],
    plans: List[Plan],
    reports: List[ExecutionReport],
    scorer: BoolLogitsGenerativeScorer
) -> List[FinalPrediction]:
    predictions: List[FinalPrediction] = [None] * len(sequences)
    rollouts_for_generation = []
    rollout_for_generation_idx: List[int] = []
    per_stats: List[Dict] = []
    for idx, (sequence, plan, report) in enumerate(zip(sequences,plans,reports)):
        
        stats, has_fail = stats_and_has_fail(report)
        rollout = build_rollout_for_model(
            sequence=sequence, plan=plan, report=report
        )
        rollout = f"{sequence}\nJudge Rollout:\n{rollout}"

        if stats.get("failed", 3) > 1:
            prediction = FinalPrediction(
                sequence_id=report.sequence_id,
                verdict=False,
                score=0.0,
                evidence=stats,
                rollout_text=rollout
            )
            predictions[idx]=prediction
            continue
        rollouts_for_generation.append(rollout)
        rollout_for_generation_idx.append(idx)
        per_stats.append(stats)
        
    if not rollouts_for_generation:
        return predictions
    
    scores, metas = scorer.score(rollouts_for_generation)
    for idx, (indice,score,meta) in enumerate(zip(rollout_for_generation_idx, scores, metas)):
        report = reports[indice]
        stat = per_stats[idx]
        rollout_text=rollouts_for_generation[idx]+meta["raw_text"]
        verdict = False
        answer_tags = find_tags(rollout_text,["answer"])
        if answer_tags:
            verdict = _to_bool(answer_tags[-1].body)
        predictions[indice]=FinalPrediction(
            sequence_id=report.sequence_id,
            verdict=verdict,
            score=score,
            evidence={"final_score_meta":meta,"stat":stat,},
            rollout_text=rollout_text,
        )
        

    return predictions


POINTWISE_SYSTEM_PROMPT = """
You are a strict judge for a verification rollout. 

## TASK

You will receive a question and an assistant's answer towards it.
you are required to produce a **score** from 0 to 10 (integer or one decimal), reflecting your confidence / quality in the reasoning and result. 
* 0 means totally incorrect / many flaws; 
* 10 means perfect, no doubts, rigorous, chain is fully justified.  

## RULE

* Begin with a <reasoning></reasoning> block with your detailed analysis for the given task.
* Finally put your score exactly in a <answer></answer> block.
* Your scoring must consider both **process** (correctness / consistency of subtask chain, no hidden leaps, domain checks, edge cases) and **outcome** (final answer correctness, matching type/range).  



"""
USER_PROMPT="""
The question, answer and agent's rollout:
{sequence}

"""

def point_wise_score(
    sequences: List[str],
    plans: List[Plan],
    reports: List[ExecutionReport],
    backend: CanGenerate,
) -> Tuple[List[float], List[Dict]]:
    scores: List[float] = [0] * len(sequences)
    rollouts_for_generation = []
    rollout_for_generation_idx: List[int] = []
    per_metas: List[Dict] = [{}] * len(sequences)
    for idx, (sequence, plan, report) in enumerate(zip(sequences,plans,reports)):
        
        stats, has_fail = stats_and_has_fail(report)
        rollout = build_rollout_for_model(
            sequence=sequence, plan=plan, report=report
        )
        rollout = f"{sequence}\nJudge Rollout:\n{rollout}"

        if stats.get("failed", 3) > 1:
            scores[idx]=0
            continue
        # rollouts_for_generation.append(rollout)
        rollouts_for_generation.append(sequence)
        rollout_for_generation_idx.append(idx)
        
    if not rollouts_for_generation:
        return scores, per_metas
    
    input_msgs = []
    for rollout in rollouts_for_generation:
        msg = [
            {"role":"system","content":POINTWISE_SYSTEM_PROMPT},
            {"role":"user","content":USER_PROMPT.format(sequence=rollout)},
        ]
        input_msgs.append(msg)
    
    results, metas = backend.generate(input_msgs)
    for idx, (indice,result, meta) in enumerate(zip(rollout_for_generation_idx, results, metas)):
        score = 0
        answer_tags = find_tags(result,["answer"])
        if answer_tags:
            score = _to_float(answer_tags[-1].body)
        scores[indice] = score
        per_metas[indice].update(meta)
        per_metas[indice]["score_rollout"]=result
    return scores, per_metas