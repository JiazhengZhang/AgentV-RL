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


def _stats_and_has_fail(report: ExecutionReport) -> Tuple[Dict[str, int], bool]:
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
    return {"passed": passed, "failed": failed, "uncertain": uncertain}, (failed > 0)


def build_rollout_for_model(
    *,
    sequence: str,
    plan: Plan,
    report: ExecutionReport,
    include_tool_traces: bool = False,
    max_tool_chars: int = 1000
) -> str:
    ridx: Dict[str, VerificationSubtaskReport] = {
        r.subtask_id: r for r in report.subtask_reports
        if isinstance(r, VerificationSubtaskReport)
    }

    lines: List[str] = []
    lines.append(f'<rollout sequence_id="{_esc(report.sequence_id)}">')
    lines.append(f"  <sequence>{_esc(sequence)}</sequence>")
    # lines.append(f"  <problem>{_esc(plan.problem_brief)}</problem>")
    # lines.append(f"  <asked>{_esc(plan.asked_quantity)}</asked>")

    if plan.assumptions_required:
        lines.append("  <assumptions>")
        for a in plan.assumptions_required:
            lines.append(f"    <assumption>{_esc(a)}</assumption>")
        lines.append("  </assumptions>")

    if plan.meta:
        lines.append(f"  <plan_meta>{_esc(_json_compact(plan.meta))}</plan_meta>")

    lines.append("  <subtasks>")
    for st in plan.subtasks:
        st_inputs = _json_compact(st.inputs) if st.inputs else "{}"
        st_expected = _json_compact(st.expected_produce) if st.expected_produce else "{}"
        st_tool_hint = _json_compact(st.tool_hint) if st.tool_hint else "{}"

        lines.append(
            f'    <subtask id="{_esc(st.id)}" title="{_esc(st.title)}" category="{_esc(st.category)}">'
        )
        if st.rationale:
            lines.append(f"      <rationale>{_esc(st.rationale)}</rationale>")
        lines.append(f"      <inputs>{_esc(st_inputs)}</inputs>")
        lines.append(f"      <expected>{_esc(st_expected)}</expected>")
        lines.append(f"      <tool_hint>{_esc(st_tool_hint)}</tool_hint>")

        rr = ridx.get(st.id)
        if rr is None:
            lines.append('      <result verdict="none" rounds="0" tool_calls="0">')
            lines.append("        <verify></verify>")
            lines.append("      </result>")
        else:
            verdict = "true" if rr.verdict is True else ("false" if rr.verdict is False else "none")
            rounds_used = getattr(rr, "rounds_used", 0) or 0
            tool_calls = len(rr.tool_traces) if getattr(rr, "tool_traces", None) else 0
            lines.append(
                f'      <result verdict="{verdict}" rounds="{rounds_used}" tool_calls="{tool_calls}">'
            )
            vtext = _soft_trunc(rr.verify_text or "", 2000)
            lines.append(f"        <verify>{_esc(vtext)}</verify>")
            lines.append("      </result>")

            if include_tool_traces and getattr(rr, "tool_traces", None):
                lines.append("      <tools>")
                for tc in rr.tool_traces:
                    # 最小健壮序列化
                    name = getattr(tc, "tool_name", None) or getattr(tc, "name", None) or "tool"
                    raw = _json_compact(getattr(tc, "output", None) or getattr(tc, "result", None) or "")
                    raw = _soft_trunc(raw, max_tool_chars)
                    lines.append(f'        <tool name="{_esc(name)}">{_esc(raw)}</tool>')
                lines.append("      </tools>")

        raw = getattr(rr, "raw_trace", "") if rr is not None else ""
        if raw:
            lines.append(f"      <raw_trace>{_esc(_soft_trunc(str(raw), 2000))}</raw_trace>")

        lines.append("    </subtask>")
    lines.append("  </subtasks>")

    passed = sum(1 for r in report.subtask_reports if isinstance(r, VerificationSubtaskReport) and r.verdict is True)
    failed = sum(1 for r in report.subtask_reports if isinstance(r, VerificationSubtaskReport) and r.verdict is False)
    uncertain = sum(1 for r in report.subtask_reports if isinstance(r, VerificationSubtaskReport) and r.verdict is None)
    lines.append('  <summary>')
    lines.append(f'    <counts passed="{passed}" failed="{failed}" uncertain="{uncertain}"/>')
    if report.meta:
        lines.append(f"    <exec_meta>{_esc(_json_compact(report.meta))}</exec_meta>")
    lines.append("  </summary>")

    lines.append('  <instruction>')
    lines.append('    Decide the final judgement strictly as one token: "true" or "false".')
    lines.append('    Reply with exactly one XML tag: <answer>true</answer> or <answer>false</answer>.')
    lines.append('  </instruction>')
    lines.append("</rollout>")
    return "\n".join(lines)

def _to_bool(text: str) -> Optional[bool]:
    if text is None:
        return None
    s = str(text).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None


def integrate_and_predict(
    *,
    sequences: List[str],
    plans: List[Plan],
    reports: List[ExecutionReport],
    scorer: BoolLogitsGenerativeScorer,
    include_tool_traces: bool = False
) -> List[FinalPrediction]:
    """将 plan+report 整合为单条文本；若存在任何 fail，直接 verdict=False, score=0.0；
    否则（无 fail）若提供 scorer，则用 choice_probs 给出 P(true)，不提供则 score=None。
    注意：本函数不调用生成式 judge，只输出分数与短路后的结论。
    """
    predictions: List[FinalPrediction] = [None] * len(sequences)
    rollouts_for_generation = []
    rollout_for_generation_idx: List[int] = []
    per_stats: List[Dict] = []
    for idx, (sequence, plan, report) in enumerate(zip(sequences,plans,reports)):
        
        stats, has_fail = _stats_and_has_fail(report)
        rollout = build_rollout_for_model(
            sequence=sequence, plan=plan, report=report,
            include_tool_traces=include_tool_traces
        )

        if has_fail:
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
