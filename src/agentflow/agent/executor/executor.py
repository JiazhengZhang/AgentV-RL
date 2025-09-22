# agentflow/agent/executor/executor.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm

from agentflow.agent.basic import ToolDrivenAgent, AgentContext
from agentflow.core.interfaces import CanGenerate
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.caller import ToolCaller
from agentflow.tools.parser import TagToolParser
from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.utils.tag_util import find_tags

from agentflow.agent.planner.interfaces import Plan, Subtask
from .interfaces import SubtaskReport, ExecutionReport, SubtaskExecutor, VerificationSubtaskReport
from .prompts import _DEFAULT_SYSTEM, _USER_TPL

@dataclass
class ExecutorConfig:
    max_rounds_per_subtask: int = 6
    default_tool_max_calls: int = 3
    system_prompt: str = _DEFAULT_SYSTEM
    helper_code_snippets: List[str] = field(default_factory=list)
    helper_modules: List[str] = field(default_factory=list)


class VerificationSubtaskExecutor(SubtaskExecutor):
    def __init__(self, backend: CanGenerate, registry: Optional[ToolRegistry] = None, config: Optional[ExecutorConfig] = None):
        self.backend = backend
        self.registry = registry or ToolRegistry()
        self.config = config or ExecutorConfig()

        self.parser = TagToolParser()
        self.caller = ToolCaller(self.registry, self.parser)

        py = PythonExecutionTool(timeout_length=5)
        for code in self.config.helper_code_snippets:
            py.register_helpers_from_code(code)
        for mod in self.config.helper_modules:
            py.register_helpers_from_module(mod)
        self.registry.register(py)

        self.agent = ToolDrivenAgent(
            backend=self.backend,
            tool_caller=self.caller,
            finish_fn=lambda ctx: self._has_final(ctx),
            error_fn=lambda ctx: self._has_error(ctx),
            max_rounds=self.config.max_rounds_per_subtask,
        )

    @staticmethod
    def _has_final(ctx: AgentContext) -> bool:
        last = ctx.last_message()
        if not last: return False
        # 有 <answer> 或 <yield> 都算终止'
        answer_tags = find_tags(last.content, ["answer"])
        if not answer_tags:
            return False
        target_tag = answer_tags[-1]
        if VerificationSubtaskExecutor._to_bool(target_tag.body):
            return True
        return False
    
    @staticmethod
    def _to_bool(text: str) -> Optional[bool]:
        s = text.strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
        return None

    @staticmethod
    def _has_error(ctx: AgentContext) -> bool:
        last = ctx.last_message()
        if not last: 
            return False
        tags = ["python","answer","yield","search"]
        if find_tags(last.content,tags):
            return False
        return True
        
    
    # 批处理入口
    def execute(self, *, sequences: List[str], plans: List[Plan]) -> List[ExecutionReport]:
        return self._execute_all(sequences=sequences,plans=plans)
    
    def _execute_all(self, *, sequences: List[str], plans: List[Plan]) -> List[ExecutionReport]:
        subtasks_per_plan: List[List[Subtask]] = []
        subtask_reports_per_plan: List[List[VerificationSubtaskReport]] = [[] for _ in plans]
        for seq, plan in zip(sequences,plans):
            subtasks_per_plan.append(plan.subtasks)
        sub_task_idx = 0
        max_subtasks = 0
        for subtask_list in subtasks_per_plan:
            if len(subtask_list) > max_subtasks:
                max_subtasks = len(subtask_list)
        process_bar = tqdm(total=max_subtasks,desc="Processing subtasks:")
        while sub_task_idx < max_subtasks:
            input_msgs = []
            input_msg_indicies: List[int] = []
            for idx, (seq, plan, subtasks) in enumerate(zip(sequences,plans, subtasks_per_plan)):
                if sub_task_idx < len(subtasks):
                    input_msgs.append(self._format_subtask_prompt(seq,plan,subtasks[sub_task_idx]))
                    input_msg_indicies.append(idx)
            try:
                answers, metas = self.agent.generate(input_msgs)
            except:
                continue
            for idx, (indice, answer, meta) in enumerate(zip(input_msg_indicies,answers,metas)):
                curr_plan = subtasks_per_plan[indice]
                curr_subtask = curr_plan[sub_task_idx]
                curr_agent_context: AgentContext = meta["context"]
                
                answer_tags = find_tags(answer, ["answer"])
                if not answer_tags:
                    verdict = None
                else:
                    verdict = VerificationSubtaskExecutor._to_bool(answer_tags[-1].body)
                verify_tags = find_tags(answer, ["verify"])
                if not verify_tags:
                    verify_text = ""
                else:
                    verify_text = verify_tags[-1].body
                report=VerificationSubtaskReport(
                    subtask_id=curr_subtask.id,
                    raw_trace=answer,
                    tool_traces=curr_agent_context.tool_results,
                    rounds_used=curr_agent_context.global_round,
                    notes=curr_agent_context.meta,
                    verdict=verdict,
                    verify_text=verify_text,
                )
                subtask_reports_per_plan[indice].append(report)
            sub_task_idx+=1
            process_bar.update(1)
            
        final_reports = []
        for idx, sub_reports in enumerate(subtask_reports_per_plan):
            final_reports.append(ExecutionReport(
                sequence_id=str(idx),
                subtask_reports=sub_reports,
            ))
        return final_reports
                    
    

    def _tool_budget(self, st: Subtask) -> int:
        return int(st.tool_hint.get("max_calls",
                   self.config.default_tool_max_calls))

    def _python_allowed(self, st: Subtask) -> bool:
        return bool(st.tool_hint.get("python", True))
    
    def _format_subtask_prompt(self, sequence: str, plan: Plan, subtask: Subtask):
        user_msg = _USER_TPL.format(
            sequence=sequence,
            problem_brief=plan.problem_brief,
            asked_quantity=plan.asked_quantity,
            assumptions=plan.assumptions_required,
            sid=subtask.id,
            title=subtask.title,
            category=subtask.category,
            rationale=subtask.rationale,
            inputs=subtask.inputs,
            tool_allowed={"python": self._python_allowed(subtask)},
            tool_max=self._tool_budget(subtask),
            prod_type=subtask.expected_produce.get("type", "boolean"),
            prod_schema=subtask.expected_produce.get("schema", {}),
        )
        return [
            {"role":"system","content":self.config.system_prompt},
            {"role":"user","content":user_msg},
        ]
        
def simple_aggregate_verdict(plan: Plan, report: ExecutionReport) -> Tuple[bool, Dict[str, int]]:
    passed = failed = uncertain = 0
    for r in report.subtask_reports:
        if not isinstance(r, VerificationSubtaskReport):
            continue
        v = r.verdict
        if v is True:
            passed += 1
        elif v is False:
            failed += 1
        else:
            uncertain += 1
    final = (failed == 0 and passed > 0)
    return final, {"passed": passed, "failed": failed, "uncertain": uncertain}
    
