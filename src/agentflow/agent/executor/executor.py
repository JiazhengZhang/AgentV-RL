# agentflow/agent/executor/executor.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm

from agentflow.agent.basic import ToolDrivenAgent, AgentContext
from agentflow.core.interfaces import CanGenerate, SupportChatTemplate
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.caller import ToolCaller
from agentflow.tools.parser import TagToolParser
from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.common.messages import Message, trans_messages_to_standard, trans_messages_to_text
from agentflow.utils.tag_util import find_tags
from agentflow.utils.log_util import get_logger

from agentflow.agent.planner.interfaces import Plan, Subtask
from .interfaces import SubtaskReport, ExecutionReport, SubtaskExecutor, VerificationSubtaskReport
from .prompts import _DEFAULT_SYSTEM, _USER_TPL

@dataclass
class ExecutorConfig:
    max_rounds_per_subtask: int = 3
    default_tool_max_calls: int = 2
    system_prompt: str = _DEFAULT_SYSTEM
    helper_code_snippets: List[str] = field(default_factory=list)
    helper_modules: List[str] = field(default_factory=list)
    enable_early_stop: bool = False
    use_tqdm: bool = False
    save_full_meta: bool = True



class VerificationSubtaskExecutor(SubtaskExecutor):
    def __init__(self, backend: CanGenerate, registry: Optional[ToolRegistry] = None, config: Optional[ExecutorConfig] = None):
        self.backend = backend
        self.registry = registry or ToolRegistry()
        self.config = config or ExecutorConfig()
        self.logger = get_logger()
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
        
    def _make_skipped_report(self, st: Subtask, reason: str = "early_stop_on_fail") -> VerificationSubtaskReport:
        return VerificationSubtaskReport(
            subtask_id=st.id,
            raw_trace="<skipped/>",
            tool_traces=[],
            rounds_used=0,
            notes={"skipped": True, "reason": reason},
            verdict=None,              # 占位：None
            verify_text="",
        )

    @staticmethod
    def _has_final(ctx: AgentContext) -> bool:
        last = ctx.last_message()
        if not last: return False
        # 有 <answer> 或 <yield> 都算终止'
        answer_tags = find_tags(last.content, ["answer"])
        if not answer_tags:
            return False
        else:
            return True
    
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
        

    def execute(self, *, sequences: List[str], plans: List[Plan], **kwargs) -> List[ExecutionReport]:
        """Execute all subtask of a batch of plans
        """
        return self._execute_all(sequences=sequences,plans=plans, **kwargs)
    
    
    def execute_one(self, sequences: List[str], plans: List[Plan], subtasks: List[Subtask], **kwargs) -> List[VerificationSubtaskReport]:
        input_msgs = []
        for idx, (seq, subtask, plan) in enumerate(zip(sequences, subtasks, plans)):
            input_msgs.append(self._format_subtask_prompt(seq, plan, subtask))
        answers, metas = self.agent.generate(input_msgs, None, **kwargs)
        reports: List[VerificationSubtaskReport] = []
        for idx, (answer, meta) in enumerate(zip(answers, metas)):
            plan = plans[idx]
            subtask = subtasks[idx]
            curr_agent_context: AgentContext = meta.get("context",AgentContext())
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
                subtask_id=subtask.id,
                raw_trace=answer,
                tool_traces=curr_agent_context.tool_results,
                rounds_used=curr_agent_context.global_round,
                notes=curr_agent_context.meta,
                verdict=verdict,
                verify_text=verify_text,
                round_messages=curr_agent_context.all_messages(),
            )
            reports.append(report)
        return reports
    
    def _execute_all(self, *, sequences: List[str], plans: List[Plan], **kwargs) -> List[ExecutionReport]:
        subtasks_per_plan: List[List[Subtask]] = []
        subtask_reports_per_plan: List[List[VerificationSubtaskReport]] = [[] for _ in plans]
        stopped: List[bool] = [False] * len(plans)
        max_subtasks = max((len(sts) for sts in subtasks_per_plan), default=0)
        for seq, plan in zip(sequences,plans):
            subtasks_per_plan.append(plan.subtasks)
        sub_task_idx = 0
        max_subtasks = 0
        for subtask_list in subtasks_per_plan:
            if len(subtask_list) > max_subtasks:
                max_subtasks = len(subtask_list)
        if self.config.use_tqdm:
            process_bar = tqdm(total=max_subtasks,desc="Processing subtasks:")
        
        while sub_task_idx < max_subtasks:
            input_msgs = []
            input_msg_indicies: List[int] = []
            
            for idx, (seq, plan, subtasks) in enumerate(zip(sequences,plans, subtasks_per_plan)):
                if stopped[idx]:
                    continue
                
                if sub_task_idx < len(subtasks):
                    input_msgs.append(self._format_subtask_prompt(seq,plan,subtasks[sub_task_idx]))
                    input_msg_indicies.append(idx)
                    
            if len(input_msgs) < 1:
                break
            try:
                answers, metas = self.agent.generate(input_msgs, None, **kwargs)
                
            except Exception as e:
                self.logger.exception(e)
                sub_task_idx += 1
                if self.config.use_tqdm:
                    process_bar.update(1)
                continue
            for idx, (plan_idx, answer, meta) in enumerate(zip(input_msg_indicies,answers,metas)):
                curr_plan = subtasks_per_plan[plan_idx]
                curr_subtask = curr_plan[sub_task_idx]
                curr_agent_context: AgentContext = meta.get("context",AgentContext())
                
                return_round_msgs = kwargs.get("save_round_messages", False)
                if return_round_msgs:
                    round_messages_to_save = curr_agent_context.all_messages()
                else:
                    round_messages_to_save = []
                    
                if self.config.save_full_meta == False:
                    curr_agent_context.meta.clear()
                    curr_agent_context.tool_results.clear()
                
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
                    round_messages=round_messages_to_save,
                )
                subtask_reports_per_plan[plan_idx].append(report)
                
                if verdict is False and self.config.enable_early_stop:
                    stopped[plan_idx] = True
            sub_task_idx+=1
            if self.config.use_tqdm:
                process_bar.update(1)
            
        for plan_idx, (plan, reports) in enumerate(zip(plans, subtask_reports_per_plan)):
            num_have = len(reports)
            num_need = len(plan.subtasks)
            if num_have < num_need:
                # 从 num_have 开始的剩余 subtask 全部补占位
                for st in plan.subtasks[num_have:]:
                    reports.append(self._make_skipped_report(st, reason="early_stop_on_fail"))
            
        final_reports = []
        for idx, sub_reports in enumerate(subtask_reports_per_plan):
            final_reports.append(ExecutionReport(
                sequence_id=str(idx),
                subtask_reports=sub_reports,
            ))
        return final_reports
                    
    

    def _tool_budget(self, st: Subtask) -> int:
        try:
            return int(st.tool_hint.get("max_calls",
                   self.config.default_tool_max_calls))
        except:
            return self.config.default_tool_max_calls

    def _python_allowed(self, st: Subtask) -> bool:
        try:
            return bool(st.tool_hint.get("python", True))
        except:
            return False
    
    def _format_subtask_prompt(self, sequence: str, plan: Plan, subtask: Subtask):
        try:
            prod_type =  subtask.expected_produce.get("type", "boolean")
            prod_schema = subtask.expected_produce.get("schema", {})
        except:
            prod_type = "boolean"
            prod_schema = "schema"
           
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
            prod_type=prod_type,
            prod_schema=prod_schema,
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
    
