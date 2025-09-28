from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Callable, Sequence
import traceback
import logging


from .context import AgentContext
from agentflow.core.interfaces import CanGenerate, CanChoiceProbs, CanRMScores
from agentflow.inference.scorers.generative_scorer import BoolLogitsGenerativeScorer
from agentflow.common.messages import Message, trans_messages_to_standard, trans_messages_to_text
from agentflow.tools.base import ToolCallResult
from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.caller import ToolCaller       
from agentflow.tools.base import ToolParser      
from agentflow.agent.planner.llm_planner import LLMPlanner
from agentflow.agent.executor.executor import VerificationSubtaskExecutor, simple_aggregate_verdict, ExecutorConfig
from agentflow.agent.executor.integrator import integrate_and_predict, stats_and_has_fail, build_rollout_for_model
from agentflow.utils.chat_template import is_chat_messages   
from agentflow.utils.log_util import get_logger


SYSTEM_PROMPT = """
You are a deterministic judge. You will receive a single rollout text that summarizes a verification agent’s plan,
its subtasks, and execution traces. Decide whether the final solution is correct based ONLY on the rollout.
Do not use any outside knowledge. Be strict, binary, and consistent.

Decision rules (domain-agnostic):
1) Hard fail: If ANY subtask result has verdict="false", or the <summary><counts ... failed="k"> has k>0,
   you MUST answer <answer>false</answer>.
2) Consistency: If the rollout contains contradictory claims about the asked quantity that are not proven equivalent
   (e.g., differing values/expressions), answer <answer>false</answer>.
3) Sufficiency: If no clear bridge from premises to the claimed answer is established (e.g., reasoning lacks a justified
   connection to the asked quantity), or all verdicts are "none", answer <answer>false</answer>.
4) Otherwise (no fails, no unresolved contradictions, and the reasoning establishes a valid bridge and final consistency),
   answer <answer>true</answer>.

Output format:
- Return EXACTLY one XML tag: <answer>true</answer> OR <answer>false</answer>.
- Lowercase only. No extra text, no explanations, no quotes, no code fences.
"""

USER_PROMPT="""
Full Agent rollout: 
{sequence}

"""

class PlanSubtaskAgent(CanRMScores):
    
    def __init__(
        self,
        backend: CanGenerate,
        prob_calculator: CanChoiceProbs,
        tool_registry: Optional[ToolRegistry] = None,
        final_system_prompt: Optional[str] = None,
        final_user_prompt: Optional[str] = None,
    ):
        super().__init__()
        self.backend = backend
        self.planner = LLMPlanner(backend)
        registry = tool_registry or ToolRegistry()
        py_tool = PythonExecutionTool()
        registry.register(py_tool)
        self.executor = VerificationSubtaskExecutor(
            backend=backend,
            registry=registry,
            config=ExecutorConfig(enable_early_stop=True)
        )
        self.scorer = BoolLogitsGenerativeScorer(
            generator=backend,
            prob_calculator=prob_calculator,
            system_prompt=final_system_prompt or SYSTEM_PROMPT,
            user_prompt=final_user_prompt or USER_PROMPT,
        )
        
    
    def score(
        self, 
        sequences: Sequence[str], 
        extra: List[Dict] = None, 
        **kwargs
    ) -> Tuple[List[float],List[Dict]]: 
        try:
            plans = self.planner.plan(sequences)
            reports = self.executor.execute(sequences=sequences,plans=plans)
            results = integrate_and_predict(
                sequences=sequences,
                plans=plans,
                reports=reports,
                scorer=self.scorer,
            )
            scores = [0] * len(sequences)
            metas = [{} for _ in range(len(sequences))] 
            for idx, result in enumerate(results):
                scores[idx]=result.score
                metas[idx]["raw_text"]=result.rollout_text
                metas[idx]["plan"]=plans[idx]
                metas[idx]["subtask_reports"]=reports[idx]
                metas[idx]["final_result"]=result
                metas[idx]["judge"]=result.verdict
            return scores, metas
        except Exception as e:
            scores = [-1] * len(sequences)
            metas = [{"raw_text":"","judge":None} for _ in range(len(sequences))] 
            traceback.print_exc()
            return scores, metas
        
        
        
class RulePlanSubtaskAgent(CanRMScores):
    """直接根据规则得到正确率以及分数

    Args:
        CanRMScores (_type_): _description_
    """
    def __init__(
        self,
        backend: CanGenerate,
        tool_registry: Optional[ToolRegistry] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__()
        self.backend = backend
        self.planner = LLMPlanner(backend)
        registry = tool_registry or ToolRegistry()
        py_tool = PythonExecutionTool()
        registry.register(py_tool)
        self.executor = VerificationSubtaskExecutor(
            backend=backend,
            registry=registry,
        )
        self.logger = logger
    
        
    
    def score(
        self, 
        sequences: Sequence[str], 
        extra: List[Dict] = None, 
        **kwargs
    ) -> Tuple[List[float],List[Dict]]: 
        try:
            plans = self.planner.plan(sequences)
            reports = self.executor.execute(sequences=sequences,plans=plans)
            
            scores = [0] * len(sequences)
            metas = [{} for _ in range(len(sequences))] 
            for idx, (seq, plan,report) in enumerate(zip(sequences, plans, reports)):
                stats, has_fail = stats_and_has_fail(report)
                rolout = build_rollout_for_model(sequence=seq,plan=plan,report=report)
                if has_fail:
                    num_corr = stats.get("passed",0)
                    score = 0
                    if num_corr > 0:
                        score = min(0.5,num_corr * 0.1)
                    verdict = False
                else:
                    score = 1
                    verdict = True
                scores[idx]=score
                metas[idx]["raw_text"]=rolout
                metas[idx]["plan"]=plan
                metas[idx]["subtask_reports"]=report
                metas[idx]["final_result"]=stats
                metas[idx]["judge"]=verdict
            return scores, metas
        except Exception as e:
            scores = [-1] * len(sequences)
            metas = [{"raw_text":"","judge":None} for _ in range(len(sequences))] 
            traceback.print_exc()
            return scores, metas