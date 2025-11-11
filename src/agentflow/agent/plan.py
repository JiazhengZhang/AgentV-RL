from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Callable, Sequence
import traceback
import logging


from .context import AgentContext
from agentflow.core.interfaces import CanGenerate, CanChoiceProbs, CanRMScores, SupportChatTemplate
from agentflow.inference.scorers.generative_scorer import BoolLogitsGenerativeScorer
from agentflow.inference.scorers.base import BoolLogitsScorer
from agentflow.common.messages import Message, trans_messages_to_standard, trans_messages_to_text
from agentflow.tools.base import ToolCallResult
from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.caller import ToolCaller       
from agentflow.tools.parser import TagToolParser
from agentflow.agent.basic import ToolDrivenAgent
from agentflow.agent.planner.llm_planner import LLMPlanner
from agentflow.agent.executor.executor import VerificationSubtaskExecutor, simple_aggregate_verdict, ExecutorConfig
from agentflow.agent.executor.integrator import integrate_and_predict, stats_and_has_fail, build_rollout_for_model, point_wise_score
from agentflow.utils.chat_template import is_chat_messages   
from agentflow.utils.log_util import get_logger
from agentflow.utils.tag_util import find_tags

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


POINTWISE_SYSTEM_PROMPT = """
You are a strict judge for a verification rollout. 

## TASK

You will receive the full rollout (plan / subtasks / audit / answer) and must assign:
a **score** from 0 to 10 (integer or one decimal), reflecting your confidence / quality in the reasoning and result. 
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

def _to_bool(text: str) -> Optional[bool]:
    if text is None:
        return None
    s = str(text).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None

class MultiturnPlanSubtaskAgent:
    
    DEFAULT_SYSTEM="""You are a Verifier agent responsible for performing a multi-turn verification of a math problem’s solution.
Your mission is to determine whether the given solution is correct.

You must complete the task in three stages:

## Stage A: Task Analysis & Extraction
Analyze the original question and its provided solution. Decompose the verification into smaller, checkable steps.

## Stage B: Solution Analysis & Judgment
Execute the planned verification steps one by one across multiple turns.

## Stage C: Final Review & Verdict
Review all prior analyses and provide a final boolean verdict indicating whether the original solution correctly solves the problem.
"""
    
    DEFAULT_USER_INIT = """Stage A: Task Analysis & Extraction

## Original Question
{question}

## Original Solution (to be verified)
{answer}

In this stage, given the question and the solution above, you are required to:

* Break down the original question into its key components.
* Analyze how the provided solution addresses the question step by step.
* Based on the solution steps, design corresponding verification steps. Each verification step should examine aspects such as consistency, calculations, logic, and assumptions.
"""

    DEFAULT_USER_STAGE_SUBTASK_BEGIN="""Stage B: Solution Analysis & Judgment

You will now begin multi-turn verification of the steps planned in Stage A.

* Conduct the verification according to your plan from Stage A. After completing each verification step, output exactly one <step/> tag at the end of that turn to indicate you are proceeding to the next step.
* Once all verification steps are complete, or if you identify an obvious mistake in the original solution, output exactly one <end_of_analysis/> tag to conclude Stage B.
* If complex calculations are involved, you may call the Python tool for computation. To do so, output a <python>...</python> block instead of a <step/> tag at the end of that turn.
  - The Python code must be left-aligned, include necessary imports, and avoid input(), OS commands, file I/O, networking, or infinite loops.
  - Use print() to display results so they can be correctly captured by the system.
  - Do not invoke Python for trivial or self-evident checks.

Now begin your verification:
    """
    
    DEFAULT_USER_STAGE_SUBTASK_MIDDLE=""" Stage B: Solution Analysis & Judgment (continued)
Continue verifying the next planned step.
    """
    
    DEFAULT_USER_STAGE_REVIEW_MIDDLE="""Stage C: Final Review & Verdict

Given all prior analyses, provide your final review and boolean verdict.

Requirements:
- Review all previous verification steps and summarize why each step was correct or incorrect.
- If all previous steps were confirmed to be correct, output <answer>true</answer>.
- If any step contained errors, or if you identify new inconsistencies at this stage, output <answer>false</answer>."""
    
    def __init__(
        self,
        backend: CanGenerate,
        max_rounds: int = 8,
        max_rounds_per_block: int = 6,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
    ):
        super().__init__()
        self.backend = backend
        registry = tool_registry or ToolRegistry()
        py_tool = PythonExecutionTool(use_tqdm=False)
        registry.register(py_tool)
        tool_caller = ToolCaller(registry, TagToolParser())
        self.max_rounds = max_rounds
        self.max_rounds_per_block = max_rounds_per_block
        
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        
        self.agent = ToolDrivenAgent(
            backend=backend,
            tool_caller = tool_caller,
            finish_fn = self._stage_task_analysis,
            error_fn = self._stage_task_analysis_has_error,
            max_rounds = max_rounds_per_block
        )
        
    def generate(
        self,
        questions: List[str],
        answers: List[str],
        **kwargs,
    ) -> Tuple[List[List[Message]], List[Dict]]:
        
        assert len(questions)==len(answers), "Questioins and answers should be in the same size"
        # stage 1
        start_msgs = [[
            {"role":"system","content": self.system_prompt},
            {"role":"user","content": self.DEFAULT_USER_INIT.format(question = q, answer = a)}
        ] for q, a in zip(questions, answers)]
        
        full_msgs = [Message.from_dicts(msgs) for msgs in start_msgs]
        
        resps, gen_metas = self.agent.generate(start_msgs)
        gen_contexts: List[AgentContext] = [met["context"] for met in gen_metas]
        for idx, (resp, context) in enumerate(zip(resps, gen_contexts)):
            full_msgs[idx].extend(context.all_round_messages())
            
        # stage 2
        for fms, answer in zip(full_msgs, answers):
            fms.append(Message(role="user",content=self.DEFAULT_USER_STAGE_SUBTASK_BEGIN))
        
        subtask_rounds = 0
        active_subtasks_idxs = list(range(len(questions)))
        while (active_subtasks_idxs and subtask_rounds < self.max_rounds - 2):
            subtask_rounds += 1
            input_msgs = []
            for indice in active_subtasks_idxs:
                input_msgs.append(trans_messages_to_standard(full_msgs[indice]))
                
            with self.agent.using_func(
                finish_fn=self._stage_subtask,
                error_fn=self._stage_subtask_has_error
            ):
                resps, gen_metas = self.agent.generate(input_msgs)
                curr_contexts: List[AgentContext] = [met["context"] for met in gen_metas]
            
            next_active_idxs = []
            for indice, curr_context in zip(active_subtasks_idxs, curr_contexts):
                full_msgs[indice].extend(curr_context.all_round_messages())
                
                end_flag = self._review_stage_gate(curr_context)
                if not end_flag and (subtask_rounds < self.max_rounds - 2):
                    full_msgs[indice].append(Message("user",self.DEFAULT_USER_STAGE_SUBTASK_MIDDLE))
                    next_active_idxs.append(indice)
                    
            active_subtasks_idxs = next_active_idxs
            
        # stage 3
        for fms, answer in zip(full_msgs, answers):
            fms.append(Message(role="user",content=self.DEFAULT_USER_STAGE_REVIEW_MIDDLE))
            
        input_msgs = [trans_messages_to_standard(msgs) for msgs in full_msgs]
            
        with self.agent.using_func(
            finish_fn=self._stage_review,
            error_fn=self._stage_review_has_error
        ):
            resps, gen_metas = self.agent.generate(input_msgs)
            curr_contexts: List[AgentContext] = [met["context"] for met in gen_metas]
        
        for idx, (resp, context) in enumerate(zip(resps, curr_contexts)):
            full_msgs[idx].extend(context.all_round_messages())
            
        
        return full_msgs, [{} for _ in range(len(questions))]
    
    def score(self, full_msgs: List[List[Message]], scorer: BoolLogitsScorer):
        
        std_msgs = [trans_messages_to_standard(msgs) for msgs in full_msgs]
        
        if isinstance(self.agent.backend, SupportChatTemplate):
            texts: List[str] = self.agent.backend.apply_chat_template(std_msgs)
        else:
            texts: List[str] = [trans_messages_to_text(msg) for msg in full_msgs]
            
        scores, metas = scorer.score(texts)
        
        return scores, metas
            
        
        
        
    def _stage_task_analysis(
        self,
        context: AgentContext
    ):
        last_msg = context.last_message()
        cands = find_tags(last_msg.content, ["verification_steps"])
        if cands:
            return True
        else:
            return False
        
    def _stage_task_analysis_has_error(
        self,
        context: AgentContext
    ):
        last_msg = context.last_message()
        cands = find_tags(last_msg.content, ["verification_steps"])
        if cands:
            return False
        else:
            return True
        
    def _stage_subtask(
        self,
        context: AgentContext
    ):
        last_msg = context.last_message()
        has_tag = ("<step/>" in last_msg.content)
        cands = find_tags(last_msg.content, ["python"])
        if cands :
            return False
        else:
            if has_tag:
                return True
            return False

        
    def _stage_subtask_has_error(
        self,
        context: AgentContext
    ):
        last_msg = context.last_message()
        if "<step/>" in last_msg.content:
            return False
        cands = find_tags(last_msg.content, ["step", "python"])
        if cands:
            return False
        else:
            return True
        
    def _review_stage_gate(
        self,
        context: AgentContext
    ):
        last_msg = context.last_message()
        if "<end_of_analysis/>" in last_msg.content:
            return True
        else:
            return False
        
    def _stage_review(
        self,
        context: AgentContext        
    ):
        last_msg = context.last_message()
        cands = find_tags(last_msg.content, ["answer"])
        if cands:
            return True
        else:
            return False
        
    def _stage_review_has_error(
        self,
        context: AgentContext        
    ):
        last_msg = context.last_message()
        cands = find_tags(last_msg.content, ["answer"])
        if cands:
            return False
        else:
            return True
        
class BackwardVerifyAgent:
    
    DEFAULT_SYSTEM="""You are a backward verifier whose goal is to determine whether a given mathematical solution truly proves or computes what the question requires.

Unlike a forward, step-by-step verifier, your reasoning proceeds in reverse: 
you begin from the final result or conclusion presented in the solution 
and trace backward to determine whether it can be rigorously justified by the given question and established facts.
"""

    DEFAULT_USER_INIT="""Read the following carefully and think critically:

## Original Question
{question}

## Original Solution
{answer}

Your task:
1. Begin from the final claim or result stated in the solution.
2. Reason backward: verify whether each necessary assumption, definition, or transformation 
   is explicitly or implicitly supported by the question or by earlier reasoning steps.
3. Identify any logical gaps, unjustified assumptions, circular reasoning, or numerical inconsistencies.
4. Focus on whether the final conclusion is *logically and necessarily implied* by the premises, 
   rather than being merely plausible or numerically approximate.

At the end, provide your final judgment in a single line using:
<answer>true</answer>  — if the backward reasoning confirms that the conclusion must logically hold.
<answer>false</answer> — if there exists any missing justification, invalid reversal, or unsupported dependency."""
    
    def __init__(
        self,
        backend: CanGenerate,
        max_rounds: int = 8,
        max_rounds_per_block: int = 6,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
    ):
        super().__init__()
        self.backend = backend
        registry = tool_registry or ToolRegistry()
        py_tool = PythonExecutionTool(use_tqdm=False)
        registry.register(py_tool)
        tool_caller = ToolCaller(registry, TagToolParser())
        self.max_rounds = max_rounds
        self.max_rounds_per_block = max_rounds_per_block
        
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        
        self.agent = ToolDrivenAgent(
            backend=backend,
            tool_caller = tool_caller,
            finish_fn = self._stage_review,
            error_fn = self._stage_review_has_error,
            max_rounds = max_rounds_per_block
        )
        
    def generate(
        self,
        questions: List[str],
        answers: List[str],
        **kwargs,
    ) -> Tuple[List[List[Message]], List[Dict]]:
        assert len(questions)==len(answers), "Questioins and answers should be in the same size"
        # stage 1
        start_msgs = [[
            {"role":"system","content": self.system_prompt},
            {"role":"user","content": self.DEFAULT_USER_INIT.format(question = q, answer = a)}
        ] for q, a in zip(questions, answers)]
        
        full_msgs = [Message.from_dicts(msgs) for msgs in start_msgs]
        
        resps, gen_metas = self.agent.generate(start_msgs)
        gen_contexts: List[AgentContext] = [met["context"] for met in gen_metas]
        for idx, (resp, context) in enumerate(zip(resps, gen_contexts)):
            full_msgs[idx].extend(context.all_round_messages())
        
        return full_msgs, [{} for _ in range(len(questions))]
    
    def score(self, full_msgs: List[List[Message]], scorer: BoolLogitsScorer):
        
        std_msgs = [trans_messages_to_standard(msgs) for msgs in full_msgs]
        
        if isinstance(self.agent.backend, SupportChatTemplate):
            texts: List[str] = self.agent.backend.apply_chat_template(std_msgs)
        else:
            texts: List[str] = [trans_messages_to_text(msg) for msg in full_msgs]
            
        scores, metas = scorer.score(texts)
        
        return scores, metas
    
    def _stage_review(
        self,
        context: AgentContext        
    ):
        last_msg = context.last_message()
        cands = find_tags(last_msg.content, ["answer"])
        if cands:
            return True
        else:
            return False
        
    def _stage_review_has_error(
        self,
        context: AgentContext        
    ):
        last_msg = context.last_message()
        cands = find_tags(last_msg.content, ["answer"])
        if cands:
            return False
        else:
            return True
    

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
        py_tool = PythonExecutionTool(use_tqdm=False)
        registry.register(py_tool)
        
        self.executor = VerificationSubtaskExecutor(
            backend=backend,
            registry=registry,
            config=ExecutorConfig(enable_early_stop=False,use_tqdm = True)
        )
        self.agent = self.executor.agent
        self.scorer = BoolLogitsGenerativeScorer(
            generator=backend,
            prob_calculator=prob_calculator,
            system_prompt=final_system_prompt or SYSTEM_PROMPT,
            user_prompt=final_user_prompt or USER_PROMPT,
            chat_template_backend=self.agent.backend,
        )
        
        self.logger = get_logger(name = __name__)
        
        
        
    
    def score(
        self, 
        sequences: Sequence[str], 
        extra: List[Dict] = None, 
        **kwargs
    ) -> Tuple[List[float],List[Dict]]: 
        try:
            
            plans = self.planner.plan(sequences)
            self.logger.info(f"Planning finished.")
            reports = self.executor.execute(sequences=sequences,plans=plans)
            self.logger.info("Execution finished.")
            results = integrate_and_predict(
                sequences=sequences,
                plans=plans,
                reports=reports,
                scorer=self.scorer,
            )
            self.logger.info("Final prediction finished.")
            scores = [0] * len(sequences)
            metas = [{} for _ in range(len(sequences))] 
            for idx, result in enumerate(results):
                scores[idx]=result.score
                # metas[idx]["raw_text"]=result.rollout_text
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
        
class PlanSubtaskSingleAgent(CanRMScores):
    
    def __init__(
        self,
        backend: CanGenerate,
        prob_calculator: CanChoiceProbs,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        max_rounds: int = 48,
    ):
        super().__init__()
        self.backend = backend
        
        registry = tool_registry or ToolRegistry()
        py_tool = PythonExecutionTool()
        registry.register(py_tool)
        caller = ToolCaller(registry, parser=TagToolParser())
        self.agent = ToolDrivenAgent(
            backend=backend,
            tool_caller=caller,
            finish_fn=self._has_final,
            max_rounds=max_rounds
        )
        self.scorer = BoolLogitsGenerativeScorer(
            generator=self.agent,
            prob_calculator=prob_calculator,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            chat_template_backend=backend,
            base_backend=backend,
        )

    
    
    def _has_final(self, context: AgentContext) -> bool:
        last_msg = context.last_message()
        end_of_verify = find_tags(last_msg.content, ["end_of_verification"])
        if end_of_verify:
            return True
        # else:
        #     return False
        review_tags = find_tags(last_msg.content, ["review"])
        answer_tags = find_tags(last_msg.content, ["answer"])
        if not review_tags or not answer_tags:
            return False
        last_answer = answer_tags[-1]
        last_subtask = review_tags[-1]
        if last_answer.end <= last_subtask.start:
            return False
        if last_answer.body.lower() in ("true","false"):
            return True
        return False
    
    def score(
        self, 
        sequences: Sequence[str], 
        extra: List[Dict] = None, 
        **kwargs
    ) -> Tuple[List[float],List[Dict]]: 
        scores, metas = self.scorer.score(sequences, extra, **kwargs)
        for score, meta in zip(scores, metas):
            raw_text = meta["raw_text"]
            verdict = False
            answer_tags = find_tags(raw_text,["answer"])
            if answer_tags:
                verdict = _to_bool(answer_tags[-1].body)
            meta["judge"]=verdict
                
        return scores, metas
    

        

class PlanSubtaskMultiheadAgent(CanRMScores):
    
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
            config=ExecutorConfig(enable_early_stop=False)
        )
        self.scorer = BoolLogitsGenerativeScorer(
            generator=backend,
            prob_calculator=prob_calculator,
            system_prompt=final_system_prompt or SYSTEM_PROMPT,
            user_prompt=final_user_prompt or USER_PROMPT,
        )
        
        self.agent = self.executor.agent
        
    
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
            
            pointwise_scores, pointwise_metas = point_wise_score(
                sequences=sequences,
                plans=plans,
                reports=reports,
                backend=self.backend,
            )
            
            scores = [0] * len(sequences)
            metas = [{} for _ in range(len(sequences))] 
            
            
            def _clip(num: float):
                clipped = num
                if num > 10:
                    clipped = 10
                elif num < 0:
                    clipped = 0
                return clipped / 10

            
            for idx, (result, ps, pm) in enumerate(zip(results, pointwise_scores, pointwise_metas)):
                scores[idx]=(result.score + _clip(ps)) / 2
                # metas[idx]["raw_text"]=result.rollout_text
                metas[idx]["plan"]=plans[idx]
                metas[idx]["subtask_reports"]=reports[idx]
                metas[idx]["final_result"]=result
                metas[idx]["judge"]=result.verdict
                metas[idx]["pointwise_meta"]=pm
            return scores, metas
        except Exception as e:
            scores = [-1] * len(sequences)
            metas = [{"raw_text":"","judge":None} for _ in range(len(sequences))] 
            traceback.print_exc()
            return scores, metas

        
