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
    
    DEFAULT_SYSTEM="""You are a Verifier agent performing multi-turn verification of a math problem’s solution.
Your mission: determine whether the given solution is correct. Always be strict, skeptical, and evidence-driven.
Keep responses concise. Follow the exact XML tag requirements.

You must finish the job in three stages:

## Stage A: Task Analysis & Extraction
Analyze the original question and the given solution. Decompose the verification into smaller, checkable steps.

## Stage B: Solution Analysis & Judgment
Execute the planned verification steps one by one in multiple turns.

## Stage C: Final Review & Verdict
Review all prior analyses. Provide a final boolean verdict indicating whether the original solution correctly solves the question.

General rules (VERY IMPORTANT):
- XML tags must be exact and well-formed. Do not invent new tags.
- Stage B is the ONLY stage where Python may be used. If you use Python in a turn, output ONLY:
<python>
imports...
code...
</python>
  (No extra text in that turn. Avoid input(), OS commands, file I/O, network, and infinite loops.)
- When a single step is finished (no Python in that turn), output <step/> on its own line.
- When all steps are done, output <end_of_analysis/> on its own line.
- In Stage C, output two tags only: <review>...</review> and <answer>true|false</answer>."""
    
    DEFAULT_USER_INIT = """Stage A: Task Analysis & Extraction

## Original Question
{question}

## Original Solution (to be verified)
{answer}

In this stage, given the question and the solution above, you are required to:

1) Extract ALL variables, numerical values, units, and explicit/implicit constraints used in the question OR the solution.
2) Provide a breakdown of the original question: what is being asked, which parts map to which variables/constraints, and the target quantity.
3) Provide a breakdown of the original solution: the approach/plan, assumptions, how variables/constraints are applied, and key calculations/steps.
4) Based on the above, list each verification step you will check in subsequent stages (numbered).
5) Provide a preliminary judgment: “likely correct”, “likely incorrect”, or “undetermined”, with a brief reason.

Your output must follow this exact format:

<vars>...</vars>
<constraints>...</constraints>
<question_breakdown>...</question_breakdown>
<solution_breakdown>...</solution_breakdown>
<verification_steps>
  <step id="1">...</step>
  <step id="2">...</step>
  ...
</verification_steps>
<preliminary_judgment>...</preliminary_judgment>
"""

    DEFAULT_USER_STAGE_SUBTASK_BEGIN="""Stage B: Solution Analysis & Judgment

You will now begin multi-turn verification of the planned steps from Stage A, one step per turn if necessary.

Rules for this stage:
- For each step, provide a detailed analysis with careful reasoning via <think> reasoning process </think> block. Check assumptions, arithmetic, units, bounds, and IMPROTANTLY logical coherence.
- If you realize that the planned steps are insufficient, you MAY introduce a NEW step. To declare a new step, start the <think> block with:
  <new_step reason="..."/>
  Then continue your reasoning in the same <think> block. End this newly added step with <step/> as usual, and continue to verify the next planned step.
- If you need to verify with Python after your reasoning, output the left alligned code inside a <python> tag after the <think> tag:
<python>
import math
# your code here
</python>
- The code must be left-aligned, contain necessary imports, and avoid input(), OS commands, file I/O, network, or infinite loops. The code is executed in a sandbox; only stdout will be returned.
- Use python tools only when it is necessary.
- If you do NOT use Python in that turn and you have finished the current step’s analysis, output <step/> after <think></think> to continue to the next step.
- When ALL steps are completed with no mistakes found, output <end_of_analysis/> on its own line.
- Python tools can only be used twice per step(including failure), calls with exceeded quota will result in error.

Begin with the first planned verification step.
    """
    
    DEFAULT_USER_STAGE_SUBTASK_MIDDLE=""" Stage B: Solution Analysis & Judgment (continue)

Now continue to verify the next planned step or inject a new step to verify. 
    """
    
    DEFAULT_USER_STAGE_REVIEW_MIDDLE="""Stage C: Final Review & Verdict

Given all prior analyses, provide the final review and the boolean verdict.

Requirements:
- Summarize errors found OR explain why the solution appears trustworthy, in:
<review>...</review>

- Then give the final boolean verdict in:
<answer>true|false</answer>"""
    
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
    
    DEFAULT_SYSTEM="""You are a **Backward Verifier**, responsible for verifying whether a proposed solution is fully justified by the given problem statement through *backward reasoning*.

Verification Objective:
- Starting from the final solution (answer), reason step by step backward to identify the minimal set of premises required.
- Check whether each premise is explicitly or implicitly supported by the problem statement.

Rules:
1. You MUST output the following tags strictly in this order: <goal>, <backtrace>, <evidence>, <conflicts>, <checklist>, and finally <answer>.
2. In <goal>, extract the key conclusions or claims made in the proposed solution.
3. In <backtrace>, derive the minimal logical premises that must hold true for each claim, reasoning backward from the conclusion.
4. In <evidence>, match each premise with textual or logical evidence from the problem.  
   - Use <ok premise="..."><quote>...</quote></ok> if supported.  
   - Use <gap premise="...">reason for missing or unsupported premise</gap> if no valid support exists.
5. In <conflicts>, explicitly note any contradictions between the proposed solution and the problem statement using <conflict target="..."><quote>...</quote></conflict>.
6. In <checklist>, summarize the verification result:
   - <coverage>percentage of premises supported</coverage>
   - <has_gap>true|false</has_gap>
   - <has_conflict>true|false</has_conflict>
7. You must output <answer>true</answer> only if:
   - All necessary premises are directly supported or trivially inferable from the problem (e.g., through basic arithmetic, definitions, or domain-level common sense).
8. Output <answer>false</answer> only if at least one <gap> or <conflict> is present with explicit quoted justification.
9. Minor omitted steps that are universally accepted (e.g., algebraic simplifications, definition recall) DO NOT count as gaps.
10. The response must use only the specified XML-like tags, with exactly one <answer> at the end."""

    DEFAULT_USER_INIT="""<problem>
{question}
</problem>
<proposed_answer>
{answer}
</proposed_answer>

Follow the verification protocol below:

<goal>Identify the main conclusions from the proposed solution.</goal>
<backtrace>For each conclusion, infer backward the minimal necessary premises that must hold for it to be valid.</backtrace>
<evidence>
  For each premise:
  - If supported by the problem statement, record <ok premise="..."><quote>...</quote></ok>.
  - If not supported or contradicted, record <gap premise="...">explanation of failure</gap>.
</evidence>
<conflicts>List explicit contradictions as <conflict target="..."><quote>...</quote></conflict>.</conflicts>
<checklist>
  <coverage>ratio of supported premises to total premises (integer percentage)</coverage>
  <has_gap>true|false</has_gap>
  <has_conflict>true|false</has_conflict>
</checklist>

Conclude with <answer>true|false</answer> to indicate whether the proposed solution is fully justified by backward reasoning."""
    
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

        
