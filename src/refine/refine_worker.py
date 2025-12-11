# refine_worker.py
import os
import torch
import ray
import json
from typing import List, Dict, Any, Optional, Tuple, Union

from agentflow.config import load_config
from agentflow.core.interfaces import CanGenerate
from agentflow.backend.vllm import VllmBackend
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.code.python_execution_ray import PythonExecutionToolRay, create_python_actor
from agentflow.agent.plan import MultiturnPlanSubtaskAgent, BackwardVerifyAgent
from agentflow.common.messages import Message
from agentflow.utils.json_util import JsonUtil
from agentflow.utils.log_util import get_logger
from agentflow.utils.tag_util import find_tags



def _answer_to_label(body: str) -> str:
    if body is None:
        return "uncertain"
    text = str(body).strip().lower()
    if text in ("true", "yes", "1"):
        return "correct"
    if text in ("false", "no", "0"):
        return "incorrect"
    return "uncertain"


def _parse_verdict_from_msgs(msgs: List[Union[Message, Any]]) -> Tuple[str, str]:
    if not msgs:
        return "uncertain", ""

    last_msg = msgs[-1]
    content = getattr(last_msg, "content", "") if last_msg is not None else ""
    ans_tags = find_tags(content, ["answer"])
    if not ans_tags:
        label = "uncertain"
    else:
        body = ans_tags[-1].body
        label = _answer_to_label(body)

    reason = content
    if len(reason) > 30000:
        reason = reason[-30000:]
    return label, reason






class CandidateWorker:
    
    
    DEFAULT_SYSTEM_INITIAL="""You are a helpful math assistant. Solve the problem carefully.
And finnally put your solution in \\boxed{{}}.
    """
    
    DEFAULT_USER_INITIAL="""
### Question ###
{question}

Answer the question and provide your solution in the given format.

### Your Solution ###
    """
    
    DEFAULT_SYSTEM_REFINE="""You are revising your previous solution to a math problem based on a verifier's feedback.
The answers should be in \\boxed{{}}.
If the verifier marked your solution as CORRECT, keep your final answer exactly the same. Note that you still have to follow the format requirments. If previous answer is judged correct but not in the required format, also correct it.
If the verifier marked your solution as INCORRECT or UNCERTAIN, carefully revise your solution 
according to the feedback.
Always output ONLY your final solution.
    """
    
    DEFAULT_USER_REFINE="""
### Question ###
{question}

### Your Original Solution ###

{prev_answers}

### Verifier's Feedback ###

{feedback}

### Your Revision ###
    """
    
    
    
    def __init__(self, 
                 backend: CanGenerate,
                 system_prompt_initial: Optional[str] = None,
                 user_template_initial: Optional[str]  = None,
                 system_prompt_refine: Optional[str]  = None,
                 user_template_refine: Optional[str]  = None,
                 logger=None):
        """A candidate for generate response

        Args:
            backend (CanGenerate): A backend for generation
            system_prompt_initial (Optional[str], optional): optional system prompt for first-turn generation. Defaults to None.
            user_template_initial (Optional[str], optional): optional user prompt for first-turn generation, should contain format-key {{question}}. Defaults to None.
            system_prompt_refine (Optional[str], optional): optional system prompt for refine generation. Defaults to None.
            user_template_refine (Optional[str], optional): optional user prompt for refine generation, should contain format-key {{question}}, {{prev_answers}},{{feedback}}. Defaults to None.
            logger (_type_, optional): Logger for logging information. Defaults to None.
        """
        self.backend = backend
        self.system_prompt_initial = system_prompt_initial or self.DEFAULT_SYSTEM_INITIAL
        self.user_template_initial = user_template_initial or self.DEFAULT_USER_INITIAL
        self.system_prompt_refine = system_prompt_refine or self.DEFAULT_SYSTEM_REFINE
        self.user_template_refine = user_template_refine or self.DEFAULT_USER_REFINE
        self.logger = logger or get_logger(name=__name__)


    def _build_init_prompts(self, questions: List[str]) -> List[List[Dict[str, str]]]:
        prompts: List[List[Dict[str, str]]] = []
        for q in questions:
            prompts.append([
                {
                    "role": "system",
                    "content": self.system_prompt_initial,
                },
                {
                    "role": "user",
                    "content": self.user_template_initial.format(question=q),
                },
            ])
        return prompts

    def _build_refine_prompts(
        self,
        questions: List[str],
        prev_answers: List[str],
        feedbacks: List[str],
    ) -> List[List[Dict[str, str]]]:
        prompts: List[List[Dict[str, str]]] = []
        for q, prev, fb in zip(questions, prev_answers, feedbacks):
            prompts.append([
                {"role": "system", "content": self.system_prompt_refine},
                {"role": "user", "content": self.user_template_refine.format(
                    question=q,
                    prev_answers=prev,
                    feedback=fb,
                    
                )},
            ])
        return prompts

    def initial_answer(self, questions: List[str], **kwargs) -> List[str]:

        prompts = self._build_init_prompts(questions)
        texts, metas = self.backend.generate(
            prompts,
            extra=None,
            **kwargs,
        )
        return texts

    def refine_answer(
        self,
        questions: List[str],
        prev_answers: List[str],
        feedbacks: List[Dict[str, Any]],
        **kwargs,
    ) -> List[str]:
        prompts = self._build_refine_prompts(questions, prev_answers, feedbacks)
        texts, metas = self.backend.generate(
            prompts,
            extra=None,
            **kwargs,
        )
        return texts




class ForwardVerifierWorker:
    """Evaluate a batch of q-a pairs and returns standard results
        [
        {{
            "label": "correct"|"incorrect"|"uncertain",
            "reason": str, # The last turn message of the verifier, should be stage C for forward agent
            "process": List[Message],
        }},
        ]
        """
        
    REVIEW_PROMPT = """
Now you are required to conduct Stage C (Final Review).

You must carefully examine *all* previous verification steps (Stage A and Stage B).
Your goal in this stage is not only to judge correctness, but also to provide clear, actionable feedback that helps improve the answer.

During Stage C, you must review all earlier reasoning, identify errors, explain them, provide corrections, and finally give a judgment.

1. In <review>...</review>:
   - Summarize the reasoning process across all earlier stages.
   - If previous steps identified incorrect reasoning, or if you find new mistakes at this stage, clearly point out the flawed assumptions or logic.
   - Provide actionable suggestions on how to correct or improve these mistakes.

2. In <answer>...</answer>:
   - Output <answer>true</answer> only if all previous steps were correct and consistent.
   - Output <answer>false</answer> if any error, inconsistency, or unclear reasoning was found.

"""

    def __init__(
        self,
        backend: CanGenerate,
        tool_registry: Optional[ToolRegistry] = None,
        max_rounds: int = 8,
        max_rounds_per_block: int = 6,
        system_prompt: Optional[str] = None,
    ):
        self.backend = backend
        self.registry = tool_registry or ToolRegistry()

        self.agent = MultiturnPlanSubtaskAgent(
            backend=backend,
            max_rounds=max_rounds,
            max_rounds_per_block=max_rounds_per_block,
            tool_registry=self.registry,
            system_prompt=system_prompt,
        )
        
        self.agent.DEFAULT_USER_STAGE_REVIEW_MIDDLE = self.REVIEW_PROMPT

    def evaluate(self, questions: List[str], answers: List[str], **kwargs) -> List[Dict[str, Any]]:
        assert len(questions) == len(answers), "Questions and answers should be in the same size"
        full_msgs, gen_metas = self.agent.generate(questions, answers, **kwargs)

        out: List[Dict[str, Any]] = []
        for msgs in full_msgs:
            label, reason = _parse_verdict_from_msgs(msgs)
            out.append({
                "label": label,
                "reason": reason,
                "process": msgs,
            })
        return out



class BackwardVerifierWorker:
    REVIEW_PROMPT = """
Now you are required to conduct Stage C (Final Review).

You must carefully examine *all* previous verification steps (Stage A and Stage B).
Your goal in this stage is not only to judge correctness, but also to provide clear, actionable feedback that helps improve the answer.

During Stage C, you must review all earlier reasoning, identify errors, explain them, provide corrections, and finally give a judgment.

1. In <review>...</review>:
   - Summarize the reasoning process across all earlier stages.
   - If previous steps identified incorrect reasoning, or if you find new mistakes at this stage, clearly point out the flawed assumptions or logic.
   - Provide actionable suggestions on how to correct or improve these mistakes.

2. In <answer>...</answer>:
   - Output <answer>true</answer> only if all previous steps were correct and consistent.
   - Output <answer>false</answer> if any error, inconsistency, or unclear reasoning was found.

"""
    def __init__(
        self,
        backend: CanGenerate,
        tool_registry: Optional[ToolRegistry] = None,
        max_rounds: int = 8,
        max_rounds_per_block: int = 6,
        system_prompt: Optional[str] = None,
    ):
        self.backend = backend
        self.registry = tool_registry or ToolRegistry()

        self.agent = BackwardVerifyAgent(
            backend=backend,
            max_rounds=max_rounds,
            max_rounds_per_block=max_rounds_per_block,
            tool_registry=self.registry,
            system_prompt=system_prompt,
        )
        self.agent.DEFAULT_USER_STAGE_REVIEW_MIDDLE=self.REVIEW_PROMPT

    def evaluate(self, questions: List[str], answers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Evaluate a batch of q-a pairs and returns standard results
        [
        {{
            "label": "correct"|"incorrect"|"uncertain",
            "reason": str, # The last turn message of the verifier, should be full process for backward agent
            "process": List[Message],
        }},
        ]
        """
        assert len(questions) == len(answers), "Questions and answers should be in the same size"
        full_msgs, gen_metas = self.agent.generate(questions, answers, **kwargs)

        out: List[Dict[str, Any]] = []
        for msgs in full_msgs:
            label, reason = _parse_verdict_from_msgs(msgs)
            out.append({
                "label": label,
                "reason": reason,
                "process": msgs,
            })
        return out
    
class MultiheadVerifierWorker:
    
    CONFLICT_PROMPT_TEMPLATE = """You are a final judge reviewing two verifiers who provide conflicting verdicts on a task.

Your task:
1. Read the original question and the model's candidate answer.
2. Carefully compare the two verification reports (Forward and Backward).
3. Decide a FINAL verdict about whether the candidate answer is correct or not.
4. Provide a clear and detailed summary of your reasoning process.
5. If the candidate answer is incorrect, incomplete, or poorly justified,
   provide concrete guidance on how to modify the answer into a correct and well-justified solution.

You MUST wrap your reasoning and suggestions INSIDE a single <review>...</review> block.
The <review> block MUST clearly include:
- whether the ORIGINAL answer is correct or not and WHY;
- if it is not fully correct, how to revise it into a correct answer.

Example structure:
<review>
[Explain whether the original answer is correct or not, and why.]
[If incorrect or incomplete, explain what is wrong.]
[Provide a corrected answer or detailed instructions to fix the original answer.]
[Produce a fully rewritten and improved version of the candidate’s answer that incorporates all necessary corrections.]
</review>

After the <review> block, on a separate line at the very end of your response, you MUST output exactly one XML tag in one of the following forms:

<answer>true</answer>
<answer>false</answer>

Now begin.

[Question]
{question}

[Candidate Answer]
{answer}

[Forward Verifier]
Label: {lf}
Reason:
{rf}

[Backward Verifier]
Label: {lb}
Reason:
{rb}
"""
    
    def __init__(
        self,
        backend: CanGenerate,
        tool_registry: Optional[ToolRegistry] = None,
        max_rounds: int = 8,
        max_rounds_per_block: int = 6,
        system_prompt: Optional[str] = None,
    ):
        self.backend = backend
        self.forward_worker = ForwardVerifierWorker(
            backend,
            tool_registry,
            max_rounds,
            max_rounds_per_block,
            system_prompt
        )
        
        self.backward_worker = BackwardVerifierWorker(
            backend,
            tool_registry,
            max_rounds,
            max_rounds_per_block,
            system_prompt
        )
    
    def _final_judge_batch(
        self,
        conflict_items: List[Dict[str, Any]],
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        returns：
            [{"label": ..., "reason": ...}, ...]
        """

        if generate_kwargs is None:
            generate_kwargs = {}

        prompts = []
        for item in conflict_items:
            prompts.append(
                self.CONFLICT_PROMPT_TEMPLATE.format(
                    question=item["question"],
                    answer=item["answer"],
                    lf=item["lf"], lb=item["lb"],
                    rf=item["rf"], rb=item["rb"],
                )
            )

        gens, _ = self.backend.generate(prompts, extra=None, **generate_kwargs)

        outs = []
        for gen in gens:
            msgs = Message.from_dicts([{"role": "assistant", "content": gen}])
            label, reason = _parse_verdict_from_msgs(msgs)
            review_tags = find_tags(reason,["review"])
            if review_tags:
                reason = review_tags[-1].body

            outs.append({
                "label": label,
                "reason": reason,
            })

        return outs
        
    @staticmethod
    def _aggregate_label(lf: str, lb: str) -> str:
        if lf == "incorrect" or lb == "incorrect":
            return "incorrect"
        if lf == "correct" and lb == "correct":
            return "correct"
        return "uncertain"

    @staticmethod
    def _aggregate_reason(
        lf: str,
        lb: str,
        rf: str,
        rb: str,
    ) -> str:
        """Combine forward & backward reasons into a single string."""
        return (
            "Forward verifier verdict: {lf}\n"
            "Forward reasoning:\n{rf}\n\n"
            "Backward verifier verdict: {lb}\n"
            "Backward reasoning:\n{rb}"
        ).format(lf=lf, lb=lb, rf=rf, rb=rb)
        
    @staticmethod
    def _is_conflict(lf: str, lb: str) -> bool:
        return lf != lb

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        assert len(questions) == len(answers), "Questions and answers should be in the same size"

        out_forward = self.forward_worker.evaluate(questions, answers, **kwargs)
        out_backward = self.backward_worker.evaluate(questions, answers, **kwargs)
        assert len(out_forward) == len(out_backward), "Forward/Backward outputs size mismatch"

        n = len(questions)
        out: List[Dict[str, Any]] = [None] * n

        conflict_indices: List[int] = []
        conflict_items: List[Dict[str, Any]] = []

        for i, (q, a, f_res, b_res) in enumerate(zip(questions, answers, out_forward, out_backward)):
            lf = f_res["label"]
            lb = b_res["label"]
            rf = f_res["reason"]
            rb = b_res["reason"]

            if not self._is_conflict(lf, lb):
                out[i] = {
                    "label": f_res["label"],
                    "reason": f_res["reason"],
                    "process": f_res.get("process", []),
                }
            else:
                conflict_indices.append(i)
                conflict_items.append(
                    {
                        "question": q,
                        "answer": a,
                        "lf": lf,
                        "lb": lb,
                        "rf": rf,
                        "rb": rb,
                        "f_process": f_res.get("process", []),
                    }
                )

        if conflict_indices:
            final_judge_results = self._final_judge_batch(conflict_items)

            for idx, item, fj in zip(conflict_indices, conflict_items, final_judge_results):
                out[idx] = {
                    "label": fj["label"],
                    "reason": fj["reason"],
                    "process": item["f_process"],
                }

        return out
    
    
class VanillaVerifierWorker:
    
    DEFAULT_SYSTEM="""
You are a Verifier agent responsible for performing a verification of a math problem’s solution.
Your mission is to determine whether the given solution is correct.
    """
    
    DEFAULT_USER="""
Read the following carefully and think critically:

## Original Question
{question}

## Original Solution
{answer}

Your are required to verify the solution carefullt and show your reasoning process.
At the end, provide your final judgment in a single line using:
<answer>true</answer>  — if the reasoning confirms that the conclusion logically hold.
<answer>false</answer> — if there exists any missing justification, invalid reversal, or unsupported dependency.
    """
    
    def __init__(
        self,
        backend: CanGenerate,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        self.backend = backend
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self.user_template = user_template or self.DEFAULT_USER

    
    def _build_prompts(self, questions: List[str], answers: List[str]) -> List[List[Dict[str, str]]]:
        prompts: List[List[Dict[str, str]]] = []
        for q, a in zip(questions, answers):
            prompts.append([
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": self.user_template.format(question=q, answer=a),
                },
            ])
        return prompts

    def evaluate(self, questions: List[str], answers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Evaluate a batch of q-a pairs and returns standard results
        [
        {{
            "label": "correct"|"incorrect"|"uncertain",
            "reason": str, # The last turn message of the verifier, should be full process for backward agent
            "process": List[Message],
        }},
        ]
        """
        assert len(questions) == len(answers), "Questions and answers should be in the same size"
        prompts = self._build_prompts(questions, answers)
        full_msgs = [Message.from_dicts(p) for p in prompts ]
        texts, metas = self.backend.generate(
            prompts,
            extra=None,
            **kwargs,
        )

        out: List[Dict[str, Any]] = []
        for text, msgs in zip(texts, full_msgs):
            msgs.append(Message("assistant", text))
            label, reason = _parse_verdict_from_msgs(msgs)
            out.append({
                "label": label,
                "reason": reason,
                "process": msgs,
            })
        return out

@ray.remote(num_gpus=1, max_restarts=8, max_task_retries=8)
class CandidateActor:
    def __init__(
        self,
        config: Dict[str, Any],
        enable_thinking: bool = False,
    ):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        self.logger = get_logger(config, __name__)

        backend = VllmBackend(config)
        if hasattr(backend, "set_chat_template_defaults"):
            backend.set_chat_template_defaults(enable_thinking=enable_thinking)

        self.worker = CandidateWorker(
            backend=backend,
            logger=self.logger,
        )

    def generate_initial_batch(
        self,
        payload: List[Dict[str, Any]],
        **gen_kwargs,
    ) -> List[Dict[str, Any]]:
        if not payload:
            return []

        questions = [item["question"] for item in payload]
        answers = self.worker.initial_answer(questions, **gen_kwargs)

        out: List[Dict[str, Any]] = []
        for item, ans in zip(payload, answers):
            blk = dict(item)
            blk["answer"] = ans
            out.append(JsonUtil.json_sanitize(blk))
        return out

    def generate_refine_batch(
        self,
        payload: List[Dict[str, Any]],
        **gen_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        payload: [
          {
            "idx": int,
            "question": str,
            "prev_answer": str,
            "feedback": str, 
            ...
          }, ...
        ]
        append "answer" for each element（refined answer）
        """
        if not payload:
            return []

        questions     = [item["question"] for item in payload]
        prev_answers  = [item["prev_answer"] for item in payload]
        feedback_text = [item["feedback"] for item in payload]

        answers = self.worker.refine_answer(
            questions=questions,
            prev_answers=prev_answers,
            feedbacks=feedback_text,
            **gen_kwargs,
        )

        out: List[Dict[str, Any]] = []
        for item, ans in zip(payload, answers):
            blk = dict(item)
            blk["answer"] = ans
            out.append(JsonUtil.json_sanitize(blk))
        return out


@ray.remote(num_gpus=1, max_restarts=8, max_task_retries=8)
class VerifierActor:
    def __init__(
        self,
        config: Dict[str, Any],
        verifier_type: str = "forward",   # "forward" | "backward" | "vanilla"
        enable_thinking: bool = True,
        max_rounds: int = 8,
        max_rounds_per_block: int = 6,
    ):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        self.logger = get_logger(config, __name__)
        self.verifier_type = verifier_type

        backend = VllmBackend(config)
        if hasattr(backend, "set_chat_template_defaults"):
            backend.set_chat_template_defaults(enable_thinking=enable_thinking)

        reg = ToolRegistry()
        py_tool = PythonExecutionToolRay(actor=create_python_actor(time_limit_s=10, mem_limit_mb=16))
        reg.register(py_tool)

        if verifier_type == "forward":
            self.worker = ForwardVerifierWorker(
                backend=backend,
                tool_registry=reg,
                max_rounds=max_rounds,
                max_rounds_per_block=max_rounds_per_block,
            )
        elif verifier_type == "backward":
            self.worker = BackwardVerifierWorker(
                backend=backend,
                tool_registry=reg,
                max_rounds=max_rounds,
                max_rounds_per_block=max_rounds_per_block,
            )
        elif verifier_type == "vanilla":
            self.worker = VanillaVerifierWorker(
                backend=backend,
            )
        elif verifier_type == "multihead":
            self.worker = MultiheadVerifierWorker(
                backend=backend,
                tool_registry=reg,
                max_rounds=max_rounds,
                max_rounds_per_block=max_rounds_per_block,
            )
        else:
            raise ValueError(f"Unknown verifier_type: {verifier_type}")

    def evaluate_batch(
        self,
        payload: List[Dict[str, Any]],
        **gen_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        payload: [
          { "idx": int, "question": str, "answer": str, ... }, ...
        ]
        append "label" / "feedback" for each element
        """
        if not payload:
            return []

        questions = [item["question"] for item in payload]
        answers   = [item["answer"]   for item in payload]

        verdicts = self.worker.evaluate(questions, answers, **gen_kwargs)
        # verdicts[i] = {label, reason, process}

        out: List[Dict[str, Any]] = []
        for item, vd in zip(payload, verdicts):
            blk = dict(item)
            blk["label"] = vd["label"]
            blk["feedback"] = vd["reason"]   
            blk["process"] = vd["process"]
            out.append(JsonUtil.json_sanitize(blk))
        return out