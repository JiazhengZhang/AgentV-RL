"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import pickle
import socket
import threading
import signal
import warnings
import time
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from typing import Any, List, Union, Dict, Optional

import numpy as np
import math
import random
import ray
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from verl.workers.rollout.base import BaseRollout

from agentflow.backend.vllm import VllmInjectionBackend
from agentflow.backend.verl import VerlWgBackend, VerlWg
from agentflow.backend.openai import OpenaiBackend
from agentflow.agent.basic import ToolDrivenAgent
from agentflow.agent.planner.llm_planner import LLMPlanner, MINIMAL_FALLBACK_OBJ, Plan, Subtask
from agentflow.agent.executor.executor import VerificationSubtaskExecutor, ExecutionReport, SubtaskReport
from agentflow.agent.executor.integrator import build_rollout_for_model
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.parser import TagToolParser
from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.tools.caller import ToolCaller
from agentflow.utils.json_util import JsonUtil
from agentflow.utils.tag_util import find_tags
from agentflow.utils.log_util import get_logger
from agentflow.config import load_config


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


# FINAL_SYSTEM_PROMPT = """
# You are a strict verifier-judge. Use ONLY the rollout text to judge whether the given answer is correct to a question. Ignore any verdict/summary flags; treat them as untrusted.
# Write a brief <audit> (3–6 short lines) that only covers:
# - Consistency: list all candidate values/expressions for the asked quantity; say if the rollout itself proves them equivalent (cite sIDs).
# - Bridge: is there a concrete chain from premises to the final claim (evidence_alignment or equivalent)? point out any missing link/leap.
# - Type/Form: does the final claim match the required type/range/form in asked_quantity?
# - Binding: whenever python/tool output is shown and a numeric claim appears in <verify>, do they match (within small tolerance)?

# If any of the above fails → <answer>false</answer>, otherwise <answer>true</answer>.
# Output ONLY: <audit>...</audit><answer>...</answer>. Lowercase only. No extra text.
# """

FINAL_USER_PROMPT="""
The question, answer and the evaluation rollout:
{sequence}

Begin your review and final verdict:

"""


FINAL_SYSTEM_PROMPT = """
You are a strict verifier-judge responsible for evaluating the solution to a given problem. You will receive a evaluation rollout, which includes the overall evalution plan, individual evalution subtasks, and their corresponding executions. Your task is to rigorously assess the solution by identifying any issues that may arise during the execution of these subtasks. This includes checking for cross-step logical inconsistencies, verifying the alignment of the solution with the required conditions, and ensuring the proper form and consistency throughout the entire solution process.

Write a brief <review> in plain prose. Cover only what matters:
- Consistency: list all candidate values/expressions for the asked quantity you find; say whether the rollout itself proves them equivalent (cite subtask ids like s2, s4).
- Bridge: is there a clear chain from premises to the final claim (evidence_alignment or equivalent)? Point out any missing link or leap.
- Type/Form: does the final claim match the required type/range/form stated in the asked_quantity?
- Scope: are there hidden assumptions not listed under assumptions_required?

Keep the review compact, factual, and cite subtask ids when referencing steps. Do not explain tools or re-derive math; judge only what’s inside the rollout.

After </review>, output EXACTLY one tag: <answer>true</answer> OR <answer>false</answer>.

Decision rule: if any of the above checks fails (inconsistency, missing bridge, wrong type/form, hidden assumptions), answer <answer>false</answer>; otherwise answer <answer>true</answer>.

Formatting: output ONLY <review>...</review><answer>...</answer>. Lowercase only. No extra text, no code fences.
"""






DEFAULT_SEQ_TEMPLATE="""
### Problem ###
{problem}

### Solution ###
{solution}
"""

def _clip01(x: Any) -> float:
    try:
        f = float(x)
        f /= 10
    except Exception:
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return max(0.0, min(1.0, f))

def _to_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool) or v is None:
        return v
    s = str(v).strip().lower()
    if s in ("true"):  
        return True
    if s in ("false"):  
        return False
    return None

def _build_plan_analyze_prompt(seq: str, plan: Plan, answer: str) -> List[Dict[str, str]]:

    
    system_prompt = """You are a strict verifier for a plan that decomposes a verification task
(judging whether an assistant's answer to a question is correct) into boolean subtasks.

Your mission:
1) Compare the standard answer with the assistant's answer to determine whether the assistant's answer is also correct as a global ground truth.
2) Assess whether the plan meaningfully helps finish the verification task (That is, the plan can provided sufficient evidence to reach the global ground truth).
3) Provide a scalar score in [0, 10].
4) Give the ground truth (True/False) for each subtask ID exactly as given.

SCORING RUBRIC (0–10):
- Relevance to the main question (0–2): Subtasks align with what the question asks; no off-topic checks.
- Proper use of given conditions (0–2): Subtasks explicitly and correctly leverage the problem’s stated data/assumptions.
- Non-triviality & informativeness (0–2): Subtasks are meaningful (not filler), each contributes unique information.
- Logical sufficiency to reach a verdict (0–3): If all subtask booleans are known, one can determine whether the final answer is correct (i.e., subtasks together are sufficient).
- Clarity & minimal redundancy (0–1): Subtasks are well-scoped, non-duplicative, and avoid unnecessary overlap.

INTERPRETATION RULES:
- For subtasks involving overall judgement, the ground truth of the subtask should allign with the determined global ground truth.
- Judge each subtask by its own claim: return True if the claim is correct given the problem; False otherwise.
- Prefer strict correctness over speculation; if the subtask’s condition cannot be supported by the prompt’s facts, mark False.
- Do not invent new subtask IDs; output booleans for exactly the provided IDs.
- Be consistent: the overall score should reflect the same standards used to judge subtasks.

OUTPUT FORMAT:
<your brief reasoning trace>
```json
{
  "plan_score": <float 0..10>,
  "subtask_gt": {
    "<subtask_id_1>": true | false,
    "<subtask_id_2>": true | false,
    ...
  }
}
```
"""
    
    plan_info = f"problem_restatement: {plan.problem_brief}\n Assumptions: {plan.assumptions_required}\n target_quantity: {plan.asked_quantity}\nSubtasks:\n"
    
    for sub in plan.subtasks:
        plan_info += f"ID: \"{sub.id}\" \ntitle: {sub.title}\nrationale: {sub.rationale}\ncategory: {sub.category}"
        
    
    user_prompt = """The original Question and answer: 
{seq}
The standard answer to the question:
{answer}
The plan of the assistant:
{plan_info}

Please provide your assessment to the plan:
    """.format(
        seq = seq, answer = answer, plan_info = plan_info
    )

    return [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}]

class vLLMAgentWrapper:
    
    def __init__(
        self,
        config,
        wg: VerlWg,
        tokenizer: PreTrainedTokenizer,
        agent_config_path: str = None,
        **kwargs
    ): 
        self.config=config
        if agent_config_path:
            config = load_config(agent_config_path)
        else:
            config = None
        self.logger = get_logger(config, __name__)
        self.wg = wg
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.backend = VerlWgBackend(
            config=config,
            wg=wg,
            tokenizer=tokenizer,
            logger=self.logger,
        )
        self.backend.set_chat_template_defaults(enable_thinking=False)
        tool_registry = ToolRegistry()
        py_tool = PythonExecutionTool()
        tool_registry.register(py_tool)
        self.tool_registry = tool_registry
        self.planner = LLMPlanner(
            self.backend,
        )
        self.executor = VerificationSubtaskExecutor(
            self.backend,
            self.tool_registry,
        )
        
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = self.tokenizer.eos_token_id

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        
        extra_info: np.ndarray = non_tensor_batch.get("extra_info")
        if extra_info is None:
            extra_info=[{} for _ in range(batch_size)]
            self.logger.warning("Extra info of current batch is missing, which may cause unexpected results")
        
        problems: List[str] = [extra_info[i].get("problem","") for i in range(batch_size)]
        solutions: List[str] = [extra_info[i].get("solution","") for i in range(batch_size)]
        
        qa_sequences = [
            DEFAULT_SEQ_TEMPLATE.format(problem=prob, solution=solu)
            for prob, solu in zip(problems, solutions)
        ]
        
        timing_generate = {}
        with simple_timer("agent generation", timing_generate):
        
            plans = self.planner.plan(qa_sequences)
            self.logger.info("Planning finished, executing subtasks.")
            reports = self.executor.execute(sequences=qa_sequences, plans=plans)
            self.logger.info("Subtasks ready, preforming final judge.")
            rollouts = [build_rollout_for_model(sequence=seq, plan=plan, report=report, max_chars_per_subtask=2800) for seq, plan, report in zip(qa_sequences,plans,reports)]

            final_inputs = [[
                {"role":"system","content":FINAL_SYSTEM_PROMPT},
                {"role":"user", "content":FINAL_USER_PROMPT.format(
                    sequence = rollout,  
                )}
            ] for rollout in rollouts]
            
            final_outputs, _ = self.backend.generate(final_inputs)

            
            final_rollouts = [f"{rollout}\n{output}" for rollout, output in zip(rollouts, final_outputs)]

            response_ids_list = []
            for txt in final_rollouts:
                ids = self.tokenizer(txt, add_special_tokens=False).input_ids
                response_ids_list.append(ids)
            self.logger.info("Final judge ready, preparing data proto")
            response = pad_2d_list_to_length(
                response_ids_list,
                self.pad_token_id,
                max_length=self.config.response_length
            ).to(idx.device)

            seq = torch.cat([idx, response], dim=-1)

            if self.config.calculate_log_probs: 
                raise ValueError("Log probs not supported")

            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
            if position_ids.dim() == 3:  # qwen2vl mrope
                delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

            response_position_ids = position_ids[..., -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            response_attention_mask = get_response_mask(
                response_id=response,
                eos_token=eos_token_id,
                dtype=attention_mask.dtype
            )
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

            batch = TensorDict(
                {
                    "prompts": idx,
                    "responses": response,
                    "input_ids": seq,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                batch_size=batch_size,
            )
            gather_output = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        gather_output.meta_info["timing"] = timing_generate
        gather_output = gather_output.to("cpu")
        return gather_output
        

# TODO 需要进一步，plan部分也需要调用api模型直接把gt以及base reward 算好，因为reward wg只返回rm tensor


class vLLMAgentMultiStageWrapper:
    
    def __init__(
        self,
        config,
        wg: VerlWg,
        tokenizer: PreTrainedTokenizer,
        agent_config_path: str = None,
        max_subtasks: int = 8,
        **kwargs
    ): 
        self.config=config
        if agent_config_path:
            ag_config = load_config(agent_config_path)
        else:
            ag_config = None
        self.logger = get_logger(ag_config, __name__)
        self.wg = wg
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.backend = VerlWgBackend(
            config=ag_config,
            wg=wg,
            tokenizer=tokenizer,
            logger=self.logger,
            max_prompt_length=self.config.prompt_length,
        )
        self.remote_backend = OpenaiBackend(config=ag_config, logger=self.logger)
        self.backend.set_chat_template_defaults(enable_thinking=False)
        tool_registry = ToolRegistry()
        py_tool = PythonExecutionTool()
        tool_registry.register(py_tool)
        self.tool_registry = tool_registry
        self.planner = LLMPlanner(
            self.backend,
            max_num_subtasks=max_subtasks,
        )
        self.executor = VerificationSubtaskExecutor(
            self.backend,
            self.tool_registry,
        )
        
    
        
    def prepare_plan_labels(
        self,
        sequences: List[str],
        plans: List[Plan],
        answers: List[str] = None,
        *,
        max_retries: int = 3,
    ) -> List[Dict]:
        """
        返回与 (sequences, plans) 一一对齐：
        {
          "plan_score": float in [0,1],
          "raw_text": str,
          "subtask_gt": { "<subtask_id>": bool|None, ... }
        }
        """
        assert len(sequences) == len(plans)
        N = len(sequences)
        if not answers:
            answers = ["NO STANDARD ANSWER PROVIDED"] * N
        else:
            for i in range(N):
                if not answers[i] or not isinstance(answers[i], str):
                    answers[i] = "NO STANDARD ANSWER PROVIDED"
                
                

        def _parse_one(text: Optional[str], plan: Plan, answer) -> Optional[Dict[str,Any]]:
            if not text:
                return None
            obj = JsonUtil.parse_json(text)  
            if isinstance(obj, list):
                if not obj: 
                    return None
                obj = obj[0]
            if not isinstance(obj, dict):
                return None

            plan_score = _clip01(obj.get("plan_score", 0.0) )


            want_ids = [st.id for st in plan.subtasks]
            gt_in = obj.get("subtask_gt", {})
            gt_out: Dict[str, Optional[bool]] = {}
            if isinstance(gt_in, dict):
                for sid in want_ids:
                    gt_out[sid] = _to_bool(gt_in.get(sid, None))
            else:
                for sid in want_ids:
                    gt_out[sid] = None

            return {"plan_score": plan_score, "raw_text": text, "subtask_gt": gt_out}

        prompts = [_build_plan_analyze_prompt(seq, plan, ans) for seq, plan, ans in zip(sequences, plans, answers)]
        try:
            texts, _ = self.remote_backend.generate(prompts)
        except Exception as e:
            texts = [None] * N
            self.logger.exception(e)

        results: List[Optional[Dict[str,Any]]] = [None] * N
        pending = []
        for i, (t, plan, ans) in enumerate(zip(texts, plans, answers)):
            results[i] = _parse_one(t, plan, ans)
            if results[i] is None:
                pending.append(i)

        attempt = 1
        while pending and attempt < max_retries:
            time.sleep(1)
            retry_prompts = [prompts[i] for i in pending]
            try:
                retry_texts, _ = self.remote_backend.generate(retry_prompts)
            except Exception:
                retry_texts = [None] * len(pending)

            next_pending = []
            for local_idx, i in enumerate(pending):
                parsed = _parse_one(retry_texts[local_idx], plans[i], answer=[i])
                if parsed is None:
                    next_pending.append(i)
                else:
                    results[i] = parsed
            pending = next_pending
            attempt += 1

        final: List[Dict[str,Any]] = []
        for i in range(N):
            if results[i] is not None:
                final.append(results[i])
                continue
            want_ids = [st.id for st in plans[i].subtasks]
            fallback = {
                "plan_score": 0.0,
                "raw_text": (texts[i] or "")[:4000],
                "subtask_gt": {sid: None for sid in want_ids},
            }
            final.append(fallback)
        return final

                
                    
                
            
        
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        
        meta_info = prompts.meta_info
        stage = meta_info.get("stage","no_stage")
        output: DataProto = None
        ids: torch.Tensor = prompts.batch["input_ids"]
        batch_size = ids.size(0)
        
        timing_generate = {}
        with simple_timer("agent generation", timing_generate):
        
            if stage == "plan":
                output = self._generate_stage_plan(prompts, **kwargs)
            elif stage == "subtask":
                output = self._generate_stage_subtask(prompts, **kwargs)
            elif stage == "review":
                output = self._generate_stage_review(prompts, **kwargs)
            else:
                output = self._generate_no_stage(prompts, **kwargs)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        
        return output
    
    def _prepare_result_proto(self, original_proto: DataProto, gen_results: List[str] | List[List[int]], response_mask: Optional[List[List[int]]] = None,**kwargs) -> DataProto:
        meta_info = original_proto.meta_info
        idx: torch.Tensor = original_proto.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask: torch.Tensor = original_proto.batch["attention_mask"]
        position_ids: torch.Tensor = original_proto.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = self.tokenizer.eos_token_id
        
        non_tensor_batch = original_proto.non_tensor_batch

        batch_size = idx.size(0)
        
        response_ids_list = []
        
        for cont in gen_results:
            if isinstance(cont, str):
                ids = self.tokenizer(cont, add_special_tokens=False).input_ids
                response_ids_list.append(ids)
            else:
                response_ids_list.append(cont)

        response = pad_2d_list_to_length(
            response_ids_list,
            self.pad_token_id,
            max_length=self.config.response_length
        ).to(idx.device)

        seq = torch.cat([idx, response], dim=-1)

        if self.config.calculate_log_probs: 
            raise ValueError("Log probs not supported")

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response,
            eos_token=eos_token_id,
            dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        has_resp_mask = False
        if response_mask is not None:
            resp_mask_tensor = pad_2d_list_to_length(
                response_mask,
                self.pad_token_id,
                max_length=self.config.response_length
            ).to(idx.device)
            has_resp_mask = True
        
        if has_resp_mask:
            batch = TensorDict(
                {
                    "prompts": idx,
                    "responses": response,
                    "input_ids": seq,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "response_mask": resp_mask_tensor,
                },
                batch_size=batch_size,
            )
        else:
            batch = TensorDict(
                {
                    "prompts": idx,
                    "responses": response,
                    "input_ids": seq,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                batch_size=batch_size,
            )

        for key, value in non_tensor_batch.items():
            if not isinstance(value, np.ndarray):
                try:
                    non_tensor_batch[key] = np.array(value, dtype=object)
                except Exception as e:
                    self.logger.critical(f"Could not convert non_tensor_batch['{key}'] to numpy array. Error: {e}")
        
        gather_output = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=original_proto.meta_info)
        return gather_output
        
    def _generate_stage_plan(self, prompts: DataProto, **kwargs):
        
        meta_info = prompts.meta_info
        idx: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        
        extra_info: np.ndarray = non_tensor_batch.get("extra_info")
        if extra_info is None:
            raise ValueError("Extra info must be provided for qa info")
            extra_info=[{} for _ in range(batch_size)]
            self.logger.warning("Extra info of current batch is missing, which may cause unexpected results")
        
        problems: List[str] = [extra_info[i].get("problem","") for i in range(batch_size)]
        solutions: List[str] = [extra_info[i].get("solution","") for i in range(batch_size)]
        
        qa_sequences = [
            DEFAULT_SEQ_TEMPLATE.format(problem=prob, solution=solu)
            for prob, solu in zip(problems, solutions)
        ]
        
        input_msgs = [self.planner._build_prompt(sequence=seq) for seq in qa_sequences]
        
        processed_input = self.backend.apply_chat_template(input_msgs)
        process_proto = self.backend.prepare_dataproto(processed_input)
        process_proto.non_tensor_batch = prompts.non_tensor_batch
        process_proto.meta_info.update(meta_info)
        
        texts, gen_metas = self.backend.generate(input_msgs, sleep_after_inference=False)
        
        plans = []

        for i, raw in enumerate(texts):
            try:
                obj = self.planner._parse_plan_obj(raw)
                plans.append(self.planner._coerce_to_plan(obj))
            except Exception:
                plans.append(self.planner._coerce_to_plan(MINIMAL_FALLBACK_OBJ))
        
        extra_infos = prompts.non_tensor_batch["extra_info"]
        answers = []
        for extra in extra_infos:
            eval = extra.get("eval", {})
            parsed_gt = eval.get("parsed_gt", None)
            answers.append(parsed_gt)
        labels = self.prepare_plan_labels(qa_sequences, plans, answers=answers)
        
        mock_msgs = [[{"role":"user","content":"How are you"}] for _ in range(batch_size)]
        
        _, _ = self.backend.generate(mock_msgs, sleep_after_inference=True)
        
        subtask_labels = np.empty(batch_size, dtype=object)
        dynamic_info = np.empty(batch_size, dtype=object)
        
        for idx, label in enumerate(labels):
            subtask_labels[idx] = label["subtask_gt"]
            dynamic_info[idx] = label.copy()
            dynamic_info[idx]["stage"] = "plan"
        
        process_proto.non_tensor_batch["plans"] = np.array([p for p in plans], dtype=object)
        process_proto.non_tensor_batch["subtask_labels"] = subtask_labels
        process_proto.non_tensor_batch["dynamic_info"] = dynamic_info
        
        final_proto = self._prepare_result_proto(process_proto, texts)
        return final_proto
        
    
    def _generate_stage_subtask(self, prompts: DataProto, **kwargs):
        meta_info = prompts.meta_info

        idx: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)

        batch_size = idx.size(0)
        
        non_tensor_batch = prompts.non_tensor_batch

        plans: np.ndarray[Plan] = non_tensor_batch["plans"]
        subtasks: np.ndarray[Subtask] = non_tensor_batch["subtasks"]
        subtask_gt: np.ndarray[bool] = non_tensor_batch["subtask_gt"]
        subtask_ids: np.ndarray[str] = non_tensor_batch["subtask_ids"]
        assert subtasks.shape[0] == subtask_ids.shape[0]
        for i in range(batch_size):
            sub = subtasks[i]
            sub_id = subtask_ids[i]
            assert isinstance(sub, Subtask)
            assert sub.id == sub_id
        
        # TODO Check subtask is ok
        
        extra_info: np.ndarray = non_tensor_batch.get("extra_info")
        if extra_info is None:
            raise ValueError("Extra info must be provided for qa info")
        
        problems: List[str] = [extra_info[i].get("problem","") for i in range(batch_size)]
        solutions: List[str] = [extra_info[i].get("solution","") for i in range(batch_size)]
        
        qa_sequences = [
            DEFAULT_SEQ_TEMPLATE.format(problem=prob, solution=solu)
            for prob, solu in zip(problems, solutions)
        ]
        
        input_msgs = [self.executor._format_subtask_prompt(seq, p, s) for seq, p, s in zip(qa_sequences, plans, subtasks)]
        
        processed_input = self.backend.apply_chat_template(input_msgs)
        process_proto = self.backend.prepare_dataproto(processed_input)
        process_proto.non_tensor_batch = prompts.non_tensor_batch
        process_proto.meta_info.update(meta_info)
        
        reports = self.executor.execute_one(qa_sequences, plans, subtasks)
        
        resp_mask_ids = [[] for _ in range(len(reports))]
        resp_ids = [[] for _ in range(len(reports))]
        for idx, rep in enumerate(reports):
            msgs = rep.round_messages

            for msg in msgs:
                if msg.role == "system" or msg.role == "user":
                    continue
                elif msg.role == "assistant":
                    txt_ids = self.tokenizer(msg.content, add_special_tokens=False).input_ids
                    mask_ids = [1 for _ in range(len(txt_ids))]
                elif msg.role == "tool":
                    txt_ids = self.tokenizer(msg.content, add_special_tokens=False).input_ids
                    mask_ids = [0 for _ in range(len(txt_ids))]
                else:
                    continue
                resp_ids[idx].extend(txt_ids)
                resp_mask_ids[idx].extend(mask_ids)
                
        subtask_ids = np.array([s.subtask_id for s in reports], dtype=str)
        dynamic_info = np.empty(batch_size, dtype=object)
        for i in range(batch_size):
            sid = subtask_ids[i]
            plan = plans[i]
            subtask = subtasks[i]
            report = reports[i]
            gt = subtask_gt[i]
            
            dynamic_info[i] = {
                "subtask_id": sid,
                "plan": plan,
                "subtask": subtask,
                "report": report,
                "subtask_gt": gt,
                "stage":"subtask",
            }
        
        process_proto.non_tensor_batch["reports"] = reports
        process_proto.non_tensor_batch["subtask_ids"] = subtask_ids
        process_proto.non_tensor_batch["dynamic_info"] = dynamic_info
        
        final_proto = self._prepare_result_proto(process_proto, resp_ids, resp_mask_ids)
        return final_proto
    
    def _generate_stage_review(self, prompts: DataProto, **kwargs):
        meta_info = prompts.meta_info
        idx: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = self.tokenizer.eos_token_id

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        
        plans: np.ndarray[Plan] = non_tensor_batch["plans"]
        reports: np.ndarray[ExecutionReport] = non_tensor_batch["execution_reports"]
        
        extra_info: np.ndarray = non_tensor_batch.get("extra_info")
        if extra_info is None:
            raise ValueError("Extra info must be provided for qa info")
            extra_info=[{} for _ in range(batch_size)]
            self.logger.warning("Extra info of current batch is missing, which may cause unexpected results")
        
        problems: List[str] = [extra_info[i].get("problem","") for i in range(batch_size)]
        solutions: List[str] = [extra_info[i].get("solution","") for i in range(batch_size)]
        
        qa_sequences = [
            DEFAULT_SEQ_TEMPLATE.format(problem=prob, solution=solu)
            for prob, solu in zip(problems, solutions)
        ]
        
        plan_subtask_rollouts = [build_rollout_for_model(sequence=seq, plan=plan, report=report,max_chars_per_subtask=2048) 
                                for seq, plan, report in zip(qa_sequences, plans, reports)]
        
        input_msgs = [[
            {"role":"system","content":FINAL_SYSTEM_PROMPT},
            {"role":"user", "content":FINAL_USER_PROMPT.format(
                sequence = rollout,  
            )}
        ] for rollout in plan_subtask_rollouts]
        
        processed_input = self.backend.apply_chat_template(input_msgs)
        process_proto = self.backend.prepare_dataproto(processed_input)
        process_proto.non_tensor_batch = prompts.non_tensor_batch
        process_proto.meta_info.update(meta_info)
        
        texts, gen_metas = self.backend.generate(input_msgs)
        
                
        dynamic_info = np.empty(batch_size, dtype=object)
        for i in range(batch_size):
            dynamic_info[i] = {
                "stage":"review",
            }
        process_proto.non_tensor_batch["dynamic_info"]=dynamic_info
        final_proto = self._prepare_result_proto(process_proto, texts)

        return final_proto
        
    
    def _generate_no_stage(self, prompts: DataProto, **kwargs):
        idx: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = self.tokenizer.eos_token_id

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        
        extra_info: np.ndarray = non_tensor_batch.get("extra_info")
        if extra_info is None:
            extra_info=[{} for _ in range(batch_size)]
            self.logger.warning("Extra info of current batch is missing, which may cause unexpected results")
        
        problems: List[str] = [extra_info[i].get("problem","") for i in range(batch_size)]
        solutions: List[str] = [extra_info[i].get("solution","") for i in range(batch_size)]
        
        qa_sequences = [
            DEFAULT_SEQ_TEMPLATE.format(problem=prob, solution=solu)
            for prob, solu in zip(problems, solutions)
        ]
        
        kwargs_middle = {
            "sleep_after_inference":False,
        }
        
        kwargs_after = {
            "sleep_after_inference":True,
        }
        

        
        plans = self.planner.plan(qa_sequences, None, **kwargs_middle)
        self.logger.info("Start Planning")
        self.logger.info("Planning finished, executing subtasks.")

        self.logger.info("Start Resports.")
        reports = self.executor.execute(sequences=qa_sequences, plans=plans, **kwargs_middle)

        self.logger.info("Subtasks ready, preforming final judge.")


        rollouts = [build_rollout_for_model(sequence=seq, plan=plan, report=report, max_chars_per_subtask=2800) for seq, plan, report in zip(qa_sequences,plans,reports)]

        final_inputs = [[
            {"role":"system","content":FINAL_SYSTEM_PROMPT},
            {"role":"user", "content":FINAL_USER_PROMPT.format(
                sequence = rollout,  
            )}
        ] for rollout in rollouts]
        
        final_outputs, _ = self.backend.generate(final_inputs, None, **kwargs_after)

        
        final_rollouts = [f"{rollout}\n{output}" for rollout, output in zip(rollouts, final_outputs)]

        response_ids_list = []
        for txt in final_rollouts:
            ids = self.tokenizer(txt, add_special_tokens=False).input_ids
            response_ids_list.append(ids)
        self.logger.info("Final judge ready, preparing data proto")
        response = pad_2d_list_to_length(
            response_ids_list,
            self.pad_token_id,
            max_length=self.config.response_length
        ).to(idx.device)

        seq = torch.cat([idx, response], dim=-1)

        if self.config.calculate_log_probs: 
            raise ValueError("Log probs not supported")

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response,
            eos_token=eos_token_id,
            dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )


        non_tensor_batch["plans"] = [plan.to_dict() for plan in plans]
        non_tensor_batch["subtask_executions"] = [report.to_dict() for report in reports]

        for key, value in non_tensor_batch.items():
            if not isinstance(value, np.ndarray):
                try:
                    non_tensor_batch[key] = np.array(value)
                except Exception as e:
                    print(f"Could not convert non_tensor_batch['{key}'] to numpy array. Error: {e}")

        gather_output = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

        return gather_output
        


class vLLMAgentRollout(BaseRollout):
    def __init__(self, 
                 model_path: str, 
                 config: DictConfig, 
                 tokenizer, 
                 model_hf_config, 
                 agent_config_path: str = None,
                 do_bon = False,
                 **kwargs):
        """A vLLM agent rollout which generates full rollout in a single round . It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        
        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            # enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )
        
        # Tool configuration 
        if agent_config_path is None:
            agent_config_path = "/root/workspace/agent-rm/Agentic-Reward/config/default.yaml"
        elif (not isinstance(agent_config_path, str)):
            raise ValueError(f"Tool config path must be string, got {type(agent_config_path)}")
        tool_config = load_config(agent_config_path)
        self.tool_config = tool_config
        
        self.tokenizer = tokenizer
        
        
        

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        # print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        
        self.sampling_params.detokenize = True
        self.sampling_params.stop = self.tool_config["backend"].get("vllm", {}).get("stop_tokens",None)
        self.sampling_params.include_stop_str_in_output=True
        
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = self.inference_engine.get_tokenizer().eos_token_id
        
        if self.config.calculate_log_probs: 
            raise ValueError("Log probs not supported")
        
        
        # Adapter engine for our tool pipeline
        # We need another engine for tool-call with different sampling params
        self.agent_backend = VllmInjectionBackend(
            config=self.tool_config,
            llm = self.inference_engine,
            sampling_params=self.sampling_params,
        )
        
        self.agent_backend.set_chat_template_defaults(enable_thinking=False)
        
        tool_registry = ToolRegistry()
        py_tool = PythonExecutionTool()
        tool_registry.register(py_tool)
        self.tool_registry = tool_registry
        self.planner = LLMPlanner(
            self.agent_backend,
        )
        self.executor = VerificationSubtaskExecutor(
            self.agent_backend,
            self.tool_registry,
        )

        self.tool_logger = get_logger(tool_config, __name__)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)
            
    @GPUMemoryLogger(role="vllm agent rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        
        extra_info: np.ndarray = non_tensor_batch.get("extra_info")
        if extra_info is None:
            extra_info=[{} for _ in range(batch_size)]
            self.tool_logger.warning("Extra info of current batch is missing, which may cause unexpected results")
        
        problems: List[str] = [extra_info[i].get("problem","") for i in range(batch_size)]
        solutions: List[str] = [extra_info[i].get("solution","") for i in range(batch_size)]
        
        qa_sequences = [
            DEFAULT_SEQ_TEMPLATE.format(problem=prob, solution=solu)
            for prob, solu in zip(problems, solutions)
        ]
        
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        kwargs_middle = {
            "sleep_after_inference":False,
        }
        
        kwargs_after = {
            "sleep_after_inference":True,
        }
    
        plans = self.planner.plan(qa_sequences, None, **kwargs_middle)
        self.tool_logger.debug("Planning finished, executing subtasks.")
        reports = self.executor.execute(sequences=qa_sequences, plans=plans, **kwargs_middle)
        self.tool_logger.debug("Subtasks ready, preforming final judge.")
        rollouts = [build_rollout_for_model(sequence=seq, plan=plan, report=report, max_chars_per_subtask=2800) for seq, plan, report in zip(qa_sequences,plans,reports)]

        final_inputs = [[
            {"role":"system","content":FINAL_SYSTEM_PROMPT},
            {"role":"user", "content":FINAL_USER_PROMPT.format(
                sequence = rollout,  
            )}
        ] for rollout in rollouts]
        
        final_outputs, _ = self.agent_backend.generate(final_inputs, **kwargs_after)

        
        final_rollouts = [f"{rollout}\n{output}" for rollout, output in zip(rollouts, final_outputs)]

        response_ids_list = []
        for txt in final_rollouts:
            ids = self.tokenizer(txt, add_special_tokens=False).input_ids
            response_ids_list.append(ids)
        self.tool_logger.debug("Final judge ready, preparing data proto")
        response = pad_2d_list_to_length(
            response_ids_list,
            self.pad_token_id,
            max_length=self.config.response_length
        ).to(idx.device)

        seq = torch.cat([idx, response], dim=-1)

        if self.config.calculate_log_probs: 
            raise ValueError("Log probs not supported")

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response,
            eos_token=eos_token_id,
            dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
                
            
    @GPUMemoryLogger(role="vllm tool rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences_ref(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts 
        by runing multi-turn tool-augmented inference and generate the response sequences

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.
            - rollout_log_probs: [bsz, response_length], log probs for computation

            Multi-turn conversations
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        raise NotImplementedError("")
        # Basic tensors for generation
        input_ids: torch.LongTensor = prompts.batch["input_ids"]       # (bs, prompt_len)
        attention_mask: torch.LongTensor = prompts.batch["attention_mask"]
        position_ids: torch.LongTensor  = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        device = input_ids.device
        batch_size = input_ids.size(0)
        prompt_length = input_ids.size(1)
        
        need_lp = bool(self.config.calculate_log_probs)
        
        # Restore raw_prompt_ids
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, input_ids[i]) for i in range(batch_size)],
                dtype=object
            )
        base_prompt_ids: list[list[int]] = [
            ids.tolist() if isinstance(ids, np.ndarray) else list(ids)
            for ids in non_tensor_batch.pop("raw_prompt_ids")
        ]

        # Set multi_modal_data
        multi_modal_data = None
        if "multi_modal_data" in non_tensor_batch.keys():
            multi_modal_data = list(non_tensor_batch.pop("multi_modal_data"))
        
        # Prepare ground truth for sampling
        if "reward_model" in non_tensor_batch:
            reward_infos = non_tensor_batch.get("reward_model")
            assert isinstance(reward_infos, np.ndarray)
        
        if "data_source" in non_tensor_batch.keys():
            data_sources = non_tensor_batch.get("data_source")
            assert isinstance(data_sources, np.ndarray)
        
        extra_info: Dict = non_tensor_batch.get("extra_info")
        if extra_info is None:
            warnings.warn("Detect invalid extra info, skipping bon sampling")
            bon_sampling = False
            user_messages = None
            system_prompts = None
        else:
            bon_sampling = True
            user_messages: List[Dict[str,str]] = [extra_info[i].get("user_message") for i in range(batch_size)]
            system_prompts = [_parse_prompt(source) for source in data_sources]
            bon_prompt_ids: list[list[int]] = []
        

        # Update sampling params and set n=1 since we only need one sample
        do_sample   = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        sp_overrides = {}
        if not do_sample:
            sp_overrides = dict(best_of=1, top_p=1.0, top_k=-1, min_p=0.0, temperature=0.0, n=1)
        elif is_validate:
            sp_overrides = dict(
                top_k=self.config.val_kwargs.top_k,
                top_p=self.config.val_kwargs.top_p,
                temperature=self.config.val_kwargs.temperature,
                n=1
            )

        # Lora is the same as original verl implementation
        lora_requests = None
        if getattr(self, "lora_kwargs", None):
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path="/simon-stub-path")] * batch_size

        
        
        
        
        accepted = [False] * batch_size
        
        # Response max length
        resp_max_len = int(self.config.response_length)
        max_context = int(getattr(self.sampling_params, "max_model_len", 4096)) - 256
        
        # Only model generated outputs and system appended tool results
        resp_ids : List[List[int]] = [[] for _ in range(batch_size)]       
        # We mask all tool results
        resp_mask : List[List[int]] = [[] for _ in range(batch_size)]    
        # log probs   
        lp_history : Union[List[List[float]], None] = ([[] for _ in range(batch_size)] if self.config.calculate_log_probs else None)
        # BON sampling
        do_bon = bon_sampling and self.do_bon
        
        do_bon = False
        
        if is_validate:
            do_bon = False
        
        bon_accuracies = [0 for _ in range(batch_size)]
        if do_bon:
            self.tool_logger.debug("Doing BON sampling")
            for i in range(batch_size):

                msg = [
                    {"role": "system", "content": system_prompts[i]},
                    user_messages[i] if isinstance(user_messages[i], dict) else {"role": "user", "content": str(user_messages[i])},
                ]
                bon_text = self.tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
                bon_ids  = self.tokenizer.encode(bon_text, add_special_tokens=False)
                bon_prompt_ids.append(bon_ids)
            
            bon_inputs = []
            for i in range(batch_size):
                entry = {"prompt_token_ids": bon_prompt_ids[i]}
                if "multi_modal_data" in non_tensor_batch:
                    entry["multi_modal_data"] = non_tensor_batch["multi_modal_data"][i]
                bon_inputs.append(entry)

            lora_requests = None
            if getattr(self, "lora_kwargs", None):
                lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
                if len(lora_int_ids) > 0:
                    lora_id = lora_int_ids[0]
                    lora_requests = [LoRARequest(lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path="/simon-stub-path")] * batch_size

            outs = self.inference_engine.generate(
                prompts=bon_inputs,
                sampling_params=self.bon_sampling_params,  # n>1
                lora_request=lora_requests,
                use_tqdm=False,
            )

            n = int(getattr(self.bon_sampling_params, "n", 4))
            thr = float(getattr(self.config, "bon_accept_threshold", 0.50))
            
            for i, out in enumerate(outs):
                ground_truth = reward_infos[i]["ground_truth"]
                data_source = data_sources[i]
                

                correct = 0
                chosen_text = None
                chosen_ids = None
                chosen_lp   = None  

                for samp in out.outputs:
                    text = samp.text
                    tags = find_tags(text, ["answer"])
                    ans  = tags[-1].body if tags else "Invalid"
                    ok = (_parse_choice(data_source=data_source, content=ans) ==
                        _parse_choice(data_source=data_source, content=ground_truth))
                    if ok:
                        correct += 1
                        if chosen_text is None:
                            chosen_text = text
                            chosen_ids = samp.token_ids
                            if need_lp and samp.logprobs is not None:
                                
                                gen_ids = samp.token_ids
                                chosen_lp = [lpdict[gen_ids[t]].logprob for t, lpdict in enumerate(samp.logprobs)]

                acc = correct / max(n, 1)
                bon_accuracies[i] = acc
                if acc >= thr and chosen_text is not None:
                    
                    tok = chosen_ids
                    resp_ids[i] = tok
                    resp_mask[i] = [1] * len(tok)                 
                    if need_lp:
                        
                        if chosen_lp is not None:
                            lp_history[i] = chosen_lp
                        else:
                            lp_history[i] = [-1] * len(tok)
                        
                    accepted[i] = True
            self.tool_logger.debug(f"Bon acc {bon_accuracies}")
       
        
        # Multi-turn tool-call procedure
        
        # Current context(prompt + model generated outputs and system appended tool results)
        curr_ids : List[List[int]] = [p.copy() for p in base_prompt_ids]   
        
        # The full text that has been decoded for tool-pipeline interface
        text_tails : List[str] = [""] * batch_size    
        # Tool quota history                       
        round_cnts : List[dict] = [{} for _ in range(batch_size)]     

        # Global max rounds
        max_rounds = int(self.tool_config.get("runner",{}).get("max_rounds", 12))


        # Active indices for multi-turn inference
        unfinished = list(i for i in range(batch_size) if not accepted[i])
        self.tool_logger.info(f"Vllm tool engine starting inference for samples {unfinished}")
        with self.update_sampling_params(**sp_overrides):
            for round_index in range(max_rounds):
                if not unfinished:
                    break

                # Vllm inputs for current turn, similar to original verl implementation
                vllm_inputs = []
                for i in range(len(curr_ids)):
                    if i in unfinished:
                        entry = {"prompt_token_ids": curr_ids[i]}
                        if multi_modal_data is not None:
                            entry["multi_modal_data"] = multi_modal_data[i]
                        vllm_inputs.append(entry)

                # Call vllm engine for outputs
                try:
                    self.tool_logger.info(f"Round {round_index} calling vllm engine")
                    outs = self.inference_engine.generate(
                        prompts=vllm_inputs,
                        sampling_params=self.sampling_params,
                        lora_request=lora_requests,
                        use_tqdm=False,
                    )
                except Exception as e:
                    self.tool_logger.error(f"Vllm inference error {e}.Current test tails: {text_tails}")
                    # Exit the batch
                    raise e

                    
                # Collect generated contents and update session variables
                gen_texts: dict[int, str] = {} # Newly generated text for tool call
                for i, out in zip(unfinished,outs):
                    # We assume that we only sample once for simplicity.Thus we use out.outputs[0]
                    samp = out.outputs[0]
                    gen_ids = samp.token_ids
                    curr_ids[i].extend(gen_ids) # New generated ids for curr and resp
                    resp_ids[i].extend(gen_ids) 
                    resp_mask[i].extend([1] * len(gen_ids)) 
                    if lp_history is not None:
                        # Get the logprobs according to the original implementation
                        curr_log_prob = []
                        for log_prob_idx, logprob in enumerate(samp.logprobs):
                            curr_log_prob.append(logprob[gen_ids[log_prob_idx]].logprob)
                        lp_history[i].extend(curr_log_prob)
                    gen_texts[i] = self.tokenizer.decode(gen_ids, skip_special_tokens=False)

                # Build prompt states for tool pipeline interface
                states: List[PromptState] = []
                for i in unfinished:
                    st = PromptState.from_custom(
                        head   = text_tails[i] + gen_texts[i], 
                        text   = text_tails[i] + gen_texts[i], # All generted text 
                        current= gen_texts[i],  # text for tag match inside tool-pipeline
                        round_cnt= round_cnts[i], # quota history
                    )
                    states.append(st)

                # Call tool pipeline
                # metas: List[meta], each meta is a dict of items, which differs by tool types: 
                # "raw_search_result": search_result_str for search calls
                # "raw_output": vllm.RequestOutput if enable summarize for search calls
                done_flags, metas = self.tool_pipeline.batch_feed(states)
                # metas =  [{} for _ in unfinished]
                # done_flags = [False for _ in unfinished]
                assert all([isinstance(flag, bool) for flag in done_flags])
                assert all([isinstance(meta, dict) for meta in metas])

                # Encode tool generated texts, write back and mask then
                for st_idx, i in enumerate(unfinished):
                    meta = metas[st_idx]
                    tool_type = meta.get("tool_type")
                    enable_summarize = meta.get("enable_summarize",False)
                    mask_id = 1 if (tool_type=="search" and enable_summarize) else 0 # TODO mask是否一定
                    mask_id = 0 # Temp cancel mask
                    old_len = len(text_tails[i] + gen_texts[i])
                    new_text = states[st_idx].text
                    delta = new_text[old_len:]  # Tool append contents
                    if delta:
                        delta_ids = self.tokenizer(
                            delta, add_special_tokens=False, return_tensors=None
                        )["input_ids"]
                        # adapter for hf tokenizer
                        if isinstance(delta_ids[0], list):
                            delta_ids = delta_ids[0]
                            
                        curr_ids[i].extend(delta_ids)
                        resp_ids[i].extend(delta_ids)
                        resp_mask[i].extend([mask_id] * len(delta_ids)) # Mask tool-call ids
                        if lp_history is not None:
                            lp_history[i].extend([-1.0] * len(delta_ids))  # Mask tool-call ids log probs
                    # Update text tails and round cnt
                    text_tails[i] = new_text
                    round_cnts[i] = states[st_idx].round_cnt

                # Prepare the contents for next turn generation
                new_unfinished = []
                for flag, i in zip(done_flags, unfinished):
                    if len(resp_ids[i]) >= resp_max_len:
                        # truncate sequences that are longer than resp_max_len
                        resp_ids[i] = resp_ids[i][:resp_max_len]
                        resp_mask[i] = resp_mask[i][:resp_max_len]
                        if lp_history is not None:
                            lp_history[i] = lp_history[i][:resp_max_len]
                        continue
                    if len(curr_ids[i])>=max_context:
                        self.tool_logger.warning(
                            "Detect overlong context in generation, forcing aborting"
                        )
                        continue
                    if not flag:
                        new_unfinished.append(i)
                unfinished = new_unfinished

        self.tool_logger.info("Multiturn generation finished, preparing data proto")

        final_prompt_ids: list[list[int]] = []
        
        for i in range(batch_size):
            # final_prompt_ids.append(input_ids[i].tolist())
            if accepted[i]:
                final_prompt_ids.append(bon_prompt_ids[i][:prompt_length])        
            else:
                final_prompt_ids.append(input_ids[i].tolist())
                
        for i in range(batch_size):
            if len(resp_ids[i]) > resp_max_len:
                resp_ids[i]  = resp_ids[i][:resp_max_len]
                resp_mask[i] = resp_mask[i][:resp_max_len]
                if need_lp and lp_history[i] is not None:
                    lp_history[i] = lp_history[i][:resp_max_len]
       
        # Pad to fixed length
        prompts_tensor = pad_2d_list_to_length(final_prompt_ids, self.pad_token_id, max_length=prompt_length).to(device)         
        responses = pad_2d_list_to_length(resp_ids, self.pad_token_id, max_length=resp_max_len).to(device)
        response_mask = pad_2d_list_to_length(resp_mask, 0, max_length=resp_max_len).to(device)  # 1=生成, 0=工具/填充

        # (prompt | response)
        seq = torch.cat([prompts_tensor, responses], dim=-1)

        
        # attention_mask, same as the original verl implementation
        prompt_attention_mask = (prompts_tensor != self.pad_token_id).to(prompts_tensor.dtype)
        response_attention_mask = get_response_mask(
            response_id=responses, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        
        # position_ids, same as the original verl implementation
        prompt_position_ids = compute_position_id_with_mask(prompt_attention_mask).to(position_ids.device)
        response_length = responses.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
        response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)

        
        # Pack and return, here add response_mask to mask tool-results
        batch = TensorDict(
            {
                "prompts": prompts_tensor,
                "responses": responses,
                "response_mask": response_mask,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                
            },
            batch_size=batch_size,
        )

        if self.config.calculate_log_probs and lp_history is not None:
            rollout_log_probs = pad_2d_list_to_length(lp_history, -1.0, max_length=resp_max_len).to(device).to(torch.float32)
            batch["rollout_log_probs"] = rollout_log_probs
        
        # Write tool counter to dynamic_info
        # Important: the data consistency of DataProto class
        
        
        dynamic_info = np.empty(batch_size, dtype=object)
        for i in range(batch_size):
            dynamic_info[i] = {}
            
        # Length of round_cnts is batch size
        for i in range(batch_size):
            dynamic_info[i]["tool_counter"]=round_cnts[i]
            dynamic_info[i]["acc"]=bon_accuracies[i]
        non_tensor_batch["dynamic_info"]=dynamic_info


        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    

# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)