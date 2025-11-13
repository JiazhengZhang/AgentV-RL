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

from agentflow.backend.verl import VerlWgBackend, VerlWg
from agentflow.common.messages import Message
from agentflow.agent.basic import ToolDrivenAgent
from agentflow.agent.plan import MultiturnPlanSubtaskAgent, BackwardVerifyAgent
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

class vllmMultiturnWrapper:
    def __init__(
        self,
        config,
        wg: VerlWg,
        tokenizer: PreTrainedTokenizer,
        agent_config_path: str = None,
        enable_thinking: bool = True,
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
            max_prompt_length=30000,
        )
        self.backend.set_chat_template_defaults(enable_thinking=enable_thinking)
        tool_registry = ToolRegistry()
        py_tool = PythonExecutionTool()
        tool_registry.register(py_tool)
        self.tool_registry = tool_registry

        self.forward_agent = MultiturnPlanSubtaskAgent(
            self.backend,
            max_rounds_per_block=3,
            tool_registry=self.tool_registry,
        )
        
        self.backward_agent = BackwardVerifyAgent(
            self.backend,
            max_rounds_per_block=3,
            tool_registry=self.tool_registry,
        )
        
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
                0,
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
        
        gather_output = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
        return gather_output
        
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
        
        kwargs_middle = {
            "sleep_after_inference":False,
        }
        
        timing_generate = {}
        with simple_timer("agent generation", timing_generate):
        
            msgs, metas = self.forward_agent.generate(problems, solutions, **kwargs_middle)
            input_msgs_list = [[] for _ in range(batch_size)]
            response_ids_list = [[] for _ in range(batch_size)]
            response_mask_list = [[] for _ in range(batch_size)]
            for indice, msg_list in enumerate(msgs):
                assert msg_list[0].role == "system"
                assert msg_list[1].role == "user"
                input_msgs = msg_list[:2]
                input_msgs_std = Message.to_dicts(input_msgs)
                input_msgs_list[indice] = input_msgs_std
                rollout_msgs = msg_list[2:]
                for msg in rollout_msgs:
                    role = msg.role
                    ids = self.tokenizer(msg.content, add_special_tokens=False).input_ids
                    if role == "assistant":
                        response_ids_list[indice].extend(ids)
                        response_mask_list[indice].extend([1]*len(ids))
                    elif role == "tool":
                        response_ids_list[indice].extend(ids)
                        response_mask_list[indice].extend([0]*len(ids))
                    elif role == "user":
                        response_ids_list[indice].extend(ids)
                        response_mask_list[indice].extend([0]*len(ids))
                    else:
                        raise ValueError(f"Unknown role {role} in rollout messages")
            
            processed_input = self.backend.apply_chat_template(input_msgs_list)
            process_proto = self.backend.prepare_dataproto(processed_input)
            process_proto.non_tensor_batch = prompts.non_tensor_batch
            process_proto.meta_info = prompts.meta_info
            dynamic_info = np.empty(batch_size, dtype=object)
            for i in range(batch_size):
                dynamic_info[i] = {
                    "messages": JsonUtil.json_sanitize(msgs[i]),
                    "resp_ids": str(response_ids_list[i]),
                    "resp_mask": str(response_mask_list[i]),
                }
            process_proto.non_tensor_batch["dynamic_info"]=dynamic_info

            self.logger.info("Multiturn generation finished")
            gather_output = self._prepare_result_proto(process_proto, response_ids_list, response_mask_list)
            
        gather_output.meta_info["timing"] = timing_generate
        gather_output = gather_output.to("cpu")
        
        mock_msgs = [[{"role":"user","content":"How are you"}] for _ in range(batch_size)]
        _, _ = self.backend.generate(mock_msgs, sleep_after_inference=True)
        return gather_output
        