import os
import re
import copy
import torch  
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional, Protocol, Sequence
from collections import defaultdict
from logging import Logger

import torch
from vllm import LLM, SamplingParams
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask

from agentflow.core.interfaces import CanGenerate,SupportChatTemplate
from agentflow.utils.log_util import get_logger
from agentflow.utils.chat_template import is_chat_messages, safe_apply_chat_template, ChatTemplateDefaultsMixin, left_truncate_text_by_token, resolve_context_window_len

class VerlWg(Protocol):
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        ...
        
    
class VerlWgBackend(ChatTemplateDefaultsMixin, CanGenerate, SupportChatTemplate):
    """A backend using verl working group for RL rollout generation with verl
    """
    
    def __init__(
        self, 
        config: Dict[str,Any],
        wg: VerlWg,
        tokenizer: PreTrainedTokenizer,
        logger: Logger = None, 
        max_prompt_length: int = 8192,
        truncation: bool = True,
        chunk_size: Optional[int] = None,  
        pad_fill_text: Optional[str] = None,  
        **kwargs,
    ):
        super().__init__()
        ChatTemplateDefaultsMixin.__init__(self)
        self.config = config 
        if logger:
            self.logger = logger
        else:
            self.logger = get_logger(config, __name__)
        self.wg = wg
        self.tokenizer = tokenizer

        self.truncation = truncation
        self.max_prompt_length = max_prompt_length
        self._default_chunk_size = chunk_size
        self._pad_fill_text = pad_fill_text  
        
    def _resolve_chunk_size(self, chunk_size: Optional[int] = None) -> int:
        if isinstance(chunk_size, int) and chunk_size > 0:
            return chunk_size
        if isinstance(self._default_chunk_size, int) and self._default_chunk_size > 0:
            return self._default_chunk_size
        return 8
    
    
    def apply_chat_template(self, messages: List[List[Dict[str,str]]], 
                            tokenize=False, 
                            add_generation_prompt=True, 
                            **additional_params) -> Union[str, Any, List[str]]:
        merged = self._merge_for_call(additional_params)
        result, _ = safe_apply_chat_template(
            self.tokenizer,
            messages=messages,
            tokenize = tokenize,
            add_generation_prompt = add_generation_prompt,
            explicit_max_model_len=self.max_prompt_length,
            **merged
        )
        return result
    
    def _pad_to_multiple(
        self,
        batch: Sequence[Any],
        chunk_size: int,
        *, 
        is_chat: bool,
    ) -> Tuple[list, list]:
        batch = list(batch)
        n = len(batch)
        if n == 0 or chunk_size <= 1:
            return batch, [True] * n

        remain = n % chunk_size
        if remain == 0:
            return batch, [True] * n

        need = chunk_size - remain
        keep_mask = [True] * n

        if is_chat:
            filler = batch[-1] if n > 0 else [{"role": "user", "content": self._pad_fill_text or (self.tokenizer.eos_token or "")}]
        else:
            filler = batch[-1] if n > 0 else (self._pad_fill_text or (self.tokenizer.eos_token or ""))

        for _ in range(need):
            batch.append(filler if not isinstance(filler, (dict, list)) else copy.deepcopy(filler))
            keep_mask.append(False)

        return batch, keep_mask
    
    def prepare_dataproto(self, prompts: List[str], **kwargs) -> DataProto:
        raw_input_ids = []
        raw_atten_masks = []
        raw_position_ids = []
        for prompt in prompts:
            model_inputs = self.tokenizer(
                prompt,
                add_special_tokens=False,
                return_attention_mask=True,
                return_tensors="pt"
            )

            single_input_ids = model_inputs.pop("input_ids")
            single_attention_mask = model_inputs.pop("attention_mask")
        
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=single_input_ids,
                attention_mask=single_attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="right",
            )
            position_ids = compute_position_id_with_mask(attention_mask)
            
            raw_input_ids.append(input_ids[0])
            raw_atten_masks.append(attention_mask[0])
            raw_position_ids.append(position_ids[0])
        
        input_ids = torch.stack(raw_input_ids, dim=0)
        attention_mask = torch.stack(raw_atten_masks, dim=0)
        position_ids = torch.stack(raw_position_ids, dim=0)

        tensor_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        
        input_proto_meta = {
            "eos_token_id": self.tokenizer.eos_token_id, 
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        proto = DataProto.from_single_dict(
            tensor_dict,
            meta_info=input_proto_meta,
        )
        return proto
    
    def resolve_dataproto(self, proto: DataProto) -> List[str]:
        responses_ids = proto.batch["responses"]
        response_texts = self.tokenizer.batch_decode(responses_ids, skip_special_tokens=True)
        return response_texts
        

    def _generate(self, prompts: List, extra: List[Dict] = None, **kwargs) -> Tuple[List[str],List[Dict]]:

        chunk_size = self._resolve_chunk_size(kwargs.pop("chunk_size", None))
        sleep_after_inference = kwargs.pop("sleep_after_inference",True)

        is_chat = is_chat_messages(prompts)
        if is_chat:
            raw_prompts = self.apply_chat_template(prompts)
            if isinstance(raw_prompts, str):
                raw_prompts = [raw_prompts]
        else:
            for i in range(len(prompts)):
                prompts[i] = left_truncate_text_by_token(
                    self.tokenizer, str(prompts[i]), self.max_prompt_length
                )
            raw_prompts = prompts

        original_n = len(raw_prompts)

        
        padded_prompts, keep_mask = self._pad_to_multiple(raw_prompts, chunk_size, is_chat=False)
        
        padded_size = len(padded_prompts)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        input_proto = self.prepare_dataproto(padded_prompts)

        output_proto = self.wg.generate_sequences(input_proto, **kwargs)


        responses_ids = output_proto.batch["responses"]
        response_texts_all = self.tokenizer.batch_decode(responses_ids, skip_special_tokens=True)

        response_texts = [t for t, keep in zip(response_texts_all, keep_mask) if keep]
        

        metas = [{"raw_output": output_proto, "prompt": p, "padded_size":padded_size, "original_size":original_n} for p in (raw_prompts[:original_n])]

        return response_texts, metas
    
    
    def generate(self, prompts: List, extra: List[Dict] = None, **kwargs) -> Tuple[List[str],List[Dict]]:
        """Generate sequences with gievn prompt list

        Args:
            prompts (List): Prompt list of chat messages or raw str. If chat messages are provided, it will automatically apply chat template
            extra (List[Dict], optional): Extra info dicts. Defaults to None.

        Returns:
            Tuple[List[str],List[Dict]]: Generated sequences and any metainfo
                - The metainfo format: {"raw_output":Dataproto}
        """
        return self._generate(prompts, extra, **kwargs)
    
    def generate_sequences(self, prompts: DataProto, **kwargs):
        return self.wg.generate_sequences(prompts, **kwargs)
    
    
    

