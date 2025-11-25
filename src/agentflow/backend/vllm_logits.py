# src/backends/hybrid_backend.py
from typing import List, Tuple, Dict, Any, Sequence, Union
from logging import Logger
import math

from vllm import LLM, SamplingParams
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentflow.utils.log_util import get_logger
from agentflow.utils.vllm import update_sampling_params
from agentflow.utils.chat_template import is_chat_messages, safe_apply_chat_template, ChatTemplateDefaultsMixin, left_truncate_text_by_token, resolve_context_window_len
from agentflow.core.interfaces import CanGenerate, CanChoiceProbs,SupportChatTemplate


class VllmChoiceLogitsBackend(ChatTemplateDefaultsMixin, CanGenerate, CanChoiceProbs,SupportChatTemplate):
    """A Vllm backend for both text generation and prob calculation
    The class will both load vllm LLM object and transformers model, please make sure there are enough gpu memory.
    """
    def __init__(self, config: Dict[str, Any], logger: Logger | None = None, **kwargs):
        super().__init__()
        self.config = config
        self.logger = logger or get_logger(config, __name__)
        self._parse_config()

        self.sampling_params = SamplingParams(
            temperature=self.sampling_config.get("temperature", 1.0),
            max_tokens=self.sampling_config.get("max_tokens", 1024),
            top_p=self.sampling_config.get("top_p", 1.0),
            top_k=self.sampling_config.get("top_k", 50),
            stop=self.vllm_config.get("stop_tokens", []),
            repetition_penalty=self.vllm_config.get("repetition_penalty",1.0),
            include_stop_str_in_output=True,
        )
        self.vllm = LLM(
            model=self.backend_config["model_path"],
            dtype=self.backend_config.get("dtype", "auto"),
            gpu_memory_utilization=self.vllm_config.get("gpu_memory_utilization", 0.85),
            tensor_parallel_size=self.vllm_config.get("tensor_parallel_size", 1),
            trust_remote_code=True,
            enable_sleep_mode=self.vllm_config.get("enable_sleep_mode",False),
            max_model_len=self.vllm_config.get("max_model_len",12800),
            max_num_batched_tokens=self.vllm_config.get("max_num_batched_tokens",12800),
            max_num_seqs=self.vllm_config.get("max_num_seqs",256),
            enable_prefix_caching=self.vllm_config.get("enable_prefix_caching",True),
        )

        hf_dtype = self.hf_config.get("torch_dtype", "auto")
        torch_dtype = "auto" if hf_dtype == "auto" else getattr(torch, hf_dtype) 
        self.tokenizer = AutoTokenizer.from_pretrained(self.backend_config["model_path"], use_fast=True, trust_remote_code=True)
        self.lm  = AutoModelForCausalLM.from_pretrained(
            self.backend_config["model_path"],
            torch_dtype=("auto" if torch_dtype=="auto" else torch_dtype),
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        self.use_tqdm = self.vllm_config.get("use_tqdm",True)


    def apply_chat_template(self, messages: List[List[Dict[str,str]]], 
                            tokenize=False, 
                            add_generation_prompt=True, 
                            **additional_params) -> Union[str, Any, List[str]]:
        merged = self._merge_for_call(additional_params)
        if "add_generation_prompt" in merged.keys():
            add_generation_prompt = merged.pop("add_generation_prompt")
        result, _ = safe_apply_chat_template(
            self.tokenizer,
            messages=messages,
            tokenize = tokenize,
            add_generation_prompt = add_generation_prompt,
            explicit_max_model_len=resolve_context_window_len(self.vllm, self.tokenizer),
            generation_max_new_tokens=self.sampling_config.get("max_tokens",1024),
            **merged
        )

        return result
    def _parse_config(self):
        backend_config = self.config["backend"]
        self.backend_config = {
            "model_path": backend_config.get("model_path"),
            "dtype": backend_config.get("dtype", "auto"),
        }
        self.sampling_config = backend_config.get("sampling", {})
        self.vllm_config = backend_config.get("vllm", {})
        self.hf_config = backend_config.get("hf", {"device":"cuda","torch_dtype":"auto"})

    def _generate(self, prompts: List[Any], extra: List[Dict] | None = None, **kwargs) -> Tuple[List[str], List[Dict]]:
        """Generate sequences with gievn prompt list

        Args:
            prompts (List): Prompt list of chat messages or raw str. If chat messages are provided, it will automatically apply chat template
            extra (List[Dict], optional): Extra info dicts. Defaults to None.

        Returns:
            Tuple[List[str],List[Dict]]: Generated sequences and any metainfo
                - The metainfo format: {"raw_output":raw_vllm_output_object}
        """

        if is_chat_messages(prompts):
            prompts = self.apply_chat_template(prompts)
        else:
            max_prompt_len = resolve_context_window_len(self.vllm, self.tokenizer) - self.sampling_config.get("max_tokens",1024) - 32
            max_prompt_len = max(max_prompt_len, 128)
            for i in range(len(prompts)):
                prompts[i]=left_truncate_text_by_token(self.tokenizer, str(prompts[i]), max_prompt_len)
            
        with update_sampling_params(self.sampling_params, **kwargs):
            results = self.vllm.generate(prompts=prompts, sampling_params=self.sampling_params,use_tqdm=self.use_tqdm,)
        texts = [r.outputs[0].text if r.outputs else "" for r in results]
        metas = [{"raw_output": r, "prompt":prompt} for r, prompt in zip(results, prompts)]
        return texts, metas
    
    def generate(self, prompts: List[Any], extra: List[Dict] | None = None, **kwargs) -> Tuple[List[str], List[Dict]]:
        """Generate sequences with gievn prompt list

        Args:
            prompts (List): Prompt list of chat messages or raw str. If chat messages are provided, it will automatically apply chat template
            extra (List[Dict], optional): Extra info dicts. Defaults to None.

        Returns:
            Tuple[List[str],List[Dict]]: Generated sequences and any metainfo
                - The metainfo format: {"raw_output":raw_vllm_output_object}
        """
        texts, metas = self._generate(prompts, extra, **kwargs)
        return texts, metas


    def choice_probs(self,
                    prefixes: Sequence[str],
                    choices: Sequence[Sequence[str]],
                    normalize: str = "sum"  
                    ) -> List[List[float]]:
        probs = self._choice_probs(
            prefixes=prefixes,
            choices=choices,
            normalize=normalize,
        )
        return probs
    
    @torch.inference_mode()
    def _choice_probs(self,
                    prefixes: Sequence[str],
                    choices: Sequence[Sequence[str]],
                    normalize: str = "sum"  
                    ) -> List[List[float]]:
        device = self.lm.device  
        bos_id = getattr(self.tokenizer, "bos_token_id", None)

        all_group_probs: List[List[float]] = []
        for prefix, choice_list in zip(prefixes, choices):
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
           
            input_id_batches = []
            label_batches = []
            choice_token_lens = []

            for choice_text in choice_list:
                choice_ids = self.tokenizer(choice_text, add_special_tokens=False).input_ids
                choice_token_lens.append(len(choice_ids))

                if bos_id is not None:
                    full_ids = [bos_id] + prefix_ids + choice_ids
                    labels = [-100] * (1 + len(prefix_ids)) + choice_ids[:]
                else:
                    full_ids = prefix_ids + choice_ids
                    labels = [-100] * len(prefix_ids) + choice_ids[:]

                input_id_batches.append(full_ids)
                label_batches.append(labels)

            max_len = max(len(ids) for ids in input_id_batches)
            padded_input_ids, padded_labels, attention_masks = [], [], []

            for ids, lbl in zip(input_id_batches, label_batches):
                pad_len = max_len - len(ids)
                padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
                padded_labels.append(lbl + [-100] * pad_len)
                attention_masks.append([1] * len(ids) + [0] * pad_len)

            input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
            labels_tensor = torch.tensor(padded_labels, dtype=torch.long, device=device)
            attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long, device=device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.lm(input_ids=input_ids_tensor,
                                attention_mask=attention_mask_tensor,
                                labels=labels_tensor,
                                )

            valid_counts = (labels_tensor != -100).sum(dim=1).to(torch.float32) 
            valid_counts = torch.clamp(valid_counts, min=1.0)

            logits = outputs.logits 
            logits_shifted = logits[:, :-1, :]  
            shifted_labels = labels_tensor[:, 1:].clone()
            mask = shifted_labels != -100  

            if mask.any():
                flat_logits  = logits_shifted[mask]   # [N_mask, V]
                flat_targets = shifted_labels[mask]   # [N_mask]
                per_tok_nll  = F.cross_entropy(flat_logits, flat_targets, reduction="none")  # [N_mask]
                B = logits.size(0)
                per_sample_nll = torch.zeros(B, device=logits.device, dtype=per_tok_nll.dtype)
                counts = mask.sum(dim=1)    # [B]
                idxs = torch.cumsum(counts, dim=0)
                start = 0
                for b in range(B):
                    end = idxs[b].item()
                    if end > start:
                        per_sample_nll[b] = per_tok_nll[start:end].sum()
                    start = end
            else:
                per_sample_nll = torch.zeros(logits.size(0), device=logits.device)
                    
            total_logprob_per_sample = -per_sample_nll
            if normalize == "avg":
                total_logprob_per_sample = total_logprob_per_sample / valid_counts

            probs = torch.softmax(total_logprob_per_sample, dim=0)  
            all_group_probs.append(probs.detach().cpu().tolist())
            del outputs, logits, logits_shifted, shifted_labels, mask
            del input_ids_tensor, labels_tensor, attention_mask_tensor
            del per_sample_nll, total_logprob_per_sample, probs
            torch.cuda.empty_cache()
            
        return all_group_probs
    
    def get_vllm_instance(self) -> LLM:
        return self.vllm
    






