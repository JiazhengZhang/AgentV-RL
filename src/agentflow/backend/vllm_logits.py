# src/backends/hybrid_backend.py
from typing import List, Tuple, Dict, Any, Sequence, Union
from logging import Logger
import math

from vllm import LLM, SamplingParams
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentflow.utils.log_util import get_logger
from agentflow.utils.chat_template import is_chat_messages, safe_apply_chat_template, ChatTemplateDefaultsMixin
from agentflow.core.interfaces import CanGenerate, CanChoiceProbs,SupportChatTemplate

class VllmChoiceLogitsBackend(CanGenerate, CanChoiceProbs,SupportChatTemplate,ChatTemplateDefaultsMixin):
    """A Vllm backend for both text generation and prob calculation
    The class will both load vllm LLM object and transformers model, please make sure there are enough gpu memory.
    """
    def __init__(self, config: Dict[str, Any], logger: Logger | None = None, **kwargs):
        super().__init__()
        ChatTemplateDefaultsMixin.__init__(self)
        self.config = config
        self.logger = logger or get_logger(config, __name__)
        self._parse_config()

        self.sampling_params = SamplingParams(
            temperature=self.sampling_config.get("temperature", 1.0),
            max_tokens=self.sampling_config.get("max_tokens", 1024),
            top_p=self.sampling_config.get("top_p", 1.0),
            top_k=self.sampling_config.get("top_k", 50),
            stop=self.vllm_config.get("stop_tokens", []),
            include_stop_str_in_output=True,
        )
        self.vllm = LLM(
            model=self.backend_config["model_path"],
            dtype=self.backend_config.get("dtype", "auto"),
            gpu_memory_utilization=self.vllm_config.get("gpu_memory_utilization", 0.85),
            tensor_parallel_size=self.vllm_config.get("tensor_parallel_size", 1),
            trust_remote_code=True,
        )

        hf_dtype = self.hf_config.get("torch_dtype", "auto")
        torch_dtype = "auto" if hf_dtype == "auto" else getattr(torch, hf_dtype) 
        self.tokenizer = AutoTokenizer.from_pretrained(self.backend_config["model_path"], use_fast=True, trust_remote_code=True)
        self.lm  = AutoModelForCausalLM.from_pretrained(
            self.backend_config["model_path"],
            torch_dtype=("auto" if torch_dtype=="auto" else torch_dtype),
            trust_remote_code=True
        ).to(self.hf_config.get("device", "cuda")).eval()


    def apply_chat_template(self, messages: List[Dict[str,str]], 
                            tokenize=False, 
                            add_generation_prompt=True, 
                            **additional_params) -> Union[str,Any]:
        merged = {**self._chat_template_defaults, **additional_params}
        result, _ = safe_apply_chat_template(
            self.tokenizer,
            messages=messages,
            tokenize = tokenize,
            add_generation_prompt = add_generation_prompt,
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

    def generate(self, prompts: List[Any], extra: List[Dict] | None = None, **kwargs) -> Tuple[List[str], List[Dict]]:
        """Generate sequences with gievn prompt list

        Args:
            prompts (List): Prompt list of chat messages or raw str. If chat messages are provided, it will automatically apply chat template
            extra (List[Dict], optional): Extra info dicts. Defaults to None.

        Returns:
            Tuple[List[str],List[Dict]]: Generated sequences and any metainfo
                - The metainfo format: {"raw_output":<raw_vllm_output_object>}
        """
        sp = self.sampling_params
        if kwargs:
            sp = SamplingParams(**{**sp.__dict__, **kwargs})

        if is_chat_messages(prompts):
            prompts = self.apply_chat_template(prompts)
            
        
        results = self.vllm.generate(prompts=prompts, sampling_params=self.sampling_params)
        texts = [r.outputs[0].text if r.outputs else "" for r in results]
        metas = [{"raw_output": r} for r in results]
        return texts, metas


    
    
    @torch.inference_mode()
    def choice_probs(self,
                    prefixes: Sequence[str],
                    choices: Sequence[Sequence[str]],
                    normalize: str = "sum"  
                    ) -> List[List[float]]:
        device = self.lm.device  
        bos_id = getattr(self.tokenizer, "bos_token_id", None)

        all_group_probs: List[List[float]] = []
        for prefix, choice_list in zip(prefixes, choices):
            if len(prefix) > 8192:
                prefix = prefix[(len(prefix)-8192):]
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

            outputs = self.lm(input_ids=input_ids_tensor,
                                        attention_mask=attention_mask_tensor,
                                        labels=labels_tensor,
                                        )

            valid_counts = (labels_tensor != -100).sum(dim=1).to(torch.float32) 
            valid_counts = torch.clamp(valid_counts, min=1.0)

            avg_nll = outputs.loss  

            with torch.inference_mode():
                # logits = self.lm(input_ids=input_ids_tensor,
                #                             attention_mask=attention_mask_tensor,
                #                             use_cache=False).logits  
                logits = outputs.logits 
                log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)  
                shifted_labels = labels_tensor[:, 1:].clone()
                mask = shifted_labels != -100  

                token_log_probs = log_probs.gather(
                    dim=-1,
                    index=shifted_labels.masked_fill(~mask, 0).unsqueeze(-1)
                ).squeeze(-1)
                token_log_probs = token_log_probs.masked_fill(~mask, 0.0)
                total_logprob_per_sample = token_log_probs.sum(dim=1) 

            if normalize == "avg":
                total_logprob_per_sample = total_logprob_per_sample / valid_counts

            probs = torch.softmax(total_logprob_per_sample, dim=0)  
            all_group_probs.append(probs.detach().cpu().tolist())
            del outputs, shifted_labels, mask,token_log_probs,total_logprob_per_sample
            del input_ids_tensor, labels_tensor, attention_mask_tensor
            torch.cuda.empty_cache()


        return all_group_probs
    
    @torch.no_grad()
    def choice_probs_old(self, prefixes: Sequence[str], choices: Sequence[Sequence[str]]) -> List[List[float]]:

        outs: List[List[float]] = []
        for prefix, chs in zip(prefixes, choices):
            ids = self.tokenizer(prefix, return_tensors="pt").to(self.lm.device)
            logits = self.lm(**ids).logits[0, -1, :]        
            probs  = torch.softmax(logits, -1)

            row=[]
            for c in chs:
                toks = self.tokenizer(c, add_special_tokens=False).input_ids
                tok_id = toks[0]
                row.append(float(probs[tok_id].item()))
            s = sum(row) or 1e-12
            norm = [x/s for x in row]
            outs.append(norm)
        return outs






