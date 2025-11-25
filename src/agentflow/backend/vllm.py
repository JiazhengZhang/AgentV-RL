from typing import List, Tuple, Dict, Any, Union, Optional
from logging import Logger


from vllm import LLM, SamplingParams

from agentflow.core.interfaces import CanGenerate,SupportChatTemplate
from agentflow.utils.log_util import get_logger
from agentflow.utils.vllm import update_sampling_params
from agentflow.utils.chat_template import is_chat_messages, safe_apply_chat_template, ChatTemplateDefaultsMixin, left_truncate_text_by_token, resolve_context_window_len

class VllmBackend(ChatTemplateDefaultsMixin, CanGenerate, SupportChatTemplate):
    """A VLLM backend for text generation
    """
    
    def __init__(self, config: Dict[str,Any], logger: Logger = None, **kwargs):
        super().__init__()
        self.config = config
        if logger:
            self.logger = logger
        else:
            self.logger = get_logger(config, __name__)
        self._parse_config()
        self.sampling_params = SamplingParams(
            temperature=self.sampling_config.get("temperature",1),
            max_tokens=self.sampling_config.get("max_tokens",1024),
            top_p=self.sampling_config.get("top_p",1.0),
            top_k=self.sampling_config.get("top_k",20),
            stop=self.vllm_config.get("stop_tokens",[]),
            repetition_penalty=self.vllm_config.get("repetition_penalty",1.0),
            include_stop_str_in_output=True,
        )
        self.vllm = LLM(
            model=self.backend_config["model_path"],
            dtype=self.backend_config.get("dtype","auto"),
            gpu_memory_utilization=self.vllm_config.get("gpu_memory_utilization",0.8),
            tensor_parallel_size=self.vllm_config["tensor_parallel_size"],
            trust_remote_code=True,
            enable_sleep_mode=self.vllm_config.get("enable_sleep_mode",False),
            max_model_len=self.vllm_config.get("max_model_len",12800),
            max_num_batched_tokens=self.vllm_config.get("max_num_batched_tokens",12800),
            max_num_seqs=self.vllm_config.get("max_num_seqs",256),
            enable_prefix_caching=self.vllm_config.get("enable_prefix_caching",True),
        )
        
        self.tokenizer = self.vllm.get_tokenizer()
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
    
    def _generate(self, prompts: List, extra: List[Dict] = None, **kwargs) -> Tuple[List[str],List[Dict]]:
        """Generate sequences with gievn prompt list

        Args:
            prompts (List): Prompt list of chat messages or raw str. If chat messages are provided, it will automatically apply chat template
            extra (List[Dict], optional): Extra info dicts. Defaults to None.

        Returns:
            Tuple[List[str],List[Dict]]: Generated sequences and any metainfo
                - The metainfo format: {"raw_output":<raw_vllm_output_object>}
        """
        if is_chat_messages(prompts):
            prompts = self.apply_chat_template(prompts)
        else:
            max_prompt_len = resolve_context_window_len(self.vllm, self.tokenizer) - self.sampling_config.get("max_tokens",1024) - 32
            max_prompt_len = max(max_prompt_len, 128)
            for i in range(len(prompts)):
                prompts[i]=left_truncate_text_by_token(self.tokenizer, str(prompts[i]), max_prompt_len)
                
            
        with update_sampling_params(self.sampling_params, **kwargs):
            results = self.vllm.generate(
                prompts=prompts,
                sampling_params=self.sampling_params,
                use_tqdm=self.use_tqdm,
            )
        texts = [result.outputs[0].text for result in results]
        return texts, [{"raw_output": result, "prompt":prompt} for result, prompt in zip(results, prompts)]
    
    def generate(self, prompts: List, extra: List[Dict] = None, **kwargs) -> Tuple[List[str],List[Dict]]:
        """Generate sequences with gievn prompt list

        Args:
            prompts (List): Prompt list of chat messages or raw str. If chat messages are provided, it will automatically apply chat template
            extra (List[Dict], optional): Extra info dicts. Defaults to None.

        Returns:
            Tuple[List[str],List[Dict]]: Generated sequences and any metainfo
                - The metainfo format: {"raw_output":<raw_vllm_output_object>}
        """
        return self._generate(prompts, extra, **kwargs)
    
    def get_vllm_instance(self) -> LLM:
        return self.vllm
    
    
    def _parse_config(self):
        backend_config = self.config["backend"]
        sampling_config = backend_config["sampling"]
        vllm_config = backend_config["vllm"]
        self.backend_name = "vllm"
        self.backend_config = backend_config
        self.sampling_config = sampling_config
        self.vllm_config = vllm_config
        
    
class VllmInjectionBackend(ChatTemplateDefaultsMixin, CanGenerate, SupportChatTemplate):
    """A VLLM backend for text generation
    """
    
    def __init__(
        self, 
        config: Dict[str,Any],
        llm: LLM, 
        sampling_params: SamplingParams,
        logger: Logger = None, 
        **kwargs,
    ):
        super().__init__()
        ChatTemplateDefaultsMixin.__init__(self)
        self.config = config
        if logger:
            self.logger = logger
        else:
            self.logger = get_logger(config, __name__)
        self.sampling_params = sampling_params
        self.vllm = llm
        
        self._parse_config()
        
        self.tokenizer = self.vllm.get_tokenizer()
        self.use_tqdm = self.vllm_config.get("use_tqdm",True)
    
    
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
            explicit_max_model_len=resolve_context_window_len(self.vllm, self.tokenizer),
            generation_max_new_tokens=self.sampling_config.get("max_tokens",1024),
            **merged
        )
        return result
    
    def _generate(self, prompts: List, extra: List[Dict] = None, **kwargs) -> Tuple[List[str],List[Dict]]:
        """Generate sequences with gievn prompt list

        Args:
            prompts (List): Prompt list of chat messages or raw str. If chat messages are provided, it will automatically apply chat template
            extra (List[Dict], optional): Extra info dicts. Defaults to None.

        Returns:
            Tuple[List[str],List[Dict]]: Generated sequences and any metainfo
                - The metainfo format: {"raw_output":<raw_vllm_output_object>}
        """
        if is_chat_messages(prompts):
            prompts = self.apply_chat_template(prompts)
        else:
            max_prompt_len = resolve_context_window_len(self.vllm, self.tokenizer) - self.sampling_config.get("max_tokens",1024) - 32
            max_prompt_len = max(max_prompt_len, 128)
            for i in range(len(prompts)):
                prompts[i]=left_truncate_text_by_token(self.tokenizer, str(prompts[i]), max_prompt_len)
    
            
        with update_sampling_params(self.sampling_params, **kwargs):
            results = self.vllm.generate(
                prompts=prompts,
                sampling_params=self.sampling_params,
                use_tqdm=self.use_tqdm,
            )
        texts = [result.outputs[0].text for result in results]
        return texts, [{"raw_output": result, "prompt":prompt} for result, prompt in zip(results, prompts)]
    
    def generate(self, prompts: List, extra: List[Dict] = None, **kwargs) -> Tuple[List[str],List[Dict]]:
        """Generate sequences with gievn prompt list

        Args:
            prompts (List): Prompt list of chat messages or raw str. If chat messages are provided, it will automatically apply chat template
            extra (List[Dict], optional): Extra info dicts. Defaults to None.

        Returns:
            Tuple[List[str],List[Dict]]: Generated sequences and any metainfo
                - The metainfo format: {"raw_output":<raw_vllm_output_object>}
        """
        return self._generate(prompts, extra, **kwargs)
    

    def get_vllm_instance(self) -> LLM:
        return self.vllm
    
    def _parse_config(self):
        backend_config = self.config["backend"]
        sampling_config = backend_config["sampling"]
        vllm_config = backend_config["vllm"]
        self.backend_name = "vllm"
        self.backend_config = backend_config
        self.sampling_config = sampling_config
        self.vllm_config = vllm_config