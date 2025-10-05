from typing import List, Tuple, Dict, Any, Union, Optional
from logging import Logger


from vllm import LLM, SamplingParams

from agentflow.core.interfaces import CanGenerate,SupportChatTemplate
from agentflow.utils.log_util import get_logger
from agentflow.utils.chat_template import is_chat_messages, safe_apply_chat_template, ChatTemplateDefaultsMixin, left_truncate_text_by_token, resolve_context_window_len

class VllmBackend(ChatTemplateDefaultsMixin, CanGenerate, SupportChatTemplate):
    """A VLLM backend for text generation
    """
    
    def __init__(self, config: Dict[str,Any], logger: Logger = None, **kwargs):
        super().__init__()
        ChatTemplateDefaultsMixin.__init__(self)
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
        self.engine = LLM(
            model=self.backend_config["model_path"],
            dtype=self.backend_config.get("dtype","auto"),
            gpu_memory_utilization=self.vllm_config.get("gpu_memory_utilization",0.8),
            tensor_parallel_size=self.vllm_config["tensor_parallel_size"],
            trust_remote_code=True,
        )
        
        self.tokenizer = self.engine.get_tokenizer()
    
    
    def apply_chat_template(self, messages: List[List[Dict[str,str]]], 
                            tokenize=False, 
                            add_generation_prompt=True, 
                            **additional_params) -> Union[str,Any]:
        merged = {**self._chat_template_defaults, **additional_params}
        result, _ = safe_apply_chat_template(
            self.tokenizer,
            messages=messages,
            tokenize = tokenize,
            add_generation_prompt = add_generation_prompt,
            explicit_max_model_len=resolve_context_window_len(self.engine, self.tokenizer),
            generation_max_new_tokens=self.sampling_config.get("max_tokens",1024),
            **merged
        )
        return result
    
    def generate(self, prompts: List, extra: List[Dict] = None, **kwargs) -> Tuple[List[str],List[Dict]]:
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
            max_prompt_len = resolve_context_window_len(self.engine, self.tokenizer) - self.sampling_config.get("max_tokens",1024) - 32
            max_prompt_len = max(max_prompt_len, 128)
            for i in range(len(prompts)):
                prompts[i]=left_truncate_text_by_token(self.tokenizer, str(prompts[i]), max_prompt_len)
                
            
        
        results = self.engine.generate(
            prompts=prompts,
            sampling_params=self.sampling_params,
        )
        texts = [result.outputs[0].text for result in results]
        return texts, [{"raw_output": result, "prompt":prompt} for result, prompt in zip(results, prompts)]
    
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
        self.engine = llm
        
        self._parse_config()
        
        self.tokenizer = self.engine.get_tokenizer()
    
    
    def apply_chat_template(self, messages: List[List[Dict[str,str]]], 
                            tokenize=False, 
                            add_generation_prompt=True, 
                            **additional_params) -> Union[str,Any]:
        merged = {**self._chat_template_defaults, **additional_params}
        result, _ = safe_apply_chat_template(
            self.tokenizer,
            messages=messages,
            tokenize = tokenize,
            add_generation_prompt = add_generation_prompt,
            explicit_max_model_len=resolve_context_window_len(self.engine, self.tokenizer),
            generation_max_new_tokens=self.sampling_config.get("max_tokens",1024),
            **merged
        )
        return result
    
    def generate(self, prompts: List, extra: List[Dict] = None, **kwargs) -> Tuple[List[str],List[Dict]]:
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
            max_prompt_len = resolve_context_window_len(self.engine, self.tokenizer) - self.sampling_config.get("max_tokens",1024) - 32
            max_prompt_len = max(max_prompt_len, 128)
            for i in range(len(prompts)):
                prompts[i]=left_truncate_text_by_token(self.tokenizer, str(prompts[i]), max_prompt_len)
                
            
        
        results = self.engine.generate(
            prompts=prompts,
            sampling_params=self.sampling_params,
        )
        texts = [result.outputs[0].text for result in results]
        return texts, [{"raw_output": result, "prompt":prompt} for result, prompt in zip(results, prompts)]
    
    def _parse_config(self):
        backend_config = {}
        sampling_config = {}
        vllm_config = {}
        self.backend_name = "vllm"
        self.backend_config = backend_config
        self.sampling_config = sampling_config
        self.vllm_config = vllm_config