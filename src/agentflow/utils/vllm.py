from __future__ import annotations
from typing import List, Dict, Optional, Union, Protocol, runtime_checkable
from contextlib import contextmanager
import inspect


from vllm import LLM, SamplingParams

from agentflow.backend.vllm import VllmBackend, VllmInjectionBackend
from agentflow.backend.vllm_logits import VllmChoiceLogitsBackend

UseVllmBackend = Union[VllmBackend, VllmInjectionBackend, VllmChoiceLogitsBackend]

@runtime_checkable
class SupportVllm(Protocol):
    
    def get_vllm_instance(self) -> LLM:
        ...
        
def _is_sleeping(llm: LLM):
    return llm.llm_engine.is_sleeping()
        
@contextmanager
def free_cache(backend: UseVllmBackend, level: int = 1):
    backend: SupportVllm
    llm = backend.get_vllm_instance()
    if not _is_sleeping(llm):
        llm.sleep(level)
    yield
    
        
    


