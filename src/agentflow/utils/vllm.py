from __future__ import annotations
from typing import List, Dict, Optional, Union, Protocol, runtime_checkable
from contextlib import contextmanager
import inspect
import torch

from vllm import LLM, SamplingParams



@runtime_checkable
class SupportVllm(Protocol):
    
    def get_vllm_instance(self) -> LLM:
        ...
        
def _is_sleeping(llm: LLM):
    return llm.llm_engine.is_sleeping()

@contextmanager
def update_sampling_params(sampling_params: SamplingParams, **kwargs):
    old_sampling_params_args = {}
    if kwargs:
        for key, value in kwargs.items():
            if hasattr(sampling_params, key):
                old_value = getattr(sampling_params, key)
                old_sampling_params_args[key] = old_value
                setattr(sampling_params, key, value)
    try:
        yield
    finally:
        for key, value in old_sampling_params_args.items():
            setattr(sampling_params, key, value)
        
@contextmanager
def free_vllm_mem(backend, level: int = 1):
    """Free memory of a vllm-backend for other usage.

    Args:
        backend (SupportVllm): Any backend based on vllm.
        level (int, optional): The sleep level of vllm engine . Defaults to 1.
    """
    assert isinstance(backend, SupportVllm), "backend must be vllm-based to use free cache engine"
    llm = backend.get_vllm_instance()
    if llm and not _is_sleeping(llm):
        llm.sleep(level)
        torch.cuda.empty_cache()
    try:
        yield
    finally:
        if llm:
            torch.cuda.empty_cache()
            llm.wake_up(tags = ["weights","kv_cache"])
    
        
    


