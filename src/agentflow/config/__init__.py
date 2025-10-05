import json
import os
from copy import deepcopy
from typing import Any, Dict, Any
import yaml  

logging_config = {
    "level":"DEBUG",
    "log_to_file":True,
    "log_file_dir":"/root/workspace/agent-rm/Agentic-Reward/config",
    "log_file_name":"default.log",
}



sampling_config = {
    "temperature":1,
    "max_tokens":1024,
    "top_p":1.0,
    "top_k":20,
    "repetition_penalty":1.0,
}

vllm_config={
    "gpu_memory_utilization":0.8,
    "tensor_parallel_size":1,
    "stop_tokens":[]
}

openai_config={
    "api_key":"sk-xxx",
    "url":"",
    "max_retries":"",
}

rm_config={
    "device":"cuda",
    "padding_side":"right",
    "apply_sigmoid":True,
    "head_type":"regression",
    "positive_class_index":0,
    "clip_range":[0,100.0],
    "batch_size":16,
}

hf_config = {
    "device":"cuda",
    "torch_dtype":"auto",
}

backend_config={
    "model_path":"",
    "dtype":"auto",
    "sampling":sampling_config,
    "vllm":vllm_config,
    "openai":openai_config,
    "rm":rm_config,
    "hf":hf_config,
}



FULL_CONFIG={
    "logging":logging_config,
    "backend":backend_config,
}

def validate_keys(loaded: Dict[str, Any], default: Dict[str, Any], path: str = ""):
    for key, val in loaded.items():
        if key not in default:
            raise KeyError(f"Unknown config field: '{path + key}'")
        if isinstance(val, dict) and isinstance(default[key], dict):
            validate_keys(val, default[key], path + key + ".")


def merge_config(default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(default)
    for key, val in loaded.items():
        if isinstance(val, dict) and isinstance(default.get(key), dict):
            merged[key] = merge_config(default[key], val)
        else:
            merged[key] = val
    return merged


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from the given path
    
    All the keys should be contained in the config module, with a default value

    Args:
        path (str): config path

    Raises:
        FileNotFoundError: when the path does not exists
        ValueError: 
        - when the config file is not json or yml format; 
        - when the config file contains unexpected keys

    Returns:
        Dict[str, Any]: configuration dict
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            loaded = yaml.safe_load(f)
        else:
            loaded = json.load(f)

    if not isinstance(loaded, dict):
        raise ValueError("Config file root must be a JSON/YAML object")
    
    validate_keys(loaded, FULL_CONFIG)
    
    complete_config = merge_config(FULL_CONFIG, loaded)
    
    return complete_config
