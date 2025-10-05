import sys
import os
import time
import argparse

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)   

from typing import Dict, List, Literal

from agentflow.backend.openai import OpenaiBackend
from agentflow.backend.vllm import VllmBackend
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.agent.planner.llm_planner import LLMPlanner, JsonPlanParser, Plan
from agentflow.agent.executor.executor import VerificationSubtaskExecutor, ExecutionReport
from agentflow.agent.executor.integrator import build_rollout_for_model
from agentflow.config import load_config
from agentflow.utils.json_util import JsonUtil
from agentflow.utils.data_util import DataUtil


def get_backend(
    config: Dict,
    type: Literal["openai","vllm"],
):
    if type == "openai":
        return OpenaiBackend(config)
    elif type == "vllm":
        return VllmBackend(config)
    else:
        raise ValueError(f"Invalid backend type {type}")
    
    
DEFAULT_SEQ_TEMPLATE="""
### Problem ###
{problem}

### Solution ###
{solution}
"""


def generate_data(
    backend,
    input_path: str,
    output_path: str,
    batch_size: int = 8,
    start_idx: int = 0,
):  
    registry = ToolRegistry()
    registry.register(PythonExecutionTool())
    executor = VerificationSubtaskExecutor(
        backend=backend,
        registry=registry
    )
    data: List[Dict] = JsonUtil.read_jsonlines(input_path)
    
    data = data[start_idx:]
    
    split_data = DataUtil.split_data(data,batch_size)
    
    for idx, chunk in enumerate(split_data):
        curr_plans = []
        curr_seqs = []
        for block in chunk:
            plan_dict = block.get("plan")
            plan = Plan.from_dict(plan_dict)
            seq = block.get("sequence")
            curr_plans.append(plan)
            curr_seqs.append(seq)
        reports = executor.execute(
            sequences=curr_seqs,
            plans = curr_plans,
        )
        
        for report, block in zip(reports,chunk):
            block["report"]=report
        JsonUtil.write_jsonlines(output_path, JsonUtil.json_sanitize(chunk))

        
    

if __name__ == "__main__":
    
    CONFIG_PATH="/root/workspace/agent-rm/Agent-Verifier/config/data_produce.yaml"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Input data path")
    parser.add_argument("--output_path", type=str, required=True, help="Output data path")
    parser.add_argument("--backend_type", type=str, default="openai", help="Backend type")
    parser.add_argument("--config_path", type=str, default=CONFIG_PATH, help="Configuration path")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--start_idx", type=int, default=0, help="Batch size for generation")
    args = parser.parse_args()
    config = load_config(args.config_path)
    
    generate_data(
        backend=get_backend(config,args.backend_type),
        input_path=args.input_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        start_idx=args.start_idx,
    )
    