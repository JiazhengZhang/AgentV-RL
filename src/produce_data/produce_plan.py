import sys
import os
import time
import argparse

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)   

from typing import Dict, List, Literal, Any

from agentflow.backend.openai import OpenaiBackend
from agentflow.backend.vllm import VllmBackend
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.agent.planner.llm_planner import LLMPlanner, JsonPlanParser, Plan
from agentflow.agent.executor.executor import VerificationSubtaskExecutor, ExecutionReport
from agentflow.agent.executor.integrator import build_rollout_for_model
from agentflow.config import load_config
from agentflow.utils.json_util import JsonUtil


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
    planner = LLMPlanner(backend)
    data: List[Dict[str, Any]] = JsonUtil.read_jsonlines(input_path)

    pending: List[Dict[str, Any]] = []

    def process_and_append(batch: List[Dict[str, Any]]):
        """对一个批次执行 planner.plan，并将结果以 append 方式写出"""
        if not batch:
            return
        sequences = [it["sequence"] for it in batch]
        plans = planner.plan(sequences)
        assert len(plans) == len(batch)

        out_batch: List[Dict[str, Any]] = []
        for it, plan in zip(batch, plans):
            out_batch.append({
                "idx": f"{it['block_id']}-{it['samp_idx']}",
                "question": it["question"],
                "ground_truth": it["ground_truth"],
                "samp": it["samp"],
                "evaluation": it["evaluation"],
                "sequence": it["sequence"],
                "plan": plan,
            })
        # 逐批追加写入
        JsonUtil.write_jsonlines(output_path, JsonUtil.json_sanitize(out_batch))

    for block_idx, block in enumerate(data):
        if block_idx < start_idx:
            continue

        question = block["question"]
        block_id = block.get("idx", block.get("id", block_idx))
        ground_truth = block.get("ground_truth", block.get("answer"))

        all_samps = block.get("samples", [])
        all_evals = block.get("evaluations", [])
        assert len(all_samps) == len(all_evals), f"block {block_id}: samples/evaluations length mismatch"

        for samp_idx, samp in enumerate(all_samps):
            sequence = DEFAULT_SEQ_TEMPLATE.format(problem=question, solution=samp)
            pending.append({
                "block_id": block_id,
                "question": question,
                "ground_truth": ground_truth,
                "samp": samp,
                "evaluation": all_evals[samp_idx],
                "sequence": sequence,
                "samp_idx": samp_idx,
            })

            if len(pending) >= batch_size:
                process_and_append(pending)
                pending.clear()

    # 处理尾批
    if pending:
        process_and_append(pending)
        pending.clear()
    

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
    


