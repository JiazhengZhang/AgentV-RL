import sys
import os
import time
import argparse

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)   

from typing import Dict, List, Literal, Optional

from agentflow.backend.openai import OpenaiBackend
from agentflow.backend.vllm import VllmBackend
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.agent.planner.llm_planner import LLMPlanner, JsonPlanParser, Plan
from agentflow.agent.executor.executor import VerificationSubtaskExecutor, ExecutionReport
from agentflow.agent.executor.integrator import build_rollout_for_model
from agentflow.config import load_config
from agentflow.utils.json_util import JsonUtil
from agentflow.utils.tag_util import find_tags


input_path = "/root/workspace/agent-rm/datasets/polaris/sft-1001/polaris-qwen2.5-7b-idx0-1000-exp3-integration.jsonl"
output_path = "/root/workspace/agent-rm/datasets/polaris/sft-1001/polaris-qwen2.5-7b-idx0-1000-exp3-integration-filtered.jsonl"

def _to_bool(text: str) -> Optional[bool]:
    s = text.strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None

def main():
    num_hit_true_positive = 0
    num_hit_false_negative = 0
    num_all_tp = 0
    num_all_fn = 0
    data = JsonUtil.read_jsonlines(input_path)
    for block in data:
        evaluation = block["evaluation"]
        
        final_judge_text = block["final_judge_text"]
        answers = find_tags(final_judge_text,["answer"])
        judge_hit = False
        if answers:
            result = answers[-1].body
            parsed_result = _to_bool(result)
            gt = evaluation.get("correct")
            if gt:
                num_all_tp+=1
            else:
                num_all_fn+=1
            if parsed_result == gt:
                judge_hit = True
                if parsed_result is True:
                    num_hit_true_positive += 1
                else:
                    num_hit_false_negative += 1
        block["judge_hit"]=judge_hit
        if judge_hit:
            JsonUtil.write_jsonlines(
                output_path,
                [block],
            )
    print(f"True-positive-percentage: {num_hit_true_positive/num_all_tp}")
    print(f"False-negative-percentage: {num_hit_false_negative/num_all_fn}")
    print(f"Correct-percentage: {(num_hit_true_positive+num_hit_false_negative)/len(data)}")
            
            
            
    

if __name__ == "__main__":
    main()