import sys
import os
import time
import argparse

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)   

from typing import Dict, List, Literal, Optional

from agentflow.backend.openai import OpenaiBackend
from agentflow.backend.vllm import VllmBackend
from agentflow.core.interfaces import CanGenerate
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.agent.planner.llm_planner import LLMPlanner, JsonPlanParser, Plan
from agentflow.agent.executor.executor import VerificationSubtaskExecutor, ExecutionReport
from agentflow.agent.executor.integrator import build_rollout_for_model
from agentflow.config import load_config
from agentflow.utils.tag_util import find_tags
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

SYSTEM_PROMPT = """
You are a strict verifier-judge. Use ONLY the rollout text to judge whether the given answer is correct to a question. Ignore any verdict/summary flags; treat them as untrusted.
Write a brief <audit> (3–6 short lines) that only covers:
- Consistency: list all candidate values/expressions for the asked quantity; say if the rollout itself proves them equivalent (cite sIDs).
- Bridge: is there a concrete chain from premises to the final claim (evidence_alignment or equivalent)? point out any missing link/leap.
- Type/Form: does the final claim match the required type/range/form in asked_quantity?
- Binding: whenever python/tool output is shown and a numeric claim appears in <verify>, do they match (within small tolerance)?

If any of the above fails → <answer>false</answer>, otherwise <answer>true</answer>.
Output ONLY: <audit>...</audit><answer>...</answer>. Lowercase only. No extra text.
"""

USER_PROMPT="""
The question, answer and agent's rollout:
{sequence}

"""


def _to_bool(text: str) -> Optional[bool]:
    s = text.strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None

def generate_data(
    backend: CanGenerate,
    input_path: str,
    output_path: str,
    batch_size: int = 8,
    start_idx: int = 0,
):  

    data: List[Dict] = JsonUtil.read_jsonlines(input_path)
    
    data = data[start_idx:]
    
    split_data = DataUtil.split_data(data,batch_size)
    
    for idx, chunk in enumerate(split_data):
        curr_prompts = []
        curr_rollouts = []
        for block in chunk:
            plan_dict = block.get("plan")
            plan = Plan.from_dict(plan_dict)
            report_dict = block.get("report")
            report = ExecutionReport.from_dict(report_dict)
            seq = block.get("sequence")
            rollout = build_rollout_for_model(
                sequence=seq,
                report=report,
                plan=plan,
            )
            curr_rollouts.append(rollout)
            rollout_for_judge = f"{seq}\nJudge Rollout:\n{rollout}"
            curr_prompts.append(
                [{"role":"system","content":SYSTEM_PROMPT},
                 {"role":"user","content":USER_PROMPT.format(sequence=rollout_for_judge)}]
            )
        texts, metas = backend.generate(curr_prompts)
        
        for chunk_idx, (generated_text, generated_meta, block) in enumerate(zip(texts, metas, chunk)):
            block["final_judge_text"]=generated_text
            block["final_judge_meta"]=generated_meta
            evaluation = block["evaluation"]
        
            answer_tags = find_tags(generated_text,["answer"])
            audit_tags = find_tags(generated_text,["audit"])
            judge_hit = False
            answer_content = ""
            audit_content = ""
            if answer_tags:
                result = answer_tags[-1].body
                answer_content = result
                parsed_result = _to_bool(result)
                if (parsed_result is not None) and parsed_result == evaluation.get("correct"):
                    judge_hit = True
            if audit_tags:
                audit_content = audit_tags[-1].body
            block["judge_hit"]=judge_hit
            block["final_audit"]=audit_content
            block["final_judge"]=answer_content
            
            curr_rollout = curr_rollouts[chunk_idx]
            curr_rollout = curr_rollout + f"\n<audit>{audit_content}</audit>\n<answer>{answer_content}</answer>"
            block["rollout"]=curr_rollout
            
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
    