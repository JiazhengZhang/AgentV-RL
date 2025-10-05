import sys
import os
import time
from typing import Dict, List

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)   

from agentflow.utils.json_util import JsonUtil

def to_parquet(data, file_path):
    df = pd.DataFrame(data)
    df.to_parquet(
        file_path,
    )
    
    
SYSTEM_PROMPT="""You are a single-model, multi-role verifier.
Given a question and a anwer to it, you are required to verify whether the answer is correct.
You are required to conduct a multi-stage verification.

## STAGE 1: PLANNING

### Goal:
- Restate the problem briefly, identify the exact asked quantity, and generate
  a JSON plan with 5–10 atomic subtasks.

### Inputs:
- Original QUESTION and ASSISTANT'S REASONING.

### Output (<plan>…</plan>):
- ONE compact JSON object with keys:
  * problem_brief: one-sentence restatement
  * asked_quantity: exact object to verify (with domain/range/form)
  * assumptions_required: list of necessary assumptions
  * subtasks: list of items, each with
      id (s1…sN contiguous),
      title,
      rationale (≤25 words),
      category ∈ {intent_check, assumption_audit, constraint_parse,
                  evidence_alignment, numeric_spotcheck, derivative_check,
                  edge_case, final_consistency},
      optional inputs/tool_hint

### Constraints:
- Exactly one evidence_alignment subtask (bridge from premises to claim).
- Exactly one final_consistency subtask (global consistency check).
- Start with intent_check (s1) and assumption_audit (s2).
- Duplicates allowed only for {derivative_check, edge_case}, disambiguated in title.
- Escape every backslash inside JSON strings as \\\\.
- Keep JSON minimal, no markdown fences.

## STAGE 2: EXECUTION

### Goal:
- For each planned subtask, verify it independently as a falsifiable check.
- Use tags to record reasoning, tool use, verification, and verdict.

### Inputs:
- The plans from Stage 1.
- Context of the original question and answer.

### Output (<subtasks>…</subtasks>):
- For EACH subtask in id order, emit one <subtask> block.

### Tag requirements:
- <rubric>: list 2–4 decisive axes chosen from:
    Dependency; Scope/Quantifiers; Role-binding/Coreference; Temporal/Causal;
    Transformation legality; Equivalence-of-form; Boundary/Counterexample; External-fact alignment
- <think>: ≤40 words; micro-goal, chosen axes, Known/Unknown, next step.
- <python>: optional python code call, at most once per subtask; left-aligned code; no I/O/os/net/loops; end with exactly one print(...).
- <result>: optional; execution result of code, system-provided.
- <verify>: 60–140 words; audit vs the rubric axes; do not re-solve the full problem.
- <answer>: exactly "true" or "false".

### Policies:
- If decisive evidence is missing, ambiguous, or transform is illegal ,directly output <answer>false</answer>.
- No incomplete tags.

## STAGE 3: JUDGING

### Goal:
- Synthesize the subtask results into an overall judgment of correctness.

### Inputs:
- The context of original question and answer, the plan and all subtask outputs.

### Output:
- <audit> covering:
  1) Consistency: candidate values/expressions for asked quantity; whether subtasks proved equivalence (cite sIDs).
  2) Bridge: is there a concrete evidence_alignment chain from premises to claim? note missing links.
  3) Type/Form: does final claim match asked_quantity’s required type/range/form?
  4) Binding: if python/tool results appear, do they match the numeric claims in <verify>?

- <answer>: one word, "true" or "false".
  Rule: true only if all subtasks passed and audit finds no inconsistency; otherwise false.

## STRICTNESS
- Always emit the four sections in order:
  <plan>…</plan><subtasks>…</subtasks><audit>…</audit><answer>…</answer>
- No extra commentary or tags.
- Before emitting, self-check:
  contiguous IDs; exactly one evidence_alignment and one final_consistency;
  one <subtask> per plan item; each subtask has exactly one <answer>.
"""    

USER_PROMPT="""
Verify the following question and answer:
{sequence}

"""
    
INPUT_PATHS=[
    "/root/workspace/agent-rm/datasets/polaris/sft-1001/polaris-qwen2.5-7b-idx0-1000-exp3-integration-filtered.jsonl",
]
OUTPUT_DIR="/root/workspace/agent-rm/datasets/polaris/sft-1001/processed/"
OUTPUT_NAME="sft_1001-full"
    
if __name__ == "__main__":
    out_data = []
    idx = 0
    for pth in INPUT_PATHS:
        data: List[Dict] = JsonUtil.read_jsonlines(pth)
        
        for block in data:
            eva =  block["evaluation"]
            if eva.get("correct",True) is True:
              continue 
            seq = block["sequence"]
            rollout = block["rollout"]
            messages = [
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":USER_PROMPT.format(sequence=seq)},
                {"role":"assistant", "content":rollout},
            ]
            out_data.append({
                "idx":idx,
                "messages":messages,
                # "meta": block,
            })
            idx+=1
    
    output_full_jsonl = os.path.join(OUTPUT_DIR,f"{OUTPUT_NAME}.jsonl")
    output_full_parquet = os.path.join(OUTPUT_DIR,f"{OUTPUT_NAME}.parquet")
    to_parquet(out_data, output_full_parquet)
    JsonUtil.write_jsonlines(output_full_jsonl,out_data)
    
    