# agentflow/planner/llm_planner.py
from __future__ import annotations
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict
from agentflow.agent.planner.interfaces import BasePlanner, Plan, Subtask
from agentflow.agent.planner.prompts import PLANNER_SYSTEM, PLANNER_USER_TMPL
from agentflow.utils.json_util import JsonUtil
from agentflow.utils.tag_util import find_tags

from agentflow.core.interfaces import CanGenerate

RETRY_FORMAT_SUFFIX = "Output a standard json-format object only"



FIXED_SELF_SOLVE_SUBTASK = {
    "id": "Judge_by_Self_Solve",
    "title": "Self-solve then compare with the solution",
    "justification": "Independently solve the question without referencing solution; then compare the given answer with self-solve solution.",
    "category": "self_solve",
    "inputs": {"from": ["Problem", "Solution"]}, 
    "tool_spec": {"python": True, "search": False, "max_calls": 5},
    "expected_output": {
        "type": "boolean",
        "schema": {
            "meaning": "Determines whether the given answer is equal to self-solve solution.",
        }
    },
    "stop_on_failure": False
}

FIXED_OVERALL_SUBTASK = {
    "id": "Overall_Correctness",
    "title": "Overall_Correctness",
    "justification": "Analyze the full reasoning process of the original answer to judge whether the given answer is correct",
    "category": "final_consistency",
    "inputs": {"from": ["Problem", "Solution"]},
    "tool_spec": {"python": True, "search": False, "max_calls": 5},
    "expected_output": {
        "type": "boolean",
        "schema": {
            "meaning": "Determines whether the final solution answer and the process are correct.",
        }
    },
    "stop_on_failure": False
}

MINIMAL_FALLBACK_OBJ = {
    "problem_statement": "",
    "target_quantity": "",
    "required_assumptions": [],
    "verification_units": [
        FIXED_OVERALL_SUBTASK,
        FIXED_SELF_SOLVE_SUBTASK,
    ],
    "termination_conditions": ["target_quantity mismatch confirmed"]
}


class JsonPlanParser:
    """Robust JSON extractor + validator."""
    REQUIRED_TOP = ["problem_statement", "target_quantity", "required_assumptions", "verification_units"]
    REQUIRED_SUB = ["id", "title", "justification", "category", "expected_output", "stop_on_failure"]

    @classmethod
    def _inject_fixed_subtasks(cls, obj: Dict[str, Any], max_num_subtasks: Optional[int] = None) -> None:
        """Append fixed subtasks if missing. The order is: self_solve -> overall."""
        ids = {st.get("id", "") for st in obj["verification_units"] if isinstance(st, dict)}
        assert isinstance(obj["verification_units"],list)
        if FIXED_SELF_SOLVE_SUBTASK["id"] not in ids:
            obj["verification_units"].insert(0, dict(FIXED_SELF_SOLVE_SUBTASK))
        ids.add(FIXED_SELF_SOLVE_SUBTASK["id"])
        if FIXED_OVERALL_SUBTASK["id"] not in ids:
            obj["verification_units"].insert(0, dict(FIXED_OVERALL_SUBTASK))
        if max_num_subtasks and isinstance(max_num_subtasks, int):
            obj["verification_units"] = obj["verification_units"][:max_num_subtasks]

    @classmethod
    def validate_and_coerce(cls, obj: Dict[str, Any], max_num_subtasks: Optional[int] = None) -> Dict[str, Any]:
        for k in cls.REQUIRED_TOP:
            obj.setdefault(k, "" if k in ["problem_statement","target_quantity"] else ([] if k!="verification_units" else []))
        if not isinstance(obj["required_assumptions"], list):
            obj["required_assumptions"] = []
        if not isinstance(obj["verification_units"], list):
            obj["verification_units"] = []

        fixed = []
        for i, st in enumerate(obj["verification_units"], 1):
            if not isinstance(st, dict):
                continue
            for k in cls.REQUIRED_SUB:
                st.setdefault(k, True if k=="stop_on_fail" else "")
            st.setdefault("inputs", {"from":["Problem","Solution"]})
            st.setdefault("tool_spec", {"python": False, "search": False, "max_calls": 1})
            st.setdefault("expected_output", {"type": "boolean", "schema": {"meaning":"pass/fail"}})
            if not st.get("id"): st["id"] = f"u{i}"
            fixed.append(st)
        obj["verification_units"] = fixed
        
        cls._inject_fixed_subtasks(obj, max_num_subtasks=max_num_subtasks)

        obj.setdefault("termination_conditions", [
            "target_quantity mismatch confirmed",
            "critical assumption violated"
        ])
        return obj

    @staticmethod
    def to_plan(obj: Dict[str, Any]) -> Plan:
        subtasks = []
        for st in obj.get("verification_units", []):
            subtasks.append(Subtask(
                id=st["id"],
                title=st["title"],
                rationale=st["justification"],
                category=st["category"],
                inputs=st.get("inputs", {}),
                tool_hint=st.get("tool_spec", {}),
                expected_produce=st.get("expected_output", {}),
                stop_on_fail=bool(st.get("stop_on_failure", True))
            ))
        return Plan(
            problem_brief=obj.get("problem_statement",""),
            asked_quantity=obj.get("target_quantity",""),
            assumptions_required=obj.get("required_assumptions",[]),
            subtasks=subtasks,
            reasoning=obj.get("reasoning",""),
            stop_conditions=obj.get("termination_conditions",[]),
            meta=obj.get("meta",{})
        )
        
        


class LLMPlanner(BasePlanner):
    def __init__(self, 
                 backend: CanGenerate, 
                 system_prompt: Optional[str]=None, 
                 * ,
                 max_retries: int = 3,
                 max_num_subtasks: Optional[int] = None
        ):
        self.backend = backend
        self.system_prompt = system_prompt or PLANNER_SYSTEM
        self.max_retries = max_retries
        self.max_num_subtasks = max_num_subtasks
    

    def _build_prompt(self, sequence: str, strengthen_format: bool=False) -> List[Dict[str, str]]:
        user_content = PLANNER_USER_TMPL.format(sequence=sequence)
        if strengthen_format:
            user_content = user_content + RETRY_FORMAT_SUFFIX
        return [
            {"role":"system","content": self.system_prompt},
            {"role":"user","content": user_content}
        ]

    def _parse_plan_obj(self, raw: str) -> Dict[str, Any]:
        obj = JsonUtil.parse_json(raw)  
        if not obj:
            raise ValueError("empty json parsed")
        if isinstance(obj, list):
            if not obj:
                raise ValueError("empty list")
            obj = obj[0]
        if not isinstance(obj, dict):
            raise ValueError("not a dict")
        reasoning_tags = find_tags(raw,["reasoning"])
        reasoning_str = ""
        if reasoning_tags:
            reasoning_str = reasoning_tags[-1].body
        obj["reasoning"] = reasoning_str
        return obj

    def _coerce_to_plan(self, obj: Dict[str, Any]) -> Plan:
        coerced = JsonPlanParser.validate_and_coerce(obj, max_num_subtasks=self.max_num_subtasks)
        return JsonPlanParser.to_plan(coerced)

    def plan(self, sequences: List[str], extra: Optional[List[Dict[str,Any]]] = None, **kwargs) -> List[Plan]:
        batch_prompts = [self._build_prompt(seq, strengthen_format=False) for seq in sequences]
        batch_outputs = ["" for _ in range(len(sequences))]
        texts, metas = self.backend.generate(batch_prompts, extra=extra if extra else None, **kwargs)



        plans: List[Optional[Plan]] = [None] * len(sequences)
        failed_idxs: List[int] = []

        for i, raw in enumerate(texts):
            try:

                batch_outputs[i] = raw
                obj = self._parse_plan_obj(raw)
                plans[i] = self._coerce_to_plan(obj)

            except Exception:
                failed_idxs.append(i)
   


        attempt = 1
        last_err: Optional[Exception] = None
        while failed_idxs and attempt <= self.max_retries:
            try:
                retry_prompts = [
                    self._build_prompt(sequences[i], strengthen_format=True)
                    for i in failed_idxs
                ]
                retry_extra = None
                if extra is not None:
                    retry_extra = [extra[i] if i < len(extra) and extra[i] is not None else {} for i in failed_idxs]

                retry_texts, retry_metas = self.backend.generate(retry_prompts, extra=retry_extra, **kwargs)

                next_failed: List[int] = []
                for pos, raw in enumerate(retry_texts):
                    orig_i = failed_idxs[pos]
                    batch_outputs[orig_i] = raw
                    try:
                        obj_i = self._parse_plan_obj(raw)
                        plans[orig_i] = self._coerce_to_plan(obj_i)
                    except Exception as e:
                        last_err = e
                        next_failed.append(orig_i)

                failed_idxs = next_failed
                attempt += 1

            except Exception as e:
                last_err = e
                attempt += 1

        if failed_idxs:
            for i in failed_idxs:
                try:
                    plans[i] = self._coerce_to_plan(MINIMAL_FALLBACK_OBJ)
                except Exception:
                    raise last_err or RuntimeError("planner failed without explicit error")

        for i, conv in enumerate(batch_prompts):
            conv.append({"role":"assistant","content": batch_outputs[i]})

        return [p for p in plans]