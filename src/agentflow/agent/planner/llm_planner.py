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

class JsonPlanParser:
    """Robust JSON extractor + validator."""
    REQUIRED_TOP = ["problem_brief", "asked_quantity", "assumptions_required", "subtasks"]
    REQUIRED_SUB = ["id", "title", "rationale", "category", "expected_produce", "stop_on_fail"]

    

    @classmethod
    def validate_and_coerce(cls, obj: Dict[str, Any]) -> Dict[str, Any]:
        for k in cls.REQUIRED_TOP:
            obj.setdefault(k, "" if k in ["problem_brief","asked_quantity"] else ([] if k!="subtasks" else []))
        if not isinstance(obj["assumptions_required"], list):
            obj["assumptions_required"] = []
        if not isinstance(obj["subtasks"], list):
            obj["subtasks"] = []

        fixed = []
        for i, st in enumerate(obj["subtasks"], 1):
            if not isinstance(st, dict):
                continue
            for k in cls.REQUIRED_SUB:
                st.setdefault(k, True if k=="stop_on_fail" else "")
            st.setdefault("inputs", {"from":["QUESTION","REASONING"]})
            st.setdefault("tool_hint", {"python": False, "search": False, "max_calls": 1})
            st.setdefault("expected_produce", {"type": "boolean", "schema": {"meaning":"pass/fail"}})
            if not st.get("id"): st["id"] = f"s{i}"
            fixed.append(st)
        obj["subtasks"] = fixed

        obj.setdefault("stop_conditions", [
            "asked_quantity mismatch confirmed",
            "critical assumption violated"
        ])
        return obj

    @staticmethod
    def to_plan(obj: Dict[str, Any]) -> Plan:
        subtasks = [Subtask(
            id=FIXED_OVERALL_SUBTASK["id"],
            title=FIXED_OVERALL_SUBTASK["title"],
            rationale=FIXED_OVERALL_SUBTASK["rationale"],
            category=FIXED_OVERALL_SUBTASK["category"],
            inputs=FIXED_OVERALL_SUBTASK["inputs"],
            tool_hint=FIXED_OVERALL_SUBTASK["tool_hint"],
            expected_produce=FIXED_OVERALL_SUBTASK["expected_produce"],
            stop_on_fail=FIXED_OVERALL_SUBTASK["stop_on_fail"],
        )]
        for st in obj.get("subtasks", []):
            subtasks.append(Subtask(
                id=st["id"],
                title=st["title"],
                rationale=st["rationale"],
                category=st["category"],
                inputs=st.get("inputs", {}),
                tool_hint=st.get("tool_hint", {}),
                expected_produce=st.get("expected_produce", {}),
                stop_on_fail=bool(st.get("stop_on_fail", True))
            ))
        return Plan(
            problem_brief=obj.get("problem_brief",""),
            asked_quantity=obj.get("asked_quantity",""),
            assumptions_required=obj.get("assumptions_required",[]),
            subtasks=subtasks,
            reasoning=obj.get("reasoning",""),
            stop_conditions=obj.get("stop_conditions",[]),
            meta=obj.get("meta",{})
        )
        
        
RETRY_FORMAT_SUFFIX = (
   "Output a standard json-format object only",
)

MINIMAL_FALLBACK_OBJ = {
    "problem_brief": "",
    "asked_quantity": "",
    "assumptions_required": [],
    "subtasks": [
        {"id":"s0","title":"intent check","rationale":"WHAT vs RESULT",
         "category":"intent_check","inputs":{"from":["QUESTION","REASONING"]},
         "tool_hint":{"python":False,"search":False,"max_calls":1},
         "expected_produce":{"type":"boolean","schema":{"meaning":"match"}},
         "stop_on_fail": True},
        {"id":"s1","title":"assumption audit","rationale":"illegal premise?",
         "category":"assumption_audit","inputs":{"from":["QUESTION","REASONING"]},
         "tool_hint":{"python":False,"search":False,"max_calls":1},
         "expected_produce":{"type":"boolean","schema":{"meaning":"ok"}},
         "stop_on_fail": True}
    ],
    "stop_conditions": ["asked_quantity mismatch confirmed"]
}

FIXED_OVERALL_SUBTASK = {
    "id": "fixed_overall",
    "title": "Overall QA Judgement",
    "rationale": "Directly judge correctness using only QUESTION and full REASONING (final answer included).",
    "category": "final_consistency",   
    "inputs": {"from": ["QUESTION", "REASONING"]},
    "tool_hint": {"python": False, "search": False, "max_calls": 1},
    "expected_produce": {
        "type": "boolean",
        "schema": {
            "meaning": "is the final stated answer correct?",
            "must_extract_final_answer": True,     
            "must_state_reason": True              
        }
    },
    "stop_on_fail": False
}

class LLMPlanner(BasePlanner):
    def __init__(self, backend: CanGenerate, system_prompt: Optional[str]=None, * ,max_retries: int = 3):
        self.backend = backend
        self.system_prompt = system_prompt or PLANNER_SYSTEM
        self.max_retries = max_retries
    

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
        coerced = JsonPlanParser.validate_and_coerce(obj)
        return JsonPlanParser.to_plan(coerced)

    def plan(self, sequences: List[str], extra: Optional[List[Dict[str,Any]]] = None, **kwargs) -> List[Plan]:
        batch_prompts = [self._build_prompt(seq, strengthen_format=False) for seq in sequences]
        texts, metas = self.backend.generate(batch_prompts, extra=extra if extra else None)

        plans: List[Optional[Plan]] = [None] * len(sequences)
        failed_idxs: List[int] = []

        for i, raw in enumerate(texts):
            try:
                obj = self._parse_plan_obj(raw)
                plans[i] = self._coerce_to_plan(obj)
            except Exception:
                failed_idxs.append(i)


        for i in failed_idxs:
            attempt = 1
            last_err: Optional[Exception] = None
            strengthen = True  

            while attempt <= self.max_retries:
                try:
                    prompt_i = self._build_prompt(sequences[i], strengthen_format=strengthen)
                    texts_i, metas_i = self.backend.generate(
                        [prompt_i],
                        extra=[(extra[i] if extra else {})],
                    )
                    raw_i = texts_i[0]
                    obj_i = self._parse_plan_obj(raw_i)
                    plans[i] = self._coerce_to_plan(obj_i)
                    break  

                except Exception as e:
                    last_err = e
                    attempt += 1

            if plans[i] is None:
                try:
                    plans[i] = self._coerce_to_plan(MINIMAL_FALLBACK_OBJ)
                except Exception:
                    raise last_err or RuntimeError("planner failed without explicit error")
        return [p for p in plans]  
