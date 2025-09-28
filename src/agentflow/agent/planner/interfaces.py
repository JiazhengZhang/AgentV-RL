# agentflow/planner/interfaces.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol

@dataclass
class Subtask:
    id: str
    title: str
    rationale: str
    # category 尽量通用：intent_check, assumption_audit, constraint_parse,
    # evidence_alignment, numeric_spotcheck, derivative_check, edge_case, final_consistency 等
    category: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    tool_hint: Dict[str, Any] = field(default_factory=dict)  # {"python": true, "search": false, "max_calls": 1}
    expected_produce: Dict[str, Any] = field(default_factory=dict)  # {"type":"boolean"/"text"/"number","schema":{...}}
    stop_on_fail: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict):
        return Subtask(
            id=data.get("id"),
            title=data.get("title"),
            rationale=data.get("rationale"),
            category=data.get("category"),
            inputs=data.get("inputs",{}),
            tool_hint=data.get("tool_hint",{}),
            expected_produce=data.get("expected_produce",{}),
            stop_on_fail=data.get("stop_on_fail",True)
        )

@dataclass
class Plan:
    problem_brief: str
    asked_quantity: str
    assumptions_required: List[str]
    subtasks: List[Subtask]
    reasoning: str = ""
    stop_conditions: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict):
        subtasks = [Subtask.from_dict(raw_sub) for raw_sub in data.get("subtasks",[])]
        return Plan(
            problem_brief=data.get("problem_brief",""),
            asked_quantity=data.get("asked_quantity",""),
            assumptions_required=data.get("assumptions_required",""),
            subtasks=subtasks,
            reasoning=data.get("reasoning",""),
            stop_conditions=data.get("stop_conditions",[]),
            meta=data.get("meta",{})
        )
    
    

class BasePlanner(Protocol):
    def plan(self, sequences: List[str], extra: Optional[List[Dict[str,Any]]] = None, **kwargs) -> List[Plan]:
        ...
