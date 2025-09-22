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

@dataclass
class Plan:
    problem_brief: str
    asked_quantity: str
    assumptions_required: List[str]
    subtasks: List[Subtask]
    stop_conditions: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

class BasePlanner(Protocol):
    def plan(self, sequences: List[str], extra: Optional[List[Dict[str,Any]]] = None, **kwargs) -> List[Plan]:
        ...
