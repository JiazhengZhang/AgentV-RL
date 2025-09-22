# agentflow/agent/executor/interfaces.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from agentflow.agent.planner.interfaces import Plan
from agentflow.tools.base import ToolCallResult


@dataclass
class SubtaskReport:
    subtask_id: str
    raw_trace: str                   # 聚合原始轨迹（含 tags）
    tool_traces: List[ToolCallResult] = field(default_factory=list)
    rounds_used: int = 0
    notes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VerificationSubtaskReport(SubtaskReport):
    verdict: Optional[bool] = None          # True/False/None
    verify_text: str = ""                # <verify> 内容


@dataclass
class ExecutionReport:
    sequence_id: str
    subtask_reports: List[SubtaskReport]
    meta: Dict[str, Any] = field(default_factory=dict)
    
class SubtaskExecutor:
    def execute(self, *, sequences: List[str], plans: List[Plan]) -> List[ExecutionReport]:
        ...