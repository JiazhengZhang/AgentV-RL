# agentflow/agent/executor/interfaces.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from agentflow.agent.planner.interfaces import Plan
from agentflow.tools.base import ToolCallResult
from agentflow.common.messages import Message


@dataclass
class SubtaskReport:
    subtask_id: str
    raw_trace: str                   # 聚合原始轨迹（含 tags）
    tool_traces: List[ToolCallResult] = field(default_factory=list)
    rounds_used: int = 0
    notes: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict):
        tool_traces = []
        raw = data.get("tool_traces",[])
        for trace in raw:
            tool_traces.append(ToolCallResult.from_dict(trace))
        return SubtaskReport(
            subtask_id=data.get("subtask_id"),
            raw_trace=data.get("raw_trace"),
            tool_traces=tool_traces,
            rounds_used=data.get("rounds_used",0),
            notes=data.get("notes",{}),
        )
        
    def to_dict(self) -> Dict:
        return {
            "subtask_id": self.subtask_id,
            "raw_trace": self.raw_trace,
            "rounds_used": self.rounds_used,
            "notes": self.notes,
        }

@dataclass
class VerificationSubtaskReport(SubtaskReport):
    verdict: Optional[bool] = None          # True/False/None
    verify_text: str = ""                # <verify> 内容
    round_messages: List[Message] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict):
        tool_traces = []
        raw = data.get("tool_traces",[])
        for trace in raw:
            tool_traces.append(ToolCallResult.from_dict(trace))
        round_messages_raw = data.get("round_messages",[])
        round_messages = Message.from_dicts(round_messages_raw)
            
        return VerificationSubtaskReport(
            subtask_id=data.get("subtask_id"),
            raw_trace=data.get("raw_trace",""),
            tool_traces=tool_traces,
            rounds_used=data.get("rounds_used",0),
            notes=data.get("notes",{}),
            verdict=data.get("verdict"),
            verify_text=data.get("verify_text",""),
            round_messages = round_messages,
        )
        
    def to_dict(self) -> Dict:
        return {
            "subtask_id": self.subtask_id,
            "raw_trace": self.raw_trace,
            "rounds_used": self.rounds_used,
            "notes": self.notes,
            "verdict": self.verdict,
            "round_messages": self.round_messages,
        }


@dataclass
class ExecutionReport:
    sequence_id: str
    subtask_reports: List[SubtaskReport]
    meta: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict):
        reports = data.get("subtask_reports",[])
        sub_reps = []
        for report in reports:
            sub_reps.append(VerificationSubtaskReport.from_dict(report))
        return ExecutionReport(
            sequence_id=data.get("sequence_id"),
            subtask_reports=sub_reps,
            meta=data.get("meta",{}),
        )
        
    def to_dict(self) -> Dict:
        return {
            "sequence_id": self.sequence_id,
            "subtask_reports": [report.to_dict() for report in self.subtask_reports],
            "meta": self.meta,
        }
    
class SubtaskExecutor:
    def execute(self, *, sequences: List[str], plans: List[Plan]) -> List[ExecutionReport]:
        ...