from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict

from .registry import ToolRegistry
from .base import ToolCallRequest, ToolParser, ToolCallResult

class ToolCaller:
    """Wrapper class for conveniently call multiple tools with given text
    """
    def __init__(self, registry: ToolRegistry, parser: ToolParser):
        self.registry = registry
        self.parser = parser
        
    def call_batch(self, texts: List[str], metas: List[Dict]=None, **kwargs) -> List[List[ToolCallResult]]:
        """Call tools with batch input texts

        Args:
            texts (List[str]): Texts that contains tool-call symbols
            metas (List[Dict], optional): Meta info that contains. Defaults to None.

        If any meta contains a dict with structure {"tool_name":int}, tool quota will be applied to the corresponding tool.
        Raises:
            RuntimeError: When parsed tools do not exist in registry

        Returns:
            List[List[ToolCallResult]]: tool call results
        """
        if not metas:
            metas = [None]*len(texts)
        calls_per_text = self.parser.parse_batch(texts, metas)
        grouped: DefaultDict[str, List[Tuple[int,ToolCallRequest]]] = defaultdict(list)
        for tid, calls in enumerate(calls_per_text):
            for c in calls:
                grouped[c.name].append((tid, c))
                
        result_grid: List[Dict[int, ToolCallResult]] = [dict() for _ in range(len(texts))]
                
        for tool_name, items in grouped.items():
            tool = self.registry.get(tool_name)
            if tool is None:
                raise RuntimeError(f"Tool with tag {tool_name} does not exist")
            indices = [text_idx for (text_idx, _) in items]
            calls = [call for (_,call) in items]
            results = tool.run_batch(calls)
            
            for (text_idx, call), result in zip(items, results):
                result_grid[text_idx][call.index] = result
        
        final_results: List[List[ToolCallResult]] = []
        for tid, calls in enumerate(calls_per_text):
            if not calls:
                final_results.append([])  
                continue
            by_idx = result_grid[tid]
            ordered = [by_idx[i] for i in sorted(by_idx.keys())]
            final_results.append(ordered)
        return final_results
    
    def call_single(self, text: str, meta: Dict=None) -> List[ToolCallResult]:
        """Call tool for a single input
        If meta contains a dict with structure {"tool_name":int}, tool quota will be applied to the corresponding tool.
        """
        return self.call_batch([text],[meta])[0]