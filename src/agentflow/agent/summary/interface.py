from __future__ import annotations
from typing import Protocol, List, Tuple, Dict, Any, Optional, Union


from agentflow.common.messages import Message


SummaryItem = Union[str, List[Message]]

class SummarizerInterface(Protocol):
    """An interface to summarize the given input
    """

    def summarize(self, item: SummaryItem, extra: Dict,**kwargs) -> Tuple[str, Dict[str, Any]]:
        ...

    def summarize_batch(
        self,
        items: List[SummaryItem],
        extras: List[Dict],
        **kwargs
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        ...