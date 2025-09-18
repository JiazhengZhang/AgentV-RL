from __future__ import annotations
from typing import Protocol, List, Tuple, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass

from agentflow.common.messages import Message, trans_messages_to_text
from agentflow.core.interfaces import CanGenerate
from .interface import SummarizerInterface, SummaryItem
from agentflow.utils.chat_template import is_chat_messages




@dataclass
class GeneratorSummarizer(SummarizerInterface):
    generator: CanGenerate
    prompt_template: str

    def summarize(self, item: SummaryItem, meta: Dict[str,Any], **kwargs) -> Tuple[str, Dict[str, Any]]:
        outs, metas = self.summarize_batch([item],[meta], **kwargs)
        return outs[0], metas[0]

    def summarize_batch(
        self,
        items: List[SummaryItem],
        metas: List[Dict[str,Any]],
        **kwargs
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        texts: List[str] = []
        for it in items:
            if isinstance(it, str):
                texts.append(it)
            else:
                texts.append(trans_messages_to_text(it))
        for text, meta in zip(texts,metas):
            meta["content"] = text

        prompts = [self.prompt_template.format_map(meta) for meta in metas]
        messages = [[{"role":"user","content":prompt}] for prompt in prompts]
        outputs, gen_metas = self.generator.generate(messages, extra=None, **kwargs)

        metas = [{"generator_meta": m} for m in gen_metas]
        return outputs, metas