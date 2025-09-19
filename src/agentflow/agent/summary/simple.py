from __future__ import annotations
from typing import Protocol, List, Tuple, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass

from agentflow.common.messages import Message, trans_messages_to_text
from agentflow.core.interfaces import CanGenerate
from .interface import SummarizerInterface, SummaryItem
from agentflow.utils.chat_template import is_chat_messages




@dataclass
class GeneratorSummarizer(SummarizerInterface):
    """A summarizer based on a llm generation backend
    The prompt_template provided will be formatted with the extra dict when calling summarize method
    The provided text for summarize will be automatically put into extra dict as {"content":text} \n
    Prompt template eg1. \n
    '''Summarize the given input \n
    {content} \n
    ''' \n
    Prompt template eg2. \n
    '''Summarize the given content with context \n
    context:{context} \n
    content: {content} \n
    '''\n
    You are required to provide 'context' key in each input extra dict for eg2.\n
    """
    generator: CanGenerate
    prompt_template: str

    def summarize(self, item: SummaryItem, extra: Dict[str,Any], **kwargs) -> Tuple[str, Dict[str, Any]]:
        outs, metas = self.summarize_batch([item],[extra], **kwargs)
        return outs[0], metas[0]

    def summarize_batch(
        self,
        items: List[SummaryItem],
        extras: List[Dict[str,Any]],
        **kwargs
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        texts: List[str] = []
        for it in items:
            if isinstance(it, str):
                texts.append(it)
            else:
                texts.append(trans_messages_to_text(it))
        for text, extra in zip(texts,extras):
            extra["content"] = text

        prompts = [self.prompt_template.format_map(extra) for extra in extras]
        messages = [[{"role":"user","content":prompt}] for prompt in prompts]
        outputs, gen_metas = self.generator.generate(messages, extra=None, **kwargs)

        metas = [{"summarizer":{"gen_input":m,"gen_results":o}} for m,o in zip(messages,outputs)]
        return outputs, metas