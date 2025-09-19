import sys
import os
from typing import List, Dict, Tuple, Any, Optional

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..","..",".."))  
sys.path.insert(0, ROOT_DIR)     


from agentflow.agent.summary.simple import GeneratorSummarizer
from agentflow.core.interfaces import CanGenerate



class FakeBackend(CanGenerate):
    """
    规则：
      - 如果初始 user 里含 'NEED_TOOL'，且消息中目前还没有 tool 消息，则返回触发工具的回复：<search>foo</search>
      - 如果历史消息中已经出现过 tool 消息，则返回终止：'Final Answer: from tool'
      - 如果初始 user 含 'FINAL_FIRST'，直接返回：'Final Answer: immediate'
      - 其他情况：返回普通文本 'thinking...'（既无工具也无终止）
    """
    def generate(self, prompts: List, extra: List[Dict] | None = None, **kwargs) -> Tuple[List[str], List[Dict]]:
        outs, metas = [], []
        for messages in prompts:
            # messages 是 list[dict]，按你目前的输入约定
            # 寻找初始 user 内容
            users = [m for m in messages if m.get("role") == "user"]
            user0 = (users[0]["content"] if users else "") if users else ""
            # 是否已有 tool 观察
            has_tool = any(m.get("role") == "tool" for m in messages)

            if "FINAL_FIRST" in user0:
                outs.append("Final Answer: immediate")
            elif "NEED_TOOL" in user0 and not has_tool:
                outs.append("<search>foo</search>")
            elif has_tool:
                outs.append("Final Answer: from tool")
            else:
                outs.append("thinking...")

            metas.append({"fake": True})
        return outs, metas



def test():
    backend = FakeBackend()

    summarizer = GeneratorSummarizer(
        generator=backend, prompt_template="""
You are a summarizer specialized in processing web search results.  
Your role is to extract and organize information that may help a verifier check whether a given answer to a specific question is correct.  

Given qeestion and answer: 

{mission}  

Instructions:
- Do NOT attempt to directly solve or answer the mission yourself.  
- Only summarize or highlight relevant facts, definitions, formulas, examples, or reasoning patterns from the search results that could be useful for verification.  
- If the search results contain conflicting information, point out the differences clearly.  
- Be concise, neutral, and faithful to the source texts.  
- Your output should serve as supporting evidence for a separate verifier, not as a final judgment.  

Output format:
- **Key Facts:** bullet points of extracted facts
- **Potential Conflicts or Variations:** (if any)
- **Useful References:** (optional, if sources mention notable names, theorems, datasets, etc.)

Raw search results:
{content}

"""
    )
    results, metas = summarizer.summarize_batch(["This is a test"],[{"mission":"The mission"}])
    print(results[0])
    print(metas[0])


if __name__ == "__main__":
    test()