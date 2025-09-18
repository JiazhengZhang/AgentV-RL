from typing import List, Tuple, Dict, Any, Sequence, Callable

from agentflow.core.interfaces import CanGenerate, CanRMScores, CanChoiceProbs
from agentflow.utils.tag_util import find_tags

class SupportLogitsScore(CanGenerate,CanChoiceProbs):
    ...

class BoolLogitsGenerativeScorer(CanRMScores):
    """A scorer based on bool-logits probability.The scorer first use generator to conduct 
    LLM-as-Judge, then calclulate the log probability of producing "true" or "false" answer to give the final score
    """
    
    DEFAULT_SYSTEM = """
Given a question with an answer to it, you are required to think step by stepand judge whether the answer is correct
If is correct, finally output <answer>true</answer>, otherwise <answer>false</answer>
"""
    
    DEFAULT_USER = """The sequence for judge:
{sequence}
Your judgement:
"""
    
    def __init__(
        self, 
        generator: CanGenerate,   
        prob_calculator: CanChoiceProbs,
        system_prompt: str = None,
        user_prompt: str = None,
    ):
        super().__init__()
        self.generator = generator
        self.prob_calculator = prob_calculator
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self.user_prompt = user_prompt or self.DEFAULT_USER
    
    def score(self, sequences: Sequence[str], extra: List[Dict] = None, **kwargs) -> Tuple[List[float],List[Dict]]: 
        msg_list = [
            [{"role":"system","content":self.system_prompt},
            {"role":"user","content":self.user_prompt.format_map({"sequence":seq})}]
            for seq in sequences
        ]
        outputs, metas = self.generator.generate(msg_list,extra)
        prefixes = []
        invalid_idxs = []
        for idx, out in enumerate(outputs):
            answer_tags = find_tags(out)
            if answer_tags:
                target = answer_tags[-1]
                prefix_text = out[:target.start]+"<answer>"
            else:
                prefix_text = "Mock"
                invalid_idxs.append(idx)
            prefixes.append(prefix_text)
        probs = self.prob_calculator.choice_probs(prefixes,[["true","false"] for _ in range(len(prefixes))])
        results: List[float] = []
        for idx, prob in enumerate(probs):
            if idx in invalid_idxs:
                results.append(-1)
            else:
                prob_true = prob[0]
                prob_false = prob[1]
                results.append((prob_true)/max((prob_true+prob_false),1e-15))
        final_metas = []
        for ou, ms, meta in zip(outputs,msg_list,metas):
            meta.update({"raw_text":ou,"input_messages":ms})
            final_metas.append(meta)
        return results, final_metas
        
        
        