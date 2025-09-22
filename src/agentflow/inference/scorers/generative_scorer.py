from typing import List, Tuple, Dict, Any, Sequence, Callable

from tqdm import tqdm

from agentflow.core.interfaces import CanGenerate, CanRMScores, CanChoiceProbs, SupportChatTemplate
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
        *,
        prob_bs: int = 4,
        choice_labels: Sequence[str] = ("true", "false"), 
        eps: float = 1e-15,   
    ):
        super().__init__()
        self.generator = generator
        self.prob_calculator = prob_calculator
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self.user_prompt = user_prompt or self.DEFAULT_USER
        self.prob_bs = max(1, int(prob_bs))
        self.choice_labels = tuple(choice_labels)
        self.eps = eps
        
        
    def _chunk(self, xs: Sequence[Any], n: int):
        for i in range(0, len(xs), n):
            yield i, xs[i:i+n]

    def _batched_choice_probs(
        self,
        prefixes: Sequence[str],
        labels: Sequence[str],
        **kw
    ) -> List[List[float]]:
        """分批调用 prob_calculator.choice_probs，保持顺序不变。"""
        all_probs: List[List[float]] = []
        for _, pref_chunk in tqdm(self._chunk(prefixes, self.prob_bs),desc="Calculating probs"):
            choices_chunk = [list(labels) for _ in range(len(pref_chunk))]
            probs_chunk = self.prob_calculator.choice_probs(pref_chunk, choices_chunk, **kw)
            probs_chunk = [list(map(float, p)) for p in probs_chunk]
            all_probs.extend(probs_chunk)
        return all_probs
    
    def score(self, sequences: Sequence[str], extra: List[Dict] = None, **kwargs) -> Tuple[List[float],List[Dict]]: 
        msg_list = [
            [{"role":"system","content":self.system_prompt},
            {"role":"user","content":self.user_prompt.format_map({"sequence":seq})}]
            for seq in sequences
        ]
        
        try:
            input_texts = self.generator.apply_chat_template(msg_list)
        except:
            input_texts = ""
        
        outputs, metas = self.generator.generate(msg_list,extra)
        prefixes = []
        invalid_idxs = []
        for idx, out in enumerate(outputs):
            answer_tags = find_tags(out,["answer"])
            if answer_tags:
                target = answer_tags[-1]
                prefix_text = input_texts[idx]+out[:target.start]+"<answer>"
            else:
                prefix_text = "Mock"
                invalid_idxs.append(idx)
            prefixes.append(prefix_text)
        probs = self._batched_choice_probs(prefixes,self.choice_labels)
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
        
        
        