import sys
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)     

from agentflow.backend.vllm_logits import VllmChoiceLogitsBackend

from agentflow.config import load_config
from agentflow.utils.tag_util import find_tags


def test():
    config = load_config("/root/workspace/agent-rm/Agent-Verifier/config/score_vanilla.yaml")
    vllm_engine = VllmChoiceLogitsBackend(config)
    PROMPT = """
    Answer the given question and put the result in <answer>...</answer> tag
    You are requred to output your reasoning trace
    Task: judge if the given statement is wrong or right.
    If is right, answer true, otherwise false
    Statement: The value of e is approximitly -1
    """
    input_ids = vllm_engine.apply_chat_template([{"role":"user","content":PROMPT}])
    result, meta = vllm_engine.generate([input_ids])
    print(result[0])
    answer_tags = find_tags(result[0],["answer"])
    tag = answer_tags[0]
    pos = tag.start
    prefix_text = result[0][:pos]+f"<answer>"
    print(f"prefix: {prefix_text}")
    probs = vllm_engine.choice_probs([prefix_text],[["true","false"]])
    print(probs)
    
    over_long = "Hello world" * 100000
    result_overlong, meta = vllm_engine.generate([over_long])
    print(result_overlong)

if __name__ == "__main__":
    test()