import sys
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)     


from agentflow.backend.vllm import VllmBackend

from agentflow.config import load_config


def test():
    config = load_config("/root/workspace/agent-rm/Agent-Verifier/config/score_vanilla.yaml")
    vllm_engine = VllmBackend(config)
    result, meta = vllm_engine.generate(["What is Generative Pretrained Transformers"])
    print(result)
    
    over_long = "Hello world" * 100000
    result_overlong, meta = vllm_engine.generate([over_long])
    print(result_overlong)
    
if __name__ == "__main__":
    test()