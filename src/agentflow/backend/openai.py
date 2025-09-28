from typing import List, Tuple, Dict, Any
from logging import Logger
import random
import time

from openai import OpenAI

from concurrent.futures import ThreadPoolExecutor, as_completed

from agentflow.core.interfaces import CanGenerate,SupportChatTemplate
from agentflow.utils.log_util import get_logger
from agentflow.utils.chat_template import is_chat_messages

class OpenaiBackend(CanGenerate):
    """Backend based on Openai API
    """
    
    def __init__(self, config: Dict[str,Any], logger: Logger = None, **kwargs):
        super().__init__()
        self.config = config
        if logger:
            self.logger = logger
        else:
            self.logger = get_logger(config, __name__)
        self._parse_config()

        self.client = OpenAI(
            api_key=self.openai_config["api_key"],
            base_url=self.openai_config["url"],
        )
        self.max_concurrency = self.openai_config.get("max_concurrency",4)
    
    def generate(self, prompts: List, extra: List[Dict] = None, **kwargs) -> Tuple[List[str],List[Dict]]:
        """Generate sequences with gievn prompt list

        Args:
            prompts (List): Prompt list of chat messages or raw str. If raw str are given, they would be wrapped as user prompt for api requests.
            extra (List[Dict], optional): Extra info dicts. Defaults to None.

        Returns:
            Tuple[List[str],List[Dict]]: Generated sequences and any metainfo
                - The metainfo format: {"raw":<full_api_response_object>}
        """
        results = [""] * len(prompts)
        metas = [{}] * len(prompts)
        input_messages = prompts
        if all([isinstance(prompt,str) for prompt in prompts]):
            input_messages = [[{"role":"user","content":prompt}] for prompt in prompts]
        else:
            assert is_chat_messages(input_messages), "Prompt must be str or chat message"
        max_worker = min(len(prompts),self.max_concurrency)
        with ThreadPoolExecutor(max_workers=max_worker) as executor:
            future_to_idx = {
                executor.submit(self._request, messages): idx
                for idx, messages in enumerate(input_messages)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx], metas[idx] = future.result()
                except Exception as e:
                    self.logger.error(e)
                    results[idx]=""
        return results, metas
        
    def _request(self, messages):
        args = {
            "model":self.backend_config["model_path"],
            "messages":messages,
            "temperature": self.sampling_config.get("temperature", 1),
            "max_tokens":self.sampling_config.get("max_tokens",1024),
        }
        
        max_retries = self.openai_config.get("max_retries",2)
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**args)

                return resp.choices[0].message.content, {"raw":resp,"input_messages":messages}

            except Exception as e:
                msg = f"[API attempt {attempt}/{max_retries}] Error: {e}"
                if self.logger:
                    self.logger.warning(msg, exc_info=True)

                if attempt == max_retries:
                    final_msg = f"API Failed after repeating {max_retries} times, aborting."
                    if self.logger:
                        self.logger.error(final_msg)
                    return "", {}

                sleep_time = 1 * (2 ** (attempt - 1))

                jitter = sleep_time * 0.1
                time_to_sleep = sleep_time + (jitter * (2 * random.random() - 1))
                if self.logger:
                    self.logger.info(f"Sleeping {time_to_sleep:.2f}s before retry…")
                time.sleep(time_to_sleep)
                continue
        return "", {}
        
    
    def _parse_config(self):
        backend_config = self.config["backend"]
        sampling_config = backend_config["sampling"]
        openai_config = backend_config["openai"]
        self.backend_name = "openai"
        self.backend_config = backend_config
        self.sampling_config = sampling_config
        self.openai_config = openai_config
        