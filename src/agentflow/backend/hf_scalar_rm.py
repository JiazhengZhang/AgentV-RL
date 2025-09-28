# src/backends/hf_rm_backend.py
from typing import Sequence, List, Dict, Any, Optional, Union,Tuple
from logging import Logger
import torch
from transformers import AutoTokenizer, AutoModel

from agentflow.utils.log_util import get_logger
from agentflow.core.interfaces import SupportChatTemplate, CanRMScores
from agentflow.utils.chat_template import is_chat_messages, safe_apply_chat_template, ChatTemplateDefaultsMixin, left_truncate_text_by_token, resolve_context_window_len


class HFRMBackend(ChatTemplateDefaultsMixin, SupportChatTemplate,CanRMScores):

    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        super().__init__()
        self.logger: Logger = logger if logger else get_logger(config, __name__)

        backend_config: Dict[str, Any] = config.get("backend", {})
        rm_config: Dict[str, Any] = backend_config.get("rm", {})
        sampling_config = backend_config.get("sampling", {})

        self.model_path: str = backend_config.get("model_path", "")
        self.device: str = rm_config.get("device", "cuda")
        self.torch_dtype_name: str = backend_config.get("dtype", "auto")
        self.max_length: int = int(sampling_config.get("max_length", 1024))
        self.padding_side: str = rm_config.get("padding_side", "right")
        self.truncation_side: str = rm_config.get("truncation_side", "left")  

        self.temperature: float = float(sampling_config.get("temperature", 1.0))
        self.apply_sigmoid: bool = bool(rm_config.get("apply_sigmoid", True))
        clip_range = rm_config.get("clip_range", [0.0, 1.0])
        self.clip_low: float = float(clip_range[0])
        self.clip_high: float = float(clip_range[1])

        self.batch_size: int = int(rm_config.get("batch_size", -1))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True, trust_remote_code=True
        )
        self.tokenizer.padding_side = self.padding_side
        try:
            self.tokenizer.truncation_side = self.truncation_side
        except Exception:
            pass

        torch_dtype = "auto" if self.torch_dtype_name == "auto" else getattr(torch, self.torch_dtype_name)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

        self._cfg_num_labels = getattr(self.model.config, "num_labels", None)
        self.logger.info(
            f"[HFRMBackend] (scalar-only) config.num_labels={self._cfg_num_labels}, "
            f"padding_side={self.padding_side}, truncation_side={self.truncation_side}"
        )

    def apply_chat_template(self, messages: List[Dict[str,str]], 
                            tokenize=False, 
                            add_generation_prompt=True, 
                            **additional_params) -> Union[str,Any]:
        merged = {**self._chat_template_defaults, **additional_params}
        result, _ = safe_apply_chat_template(
            self.tokenizer,
            messages=messages,
            tokenize = tokenize,
            add_generation_prompt = add_generation_prompt,
            explicit_max_model_len=resolve_context_window_len(self.model,self.tokenizer),
            generation_max_new_tokens=32,
            **merged
        )

        return result

    def _try_remote_scores(self, batch_texts: List[str]) -> Optional[torch.Tensor]:
        model = self.model
        has_get_scores = hasattr(model, "get_scores")
        has_get_score = hasattr(model, "get_score")
        if not (has_get_scores or has_get_score):
            return None

        try:
            chats = [[{"role": "user", "content": t}] for t in batch_texts]
            if has_get_scores:
                scores = model.get_scores(self.tokenizer, chats)  
            else:
                scores = [model.get_score(self.tokenizer, c) for c in chats]
            scores_t = torch.tensor(scores, device=self.device, dtype=torch.float32)
    
            if self.temperature != 1.0:
                scores_t = scores_t / max(1e-6, self.temperature)
            if self.apply_sigmoid:
                scores_t = torch.sigmoid(scores_t)
            scores_t = scores_t.clamp(self.clip_low, self.clip_high)
            return scores_t
        except Exception as e:
            self.logger.warning(f"[HFRMBackend] remote get_scores failed, fallback to logits. err={e}")
            return None

    @torch.no_grad()
    def score(
        self,
        sequences: Sequence[str],
        extra: List[Dict[str, Any]] | None = None,
        **kwargs: Any
    ) -> Tuple[List[float],List[Dict]]:

        if not sequences:
            return []

        normalized_scores: List[float] = []

        batch_size = self.batch_size if (self.batch_size and self.batch_size > 0) else len(sequences)
        for start in range(0, len(sequences), batch_size):
            batch_texts = list(sequences[start: start + batch_size])

            remote_scores = None
            if remote_scores is not None:
                normalized_scores.extend(remote_scores.detach().float().cpu().tolist())
                continue

            encodings = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            out = self.model(**encodings)
            logits = getattr(out, "logits", None)
            if logits is None:
                if isinstance(out, (list, tuple)) and len(out) > 0:
                    logits = out[0]
                else:
                    raise RuntimeError("[HFRMBackend] Model forward has no 'logits' field.")

            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)  

            if logits.size(-1) != 1:

                raise RuntimeError(
                    f"[HFRMBackend] Expect scalar RM with logits[...,1], but got shape {list(logits.shape)}. "
                    f"Please load a regression RM (num_labels=1)."
                )

            raw = logits.squeeze(-1)  
            if self.temperature != 1.0:
                raw = raw / max(1e-6, self.temperature)
            scores = torch.sigmoid(raw) if self.apply_sigmoid else raw
            scores = scores.clamp(self.clip_low, self.clip_high)

            normalized_scores.extend(scores.detach().float().cpu().tolist())

        return normalized_scores, [{}] * len(normalized_scores)
