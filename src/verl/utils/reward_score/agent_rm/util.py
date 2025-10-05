import math
from typing import Any, Dict, Optional, List
from sentence_transformers import SentenceTransformer, util

from utils.tag_util import find_tags

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def compute_tool_bonus_old(
    extra_info: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
) -> float:
    """
    依据 BON 准确率（难度）与工具使用次数，给出额外奖励/惩罚（范围受 scale 限制）。
    需要 extra_info 形如：
      {
        "acc": <float in [0,1]>,                     # BON 正确率；越低越难
        "tool_counter": {"search": int, "python": int, ...}
      }
    可用 cfg 覆盖默认超参（见下）。
    """
    if extra_info is None or (not isinstance(extra_info,dict)):
        return 0
   
    conf = {
        "weights": {            # 各工具的“成本权重”
            "search": 5.0,
            "python": 1.0,
            "*": 1.0,          # 未列出工具的默认权重
        },
        "budget": {             # 总体预算 B(d) = b_min + (b_max-b_min)*d
            "min": 0.5,
            "max": 3.0,
        },
        "search_budget": {      # search 专项预算 S(d) = s_min + (s_max-s_min)*d
            "min": 0.0,
            "max": 1.0,
        },
        "alpha": 0.6,           # 预算内奖励强度（难题更高）
        "beta": 0.8,            # 预算外惩罚强度（易题更重）
        "beta_s": 1.0,          # search 超额的额外惩罚
        "gamma": 1.0,           # 惩罚对“容易度”(1-d)的指数缩放
        "scale": 0.3,           # 最终用 tanh 压到 [-scale, +scale]
    }
    _deep_update(conf, cfg or {})

    
    
    acc = extra_info.get("acc", None)
    if acc is None:
        return 0
    d = 0.5 if acc is None else _clamp01(1.0 - _clamp01(float(acc)))  # 难度 d∈[0,1]

    counter = (extra_info.get("tool_counter") or {})
    # 加权成本 C = Σ w_t * n_t
    C = 0.0
    for t, n in counter.items():
        try:
            C += float(conf["weights"].get(t, conf["weights"]["*"])) * float(n)
        except Exception:
            continue


    
    b_min, b_max = float(conf["budget"]["min"]), float(conf["budget"]["max"])
    B = max(0.0, b_min + (b_max - b_min) * d)      

    alpha = float(conf["alpha"])
    beta  = float(conf["beta"])
    gamma = float(conf["gamma"])
    scale = float(conf["scale"])

    # ---------- 奖励/惩罚项 ----------
    # 预算内奖励：最多 α·d，按占比 min(C,B)/B 线性给
    bonus = alpha * (min(C, B) / B) * d if B > 0 else 0.0

    # 预算外惩罚：对易题更重（乘 (1-d)^γ）
    over = max(C - B, 0.0)
    penalty = beta * over * ((1.0 - d) ** gamma)


    extra_raw = bonus - penalty
    extra = scale * math.tanh(extra_raw)  # 平滑压缩，避免盖过基本对错信号

    return float(extra)


def compute_tool_bonus_l0(acc: float,
                  q_list: List[float],
                  s: float = 0.25,
                  lam0: float = 0.5,
                  lam1: float = 0.0) -> float:
    """Compute tool bonus with L0-like formula
    bonus = s * tanh((1+d^2) * sum(q) - n * (lam0 * acc + lam1))
    Args:
        acc (float): N-sampling accuracy
        q_list (List[float]): list of tool-usage quality, each should be in [0,1]
        s (float, optional): clip bonus to [-s,s]. Defaults to 0.25.
        lam0 (float, optional): param for scaling accuracy penalrt. Defaults to 0.25.
        lam1 (float, optional): param for scaling accuracy penalrt. Defaults to 0.0.

    Returns:
        float: bonus reword
    """
    d = _clamp01(1.0 - _clamp01(acc))
    lam = lam0 * (1.0 - d) + lam1
    n = len(q_list)

    raw = (1 + d**2) * sum(_clamp01(q) for q in q_list) - lam * n
    return s * math.tanh(raw)


class QueryQualityScorerChar:
    """
    q = cosine_similarity(query, user_message) * length_penalty_by_char(len(query))

    - 语义相似度：Sentence-Transformers 句向量 + 余弦相似度（映射到[0,1]）
    - 长度惩罚：只惩罚“超过阈值”的字符数，指数衰减，避免一刀切
    """
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        self.model = SentenceTransformer(model_name, device=device)

    @staticmethod
    def length_penalty_by_char(char_len: int, L0: int = 80, tau: int = 40) -> float:
        """
        对“过长”查询做温和惩罚：lp = exp(-max(0, char_len - L0) / tau)
        - L0: 你期望的“上限字符数”（只在超过后才衰减）
        - tau: 衰减速度（越大越温和）
        说明：完全按字符统计，不依赖 tokenizer。
        """
        excess = max(0, int(char_len) - int(L0))
        return math.exp(-excess / float(tau))

    def cosine_similarity01(self, query: str, user_message: str,
                        margin: float = 0.15, gamma: float = 2.0) -> float:
        """
        Sentence-Transformer 句向量 + 余弦；用“间隔+幂”把不相关打到 0，高相似更凸显。
        返回 q∈[0,1]
        """
        emb = self.model.encode([query, user_message], normalize_embeddings=True)
        sim = util.cos_sim(emb[0], emb[1]).item()  # sim ∈ [-1, 1]

        # 带间隔的 ReLU 映射到 [0,1]，再做幂增益
        s = (sim - margin) / (1.0 - margin)
        s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
        q = s ** gamma
        return q

    def score(self,
              query: str,
              user_message: str,
              L0: int = 50,
              tau: int = 30,
              return_parts: bool = False) -> float | Dict[str, Any]:
        """
        返回 q \in [0,1]；如 return_parts=True，返回各项便于调参观测。
        """
        sim01 = self.cosine_similarity01(query, user_message)
        lp    = self.length_penalty_by_char(len(query), L0=L0, tau=tau)
        q     = _clamp01(sim01 * lp)
        if return_parts:
            return {"q": q, "sim01": sim01, "len": len(query), "lp": lp, "L0": L0, "tau": tau}
        return q
    
SCORER = None

def init_scorer():
    global SCORER
    if SCORER is None:
        SCORER = QueryQualityScorerChar(model_name="sentence-transformers/all-MiniLM-L6-v2")

def compute_tool_bonus_old2(
    solution_str: str,
    extra_info: Dict[str, Any],
    scale: float = 0.6,
    lambda_0: float = 0.6,
    lambda_1: float = 0,
) -> float:
    init_scorer()
    search_tags = find_tags(solution_str,allowed_tags=["search"])
    user_message = extra_info.get("user_message")
    accuracy = extra_info.get("acc", None)
    if accuracy is None:
        return 0
    if not user_message or not search_tags:
        return 0
    if isinstance(user_message,dict):
        user_message = user_message["content"]
    assert isinstance(user_message, str)
    queries = [tag.body for tag in search_tags]
    
    q_list = [SCORER.score(queries[i], user_message) for i in range(len(queries))]
    bonus = compute_tool_bonus_l0(acc = accuracy, q_list=q_list, s=scale, lam0=lambda_0, lam1=lambda_1)
    bonus = 0
    return bonus

def compute_tool_bonus(
    solution_str: str,
    extra_info: Dict[str, Any],
    quality_scale: float = 0, # disable
    freq_scale: float = 0.4,
    freq_b: float = 3,
):
    return 0
    quality_bonus = compute_quality_bonus(
        solution_str=solution_str,
        extra_info=extra_info,
        scale=quality_scale,
    )
    freq_bonus = compute_frequency_bonus(
        solution_str=solution_str,
        extra_info=extra_info,
        scale=freq_scale,
        b=freq_b,
    )
    return quality_bonus + freq_bonus

def compute_quality_bonus(
    solution_str: str,
    extra_info: Dict[str, Any],
    scale: float = 0.4,
) -> float:
    """bonus = scale * (2*q_mean -1) (assume q in [0,1])

    Args:
        solution_str (str): _description_
        extra_info (Dict[str, Any]): _description_
        scale (float, optional): _description_. Defaults to 0.4.

    Returns:
        float: bonus
    """
    init_scorer()
    search_tags = find_tags(solution_str,allowed_tags=["search"])
    user_message = extra_info.get("user_message")
    if not user_message or not search_tags:
        return 0
    if isinstance(user_message,dict):
        user_message = user_message["content"]
    assert isinstance(user_message, str)
    queries = [tag.body for tag in search_tags]
    
    q_list = [SCORER.score(queries[i], user_message) for i in range(len(queries))]
    if len(q_list) < 1:
        return 0
    q_mean = sum(q_list) / len(q_list) 
    quality_bonus = scale * (2*q_mean -1)
    return quality_bonus
    
def compute_frequency_bonus(
    solution_str: str,
    extra_info: Dict[str, Any],
    scale: float = 0.4,
    b: float = 3,
) -> float:
    """bonus = -s*tanh(num_search/b - (1-accuracy))

    Args:
        solution_str (str): _description_
        extra_info (Dict[str, Any]): _description_
        scale (float, optional): _description_. Defaults to 0.4.
        b (float, optional): _description_. Defaults to 3.

    Returns:
        float: _description_
    """
    accuracy = extra_info.get("acc", None)
    if accuracy is None:
        return 0
    search_tags = find_tags(solution_str,allowed_tags=["search"])
    if not search_tags:
        return 0
    num_search = len(search_tags)
    d = _clamp01(1-accuracy)
    raw = (num_search / b) - d
    return -scale * math.tanh(raw)

def _test():
    """
    Smoke + sanity tests for:
      - QueryQualityScorerChar.score(...)
      - compute_tool_bonus_l0(...)

    It prints intermediate parts and uses a few asserts that should hold in practice.
    """
    print("=== Smoke test: QueryQualityScorerChar & compute_tool_bonus_l0 ===")

    scorer = SCORER

    # A Chinese user message + two queries: short(good) vs long(very verbose)
    user_msg = "请给我一份关于 Transformer 在时间序列预测中的综述与最新进展"
    q_short  = "Transformer 时间序列 综述 最新进展"
    q_long   = (
        "请检索近年来各种应用中基于Transformer的时间序列预测模型综合综述论文和报告，"
        "并汇总关键方法与实验结果、性能比较和开源实现，最好包含数据集清单与评测细节，谢谢。"
    )
    q_irrel  = "如何用烤箱烤面包"  # 明显不相关

    print("\n-- q(parts) for short/long/irrelevant --")
    parts_short = scorer.score(q_short, user_msg, return_parts=True)
    parts_long  = scorer.score(q_long,  user_msg, return_parts=True)
    parts_irrel = scorer.score(q_irrel, user_msg, return_parts=True)
    for name, parts in [("short", parts_short), ("long", parts_long), ("irrelevant", parts_irrel)]:
        print(f"{name:>11} -> q={parts['q']:.4f}, sim01={parts['sim01']:.4f}, "
              f"len={parts['len']}, lp={parts['lp']:.4f}")

    # --- Range checks
    for x in (parts_short["q"], parts_long["q"], parts_irrel["q"]):
        assert 0.0 <= x <= 1.0, "q must be in [0,1]"

    # --- Length penalty sanity: long query should not score higher than short (same topic)
    assert parts_long["q"] <= parts_short["q"] + 1e-6, \
        "length penalty should reduce q for overly long queries with similar semantics"

    # Build a q_list for "this sample" 
    q_list = [parts_short["q"], parts_long["q"]]


    bonus_easy = compute_tool_bonus_l0(acc=0.90, q_list=q_list, s=0.3, lam0=0.5, lam1=0.0)
    bonus_hard = compute_tool_bonus_l0(acc=0.20, q_list=q_list, s=0.3, lam0=0.5, lam1=0.0)
    print("\n-- bonus by difficulty (same q_list) --")
    print(f"easy (acc=0.90): bonus={bonus_easy:.4f}")
    print(f"hard (acc=0.20): bonus={bonus_hard:.4f}")
    assert bonus_hard >= bonus_easy - 1e-6, \
        "hard problems should incur lighter frequency penalty (bonus_hard >= bonus_easy)"


    bonus_more_calls = compute_tool_bonus_l0(acc=0.90,
                                             q_list=q_list + [parts_short["q"]],
                                             s=0.25, lam0=0.25, lam1=0.0)
    print("\n-- extra: effect of an additional good call (easy case) --")
    print(f"n=2 -> bonus={bonus_easy:.4f} ; n=3 -> bonus={bonus_more_calls:.4f}")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _test()
