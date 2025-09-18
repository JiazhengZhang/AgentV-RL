from __future__ import annotations

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from agentflow.config import load_config

from agentflow.core.interfaces import CanGenerate
from agentflow.inference.samplers.bon import BONSampler
from agentflow.backend.openai import OpenaiBackend
from agentflow.agent.basic import ToolDrivenAgent, AgentContext
from agentflow.tools.caller import ToolCaller
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.parser import TagToolParser
from agentflow.tools.search.base_search import AsyncSearchTool
from agentflow.tools.search.backend.searxng import SearxngBackend
from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.utils.tag_util import find_tags
from agentflow.utils.json_util import JsonUtil
from agentflow.utils.math.answer_parser import evaluate_samples

SAMP_NUM_DICT={
    "easy":4,
    "middle":6,
    "hard":8,
}

SAMPLE_PROMPT="""

"""

VERIFY_SYSTEM_PROMPT="""

"""

VERIFY_USER_PROMPT="""

"""

def to_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        if x == 1: return True
        if x == 0: return False
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "yes", "y", "1"}: return True
        if s in {"false", "f", "no", "n", "0"}: return False
    return None

def _count_correct(evals: List[Dict[str, Any]]) -> int:
    return sum(1 for e in evals if e.get("correct"))

def _has_both(evals: List[Dict[str, Any]]) -> bool:
    c = _count_correct(evals)
    return 0 < c < len(evals)

def _resample_for_target(
    need_correct: bool,
    sampler:BONSampler,              # BONSampler 实例
    base_prompt: str,
    ground_truth: str,
    per_round: int,
    rounds: int = 2,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """定向补采样：仅返回满足目标类别（对 或 错）的样本"""
    out_samps, out_evals = [], []
    old_num = getattr(sampler, "num_samples", None)
    try:
        sampler.num_samples = per_round
        for _ in range(rounds):
            prompt = base_prompt 
            samples, _ = sampler.sample_one(prompt)
            evals = evaluate_samples(samples, ground_truth)
            for s, ev in zip(samples, evals):
                is_ok = bool(ev.get("correct"))
                if (need_correct and is_ok) or (not need_correct and not is_ok):
                    out_samps.append(s)
                    out_evals.append(ev)
            if out_samps:  # 已经补到目标样本就收手
                break
    finally:
        if old_num is not None:
            sampler.num_samples = old_num
    return out_samps, out_evals

def _select_balanced_subset(
    samples: List[str],
    evaluations: List[Dict[str, Any]],
    metas: List[Dict[str, Any]],
    *,
    max_pairs: Optional[int] = None,
    shuffle: bool = True,
) -> List[Dict[str, Any]]:
    """
    从 (samples, evaluations, metas) 里挑一个尽量 1:1 的对/错子集。
    返回的每个元素结构与 all_rollouts 一致：{"rollout","evaluation","meta"}。
    """
    # 建索引
    idx_correct = [i for i, ev in enumerate(evaluations) if ev.get("correct")]
    idx_wrong   = [i for i, ev in enumerate(evaluations) if not ev.get("correct")]

    if shuffle:
        random.shuffle(idx_correct)
        random.shuffle(idx_wrong)

    k = min(len(idx_correct), len(idx_wrong))
    if max_pairs is not None:
        k = min(k, max_pairs)

    if k == 0:
        return []  # 没法配对则留空（不强制）

    # 取前 k 个两类索引，并交叉拼接（对、错、对、错…）
    chosen = []
    for i in range(k):
        ci = idx_correct[i]
        wi = idx_wrong[i]
        chosen.append(ci)
        chosen.append(wi)

    subset = []
    for i in chosen:
        subset.append({
            "rollout": samples[i],
            "evaluation": evaluations[i],
            "meta": metas[i] if i < len(metas) else {},
        })
    return subset

def sample(
    query_file: str,
    backend: "CanGenerate",
    prompt_template: str,
    *,
    question_key: str = "problem",
    answer_key: str = "answer",
    difficulty_key: Optional[str] = None,
    # 补采样参数（沿用你前面的“尽量补到对+错，不强制”策略）
    max_extra_rounds: int = 2,
    # 平衡子集参数
    max_pairs_per_query: Optional[int] = None,  # 每个 query 最多配几对（None 表示尽可能多）
    shuffle_balanced: bool = True,              # 平衡子集是否随机打散挑选
) -> List[Dict[str, Any]]:
    data = JsonUtil.read_jsonlines(query_file)
    sampler = BONSampler(backend=backend, num_samples=16)

    results: List[Dict[str, Any]] = []

    for block in data:
        question = block[question_key]
        ground_truth = block[answer_key]
        diff = block[difficulty_key] if difficulty_key else "middle"
        samp_num = SAMP_NUM_DICT.get(diff, 8)
        sampler.num_samples = samp_num

        base_prompt = prompt_template.format_map({"question": question})

        # 1) 常规采样
        samples, metas = sampler.sample_one(base_prompt)
        evaluations = evaluate_samples(samples, ground_truth)

        # 2) “尽量”补到对+错（不强制）
        if not _has_both(evaluations):
            num_correct = _count_correct(evaluations)
            need_correct = (num_correct == 0)                 # 全错 → 补正确
            need_wrong   = (num_correct == len(evaluations))  # 全对 → 补错误

            for _ in range(max_extra_rounds):
                changed = False

                if need_correct:
                    add_s, add_e = _resample_for_target(
                        need_correct=True,
                        sampler=sampler,
                        base_prompt=base_prompt,
                        ground_truth=ground_truth,
                        per_round=max(4, samp_num // 2),
                        rounds=2,
                    )
                    if add_s:
                        metas.extend([{"fallback": True, "strategy": "resample_for_correct"} for _ in add_s])
                        samples.extend(add_s)
                        evaluations.extend(add_e)
                        changed = True

                if need_wrong:
                    add_s, add_e = _resample_for_target(
                        need_correct=False,
                        sampler=sampler,
                        base_prompt=base_prompt,
                        ground_truth=ground_truth,
                        per_round=max(4, samp_num // 2),
                        rounds=2,
                    )
                    if add_s:
                        metas.extend([{"fallback": True, "strategy": "resample_for_incorrect"} for _ in add_s])
                        samples.extend(add_s)
                        evaluations.extend(add_e)
                        changed = True

                if _has_both(evaluations) or not changed:
                    break

        # 3) 统一长度 & 组装 all_rollouts
        if len(metas) < len(samples):
            metas += [{} for _ in range(len(samples) - len(metas))]

        all_rollouts = []
        for s, ev, mt in zip(samples, evaluations, metas):
            item = {
                "rollout": s,
                "evaluation": ev,
                "meta": mt,
            }
            # 同时保留你之前的字段风格（可选）
            ev["rollout"] = s
            all_rollouts.append(item)

        # 4) 生成尽量 1:1 的 balanced 子集
        balanced_rollouts = _select_balanced_subset(
            samples=samples,
            evaluations=evaluations,
            metas=metas,
            max_pairs=max_pairs_per_query,
            shuffle=shuffle_balanced,
        )

        # 5) 产出该 query 的结果对象（你也可以在此直接写 JSONL）
        result_obj = {
            "question": question,
            "ground_truth": ground_truth,
            "difficulty": diff,
            "all_rollouts": all_rollouts,            # 全量
            "balanced_rollouts": balanced_rollouts,  # 尽量 1:1 的子集
            # 也可附加计数信息，方便检查
            "stats": {
                "num_total": len(all_rollouts),
                "num_correct": _count_correct(evaluations),
                "num_wrong": len(all_rollouts) - _count_correct(evaluations),
                "num_balanced": len(balanced_rollouts),
            },
        }
        results.append(result_obj)

    return results
        
        
def _parse_judge_bool(judge_text: str) -> Optional[bool]:
    """
    解析裁判模型输出中的 <answer>…</answer> 为布尔值。
    要求外层已存在 find_tags/judge 形式。如无/无法解析则返回 None。
    """
    tags = find_tags(judge_text, ["answer"])
    if not tags:
        return None
    try:
        return to_bool(tags[-1].body)
    except Exception:
        return None

def _classify_outcome(judge: Optional[bool], truth: bool) -> str:
    """
    返回 TP/TN/FP/FN/UNK 五类之一。
    judge 为 None 视作 UNK。
    """
    if judge is None:
        return "UNK"
    if judge and truth:
        return "TP"
    if (not judge) and (not truth):
        return "TN"
    if judge and (not truth):
        return "FP"
    if (not judge) and truth:
        return "FN"
    return "UNK"

def agentic_verify(
    agent: "ToolDrivenAgent",
    data: List[Dict],
    user_prompt: str,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    """
    对每个 block（query）：
      - 使用 balanced_rollouts 送审（你已有的 1:1 子集）
      - 解析裁判判断为布尔
      - 生成 per-rollout 的 outcome（TP/TN/FP/FN/UNK）
      - 尽量各挑 1 条 TP 和 1 条 FN（若缺则留空）
      - 汇总：judge_right(做对=TP+TN)、judge_wrong(做错=FP+FN)
      - 保留 all_rollouts（含 judge 原文与 outcome）
    返回一个结果列表，每个元素对应一个 query 的聚合结果。
    """
    results: List[Dict[str, Any]] = []

    for block in data:
        balanced_rollouts = block.get("balanced_rollouts", [])
        # 若你希望“把所有 rollouts 都送审”，可改成 block["all_rollouts"]
        # 这里按你的输入仅审 balanced_rollouts；all_rollouts 仍会完整带回。

        # 1) 组 prompts
        prompts: List[List[Dict[str, str]]] = []
        for ro in balanced_rollouts:
            rendered_user = user_prompt.format_map({})  # 如需带入 rollout 文本，请在模板中占位并传入
            prompts.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rendered_user},
            ])

        # 2) 裁判推理
        judges: List[str] = []
        metas: List[Dict[str, Any]] = []
        if prompts:
            judges, metas = agent.generate(prompts)
        else:
            judges, metas = [], []

        # 3) 汇总每条（balanced）rollout 的评判结果
        per_rollout_records: List[Dict[str, Any]] = []
        tp_sample: Optional[Dict[str, Any]] = None
        fn_sample: Optional[Dict[str, Any]] = None

        judge_right: List[Dict[str, Any]] = []  # TP+TN
        judge_wrong: List[Dict[str, Any]] = []  # FP+FN

        for judge_text, ro, meta in zip(judges, balanced_rollouts, metas):
            truth = bool(to_bool(ro["evaluation"]["correct"]))
            judge_bool = _parse_judge_bool(judge_text)
            outcome = _classify_outcome(judge_bool, truth)

            record = {
                "rollout": ro.get("rollout"),
                "evaluation": ro.get("evaluation"),
                "rollout_meta": ro.get("meta", {}),
                "judge_text": judge_text,
                "judge_choice": judge_bool,        # True/False/None
                "judge_meta": meta,
                "outcome": outcome,                # TP/TN/FP/FN/UNK
            }
            per_rollout_records.append(record)

            # 分类收集
            if outcome in ("TP", "TN"):
                judge_right.append(record)
            elif outcome in ("FP", "FN"):
                judge_wrong.append(record)

            # 选“尽量一个”TP/FN
            if outcome == "TP" and tp_sample is None:
                tp_sample = record
            if outcome == "FN" and fn_sample is None:
                fn_sample = record

        # 4) 产出该 query 的汇总对象
        #    - 保留 all_rollouts：直接带回 block["all_rollouts"]（包含所有生成+评测）
        #    - balanced_judge_records：仅对 balanced_rollouts 做的裁判记录（含 outcome）
        #    - chosen_tp/fn：尽量各一条；若无则为 None
        #    - judge_right/judge_wrong：分开记录
        #    - stats：计数信息
        num_tp = sum(1 for r in per_rollout_records if r["outcome"] == "TP")
        num_tn = sum(1 for r in per_rollout_records if r["outcome"] == "TN")
        num_fp = sum(1 for r in per_rollout_records if r["outcome"] == "FP")
        num_fn = sum(1 for r in per_rollout_records if r["outcome"] == "FN")
        num_unk = sum(1 for r in per_rollout_records if r["outcome"] == "UNK")

        result_obj = {
            # 基本信息（若源 block 有 question/ground_truth/difficulty 则透传）
            "question": block.get("question"),
            "ground_truth": block.get("ground_truth"),
            "difficulty": block.get("difficulty"),

            # 全量保留（用于后续补齐/训练）：这是生成阶段的“所有 rollout”
            "all_rollouts": block.get("all_rollouts", []),

            # 本轮裁判仅针对 balanced_rollouts 的判定与归档
            "balanced_judge_records": per_rollout_records,

            # “尽量一个”的挑选结果（可能为 None）
            "chosen_tp": tp_sample,
            "chosen_fn": fn_sample,

            # 分开记录：做对/做错（便于后续补齐或分析）
            "judge_right": judge_right,  # TP + TN
            "judge_wrong": judge_wrong,  # FP + FN

            "stats": {
                "num_balanced_input": len(balanced_rollouts),
                "num_tp": num_tp,
                "num_tn": num_tn,
                "num_fp": num_fp,
                "num_fn": num_fn,
                "num_unk": num_unk,
                "has_tp": tp_sample is not None,
                "has_fn": fn_sample is not None,
            },
        }

        results.append(result_obj)

    return results

def _read_text(x: Optional[str]) -> str:
    """若 x 指向存在的文件则读文件，否则按字面字符串返回；None -> 空串"""
    if not x:
        return ""
    p = Path(x)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")
    return x

def build_backend(cfg: Dict[str, Any]) -> CanGenerate:
    """
    这里给一个最常见的 OpenAIBackend 示例；你也可以根据 cfg 切换到 vLLM/SGLang 等后端。
    """
    return OpenaiBackend(cfg)

def build_tool_agent(cfg: Dict[str, Any],backend: CanGenerate) -> ToolDrivenAgent:
    """
    一个简易的 ToolDrivenAgent 装配示例（含常用工具注册）。
    你也可以替换为自己工程里更完整的构造函数。
    """
    registry = ToolRegistry()
    registry.register("python", PythonExecutionTool(cfg))
    registry.register("search", AsyncSearchTool(SearxngBackend(cfg)))
    caller = ToolCaller(registry=registry, parser=TagToolParser())
    agent = ToolDrivenAgent(
        backend=backend,
        tool_caller=caller,
        context=AgentContext(),
        config=cfg,
    )
    return agent

def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    JsonUtil.write_jsonlines(str(path), rows)

def main():
    ap = argparse.ArgumentParser(description="One-shot: sampling + judging, save two JSONL files.")
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config for backends/tools.")
    ap.add_argument("--query-file", required=True, help="Input queries JSONL (must contain question/answer keys).")
    ap.add_argument("--prompt-template-file", default=SAMPLE_PROMPT, help="Prompt template file with {question} placeholder.")
    ap.add_argument("--system-prompt", default=VERIFY_SYSTEM_PROMPT, help="System prompt text or file path.")
    ap.add_argument("--user-prompt", default=VERIFY_USER_PROMPT, help="User prompt text or file path (for judge).")
    ap.add_argument("--exp-name", required=True, help="Unified experiment name for output file naming.")
    ap.add_argument("--out-dir", default="outputs", help="Directory to save outputs.")
    # sample 控制
    ap.add_argument("--max-extra-rounds", type=int, default=2, help="Extra resample rounds (best-effort, non-forced).")
    ap.add_argument("--max-pairs-per-query", type=int, default=None, help="Max balanced pairs (None = as many as possible).")
    ap.add_argument("--no-shuffle-balanced", action="store_true", help="Do not shuffle when picking balanced subset.")
    # 可选：覆盖 SAMP_NUM_DICT
    ap.add_argument("--samp-easy", type=int, default=None)
    ap.add_argument("--samp-middle", type=int, default=None)
    ap.add_argument("--samp-hard", type=int, default=None)

    args = ap.parse_args()

    cfg = load_config(args.config)
    prompt_template = Path(args.prompt_template_file).read_text(encoding="utf-8")
    system_prompt = _read_text(args.system_prompt)
    user_prompt = _read_text(args.user_prompt)

    # 可选：动态覆盖采样数（如果你在 your_module 里暴露了 SAMP_NUM_DICT，可以在此修改）
    try:
        if args.samp_easy is not None:
            SAMP_NUM_DICT["easy"] = args.samp_easy
        if args.samp_middle is not None:
            SAMP_NUM_DICT["middle"] = args.samp_middle
        if args.samp_hard is not None:
            SAMP_NUM_DICT["hard"] = args.samp_hard
    except Exception:
        pass  # 忽略：如果你不希望在入口层改全局采样数

    # 1) 采样阶段
    backend = build_backend(cfg)
    sample_results: List[Dict[str, Any]] = sample(
        query_file=args.query_file,
        backend=backend,
        prompt_template=prompt_template,
        max_extra_rounds=args.max_extra_rounds,
        max_pairs_per_query=args.max_pairs_per_query,
        shuffle_balanced=not args.no_shuffle_balanced,
    )

    # 2) 评判阶段（用工具型 Agent）
    judge_agent = build_tool_agent(cfg,backend=backend)
    judge_results: List[Dict[str, Any]] = agentic_verify(
        agent=judge_agent,
        data=sample_results,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )

    # 3) 保存：两份结果分开存
    out_dir = Path(args.out_dir)
    samples_path = out_dir / f"{args.exp_name}.samples.jsonl"
    judges_path = out_dir / f"{args.exp_name}.judges.jsonl"
    save_jsonl(samples_path, sample_results)
    save_jsonl(judges_path, judge_results)

    # 4) 终端友好提示
    num_queries = len(sample_results)
    num_balanced = sum(x.get("stats", {}).get("num_balanced", 0) for x in sample_results)
    print(f"[DONE] Queries: {num_queries} | total balanced items: {num_balanced}")
    print(f"Saved samples → {samples_path}")
    print(f"Saved judges  → {judges_path}")

if __name__ == "__main__":
    main()