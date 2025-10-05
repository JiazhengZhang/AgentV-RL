import sys
import os
import argparse
import random
from typing import Dict, List, Any

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT_DIR)

from agentflow.utils.json_util import JsonUtil  # noqa: E402


def select_tp_fn(evals: List[Dict[str, Any]], tp_limit: int = 2, fn_limit: int = 2):
    tp_ids = []
    fn_ids = []
    random.shuffle(evals)
    for ev in evals:
        sid = ev.get("sampling_id")
        if sid is None:
            continue
        if ev.get("correct") is True:
            if len(tp_ids) < tp_limit:
                tp_ids.append(sid)
        elif ev.get("correct") is False:
            if len(fn_ids) < fn_limit:
                fn_ids.append(sid)
        if len(tp_ids) >= tp_limit and len(fn_ids) >= fn_limit:
            break
    return tp_ids, fn_ids


def filter_block(block: Dict[str, Any], tp_limit: int = 2, fn_limit: int = 2) -> Dict[str, Any]:
    """返回过滤后的新 block：samples/evaluations 只保留选中项，其余进 meta"""
    samples: List[str] = block.get("samples", [])
    evaluations: List[Dict[str, Any]] = block.get("evaluations", [])

    # 选择采样 id
    tp_ids, fn_ids = select_tp_fn(evaluations, tp_limit=tp_limit, fn_limit=fn_limit)
    selected_ids = tp_ids + fn_ids
    selected_id_set = set(selected_ids)

    # 过滤 evaluations，保持原顺序但只保留选中的 sampling_id
    new_evals: List[Dict[str, Any]] = []
    for ev in evaluations:
        sid = ev.get("sampling_id")
        if sid in selected_id_set:
            new_evals.append(ev)

    # 过滤 samples：按 selected_ids 的顺序收集，注意越界保护
    new_samples: List[str] = []
    for sid in selected_ids:
        if isinstance(sid, int) and 0 <= sid < len(samples):
            new_samples.append(samples[sid])

    # 组织 meta：把原来的正确率相关和一些原始统计放进去
    meta: Dict[str, Any] = {
        "original_num_samples": len(samples),
        "original_num_evaluations": len(evaluations),
        "original_num_correct": block.get("num_correct"),
        "original_any_correct": block.get("any_correct"),
        "original_accuracy": block.get("accuracy"),
        "selected_indices": {"tp": tp_ids, "fn": fn_ids},
    }

    # 构造输出 block：保留指定键；替换 samples/evaluations；放入 meta；移除顶层正确率相关键
    out: Dict[str, Any] = {}
    for k in ["id", "question", "answer", "difficulty", "prompt"]:
        if k in block:
            out[k] = block[k]

    out["samples"] = new_samples
    out["evaluations"] = new_evals
    out["meta"] = meta

    return out


def main():
    parser = argparse.ArgumentParser(description="Filter TP/FN samples per block.")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL path")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL path")
    parser.add_argument("--tp", type=int, default=2, help="Max number of true-positive samples")
    parser.add_argument("--fn", type=int, default=2, help="Max number of false-negative samples")
    args = parser.parse_args()

    data: List[Dict[str, Any]] = JsonUtil.read_jsonlines(args.input)
    out_blocks: List[Dict[str, Any]] = []
    print(len(data))
    for block in data:
        out_blocks.append(filter_block(block, tp_limit=args.tp, fn_limit=args.fn))

    JsonUtil.write_jsonlines(args.output, out_blocks)


if __name__ == "__main__":
    main()
