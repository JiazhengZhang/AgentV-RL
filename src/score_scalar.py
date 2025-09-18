# rm_scoring.py
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from agentflow.backend.hf_scalar_rm import HFRMBackend
from agentflow.config import load_config
from agentflow.utils.json_util import JsonUtil

DEFAULT_SEQ_TEMPLATE="""
### Problem ###
{problem}

### Solution ###
{solution}
"""

def ensure_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl_stream(path: str, *, max_records: Optional[int] = None):
    """逐行读取 JSONL，生成 (idx, obj)。"""
    with open(path, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield count, json.loads(line)
            count += 1
            if max_records is not None and count >= max_records:
                break


def to_text(x: Any) -> str:
    """样本可能是 dict 或 str，这里尽量取文本。"""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        # 常见字段兜底
        for k in ("text", "output", "answer", "content"):
            if k in x:
                v = x[k]
                return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def try_apply_chat_template(
    rm: HFRMBackend,
    prompt_text: str,
    response_text: str,
) -> Optional[str]:
    """尽量用 RM 的 chat template，失败返回 None（由上层回退朴素拼接）。"""
    try:
        if not hasattr(rm, "apply_chat_template"):
            return None
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ]
        rendered = rm.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if isinstance(rendered, str) and rendered.strip():
            return rendered
        # 某些实现可能返回 token ids 等，这里仅处理 str，其他情况交回退逻辑
        return None
    except Exception:
        return None


def build_sequences_for_block(
    block: Dict[str, Any],
    *,
    join_template: str,
) -> List[str]:
    """
    把一条记录 (含 prompt 与 samples[*]) 转为若干 judge 'sequence' 文本：
    默认格式： "User: {problem}\nAssistant: {solution}"
    """
    prompt_text = block.get("question", "")
    samples = block.get("samples", []) or []
    seqs: List[str] = []
    for samp in samples:
        resp = to_text(samp)
        seq = join_template.format(problem=prompt_text, solution=resp)
        seqs.append(seq)
    return seqs


def flush_batch_and_write(
    rm: HFRMBackend,
    batch_blocks: List[Dict[str, Any]],
    batch_seqs_per_block: List[List[str]],
    output_path: str,
    *,
    first_write_mode: str,
) -> str:
    """
    将一个“记录批次”的所有序列拉平，调用 rm.score，然后按块拆回并写出。
    返回下一次写入应使用的文件模式（一般切到 'a'）。
    """
    # 拉平成一个序列列表，并记录每块的长度
    flat: List[str] = []
    lens: List[int] = []
    for seqs in batch_seqs_per_block:
        lens.append(len(seqs))
        flat.extend(seqs)

    scores: List[float] = []
    if flat:
        scores, metas = rm.score(flat)

    # 回切回每块，并写回 JSONL
    offset = 0
    for block, L in zip(batch_blocks, lens):
        block_out = dict(block)  # 避免污染原块
        evaluations: List[Dict] = block_out.get("evaluations",None)
        if not evaluations:
            evaluations = [{}] * L
        if L == 0:
            block_scores: List[float] = []
            block_metas = []
        else:
            block_scores = scores[offset: offset + L]
            block_metas = metas[offset : offset + L]
        offset += L

        for eva, meta, score in zip(evaluations,block_metas,block_scores):
            eva["score"]=score
            
        block_out["evaluations"] = evaluations
        block_out["metas"] = block_metas
        
        # 可选：选出最佳样本（若存在）
        if block_scores:
            best_idx = max(range(len(block_scores)), key=lambda i: block_scores[i])
            block_out["best_index"] = best_idx
            block_out["best_score"] = float(block_scores[best_idx])
            # 同时给出最佳样本文本（若原始有 samples）
            try:
                block_out["best_sample"] = to_text(block_out["samples"][best_idx])
            except Exception:
                pass
        else:
            block_out["best_index"] = None
            block_out["best_score"] = None

        JsonUtil.write_jsonlines(output_path, block_out, mode=first_write_mode)
        if first_write_mode == "w":
            first_write_mode = "a"

    return "a"


def score_streaming(
    config_path: str,
    input_path: str,
    output_path: str,
    *,
    batch_size: int,
    append: bool,
    use_chat_template: bool,
    join_template: str,
    max_records: Optional[int],
):
    ensure_parent_dir(output_path)

    # 初始化 RM
    config = load_config(config_path)
    rm = HFRMBackend(config)

    # 写入模式：是否先清空文件
    write_mode = "a" if append else "w"
    if not append:
        JsonUtil.write_jsonlines(output_path, [], mode="w")  # 清空

    batch_blocks: List[Dict[str, Any]] = []
    batch_seqs_per_block: List[List[str]] = []
    total = 0

    for _, block in read_jsonl_stream(input_path, max_records=max_records):
        # 从一条记录构建该记录的所有“prompt+response”序列（一个 block 多个序列）
        seqs = build_sequences_for_block(
            block,
            use_chat_template=use_chat_template,
            join_template=join_template,
            rm=rm if use_chat_template else None,
        )
        batch_blocks.append(block)
        batch_seqs_per_block.append(seqs)

        # 凑满一个记录批次则评分并写出
        if len(batch_blocks) >= batch_size:
            write_mode = flush_batch_and_write(
                rm,
                batch_blocks,
                batch_seqs_per_block,
                output_path,
                first_write_mode=write_mode,
            )
            total += len(batch_blocks)
            batch_blocks.clear()
            batch_seqs_per_block.clear()

    # 处理尾批
    if batch_blocks:
        write_mode = flush_batch_and_write(
            rm,
            batch_blocks,
            batch_seqs_per_block,
            output_path,
            first_write_mode=write_mode,
        )
        total += len(batch_blocks)

    print(f"[DONE] Scored {total} records → {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Scalar RM scoring (streaming JSONL)")
    p.add_argument("--config", required=True, type=str, help="Path to backend config for RM")
    p.add_argument("--input", required=True, type=str, help="Input JSONL produced by sampling (with 'prompt' and 'samples')")
    p.add_argument("--output", required=True, type=str, help="Output JSONL with scores")
    p.add_argument("--batch-size", type=int, default=32, help="How many records (questions) per scoring batch")
    p.add_argument("--append", action="store_true", help="Append to output instead of overwrite")
    p.add_argument("--use-chat-template", action="store_true", help="Use RM's chat template to join prompt+response")
    p.add_argument("--join-template", type=str, default=DEFAULT_SEQ_TEMPLATE, help="Fallback join format when not using chat template or when template fails")
    p.add_argument("--max-records", type=int, default=None, help="Only process first N records")
    return p.parse_args()


def main():
    args = parse_args()
    score_streaming(
        config_path=args.config,
        input_path=args.input,
        output_path=args.output,
        batch_size=max(1, int(args.batch_size)),
        append=bool(args.append),
        use_chat_template=bool(args.use_chat_template),
        join_template=args.join_template,
        max_records=args.max_records,
    )


if __name__ == "__main__":

    main()
