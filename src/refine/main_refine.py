# refine_main.py

import argparse
from typing import List, Dict, Any, Tuple

import ray

from agentflow.config import load_config
from agentflow.utils.json_util import JsonUtil
from agentflow.utils.log_util import get_logger
from agentflow.utils.fs_util import ensure_parent_dir
from agentflow.utils.math.answer_parser import grade_answer_verl

from .refine_worker import CandidateActor, VerifierActor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Refine pipeline (candidate+verifier multi-round, two-phase loading, per-round outputs + metrics)"
    )
    p.add_argument("--candidate-config", required=True, type=str,
                   help="Config file for candidate backend.")
    p.add_argument("--verifier-config", required=True, type=str,
                   help="Config file for verifier backend.")

    p.add_argument("--input", required=True, type=str,
                   help="Input jsonl, each line must contain at least {'question': ..., 'ground_truth': ...}.")
    p.add_argument("--output", required=True, type=str,
                   help="Final merged output jsonl with refine history and final answers.")

    p.add_argument("--round-output-dir", required=True, type=str,
                   help="Directory to store per-round jsonl outputs.")
    p.add_argument("--exp-name", required=True, type=str,
                   help="Experiment name for naming per-round files.")

    p.add_argument("--metrics-output-dir", required=True, type=str,
                   help="Directory to store per-round metrics json.")

    p.add_argument("--candidate-model-path", type=str, default=None)
    p.add_argument("--verifier-model-path", type=str, default=None)

    p.add_argument("--num-candidate-workers", type=int, default=2)
    p.add_argument("--num-verifier-workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-refine-rounds", type=int, default=3)

    p.add_argument("--verifier-type", type=str, default="forward")

    p.add_argument("--enable-thinking-candidate", action="store_true")
    p.add_argument("--enable-thinking-verifier", action="store_true")

    p.add_argument("--candidate-temperature", type=float, default=None)
    p.add_argument("--candidate-top-p", type=float, default=None)
    p.add_argument("--candidate-max-new-tokens", type=int, default=None)

    p.add_argument("--verifier-temperature", type=float, default=None)
    p.add_argument("--verifier-top-p", type=float, default=None)
    p.add_argument("--verifier-max-new-tokens", type=int, default=None)

    p.add_argument("--ray-address", type=str, default=None)

    return p.parse_args()


def build_gen_kwargs(temperature, top_p, max_new_tokens) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_new_tokens is not None:
        kwargs["max_tokens"] = max_new_tokens
    return kwargs


def schedule_batches_balanced(
    actors: List["ray.actor.ActorHandle"],
    batch_payloads: List[List[Dict[str, Any]]],
    submit_fn,  # (actor, payload) -> ObjectRef
) -> List[List[Dict[str, Any]]]:
    if not batch_payloads:
        return []

    num_actors = len(actors)
    num_batches = len(batch_payloads)

    results: List[List[Dict[str, Any]]] = [None] * num_batches

    inflight: Dict["ray.ObjectRef", Tuple[int, int]] = {}
    next_batch = 0

    for actor_idx in range(min(num_actors, num_batches)):
        payload = batch_payloads[next_batch]
        obj = submit_fn(actors[actor_idx], payload)
        inflight[obj] = (actor_idx, next_batch)
        next_batch += 1

    while inflight:
        ready, _ = ray.wait(list(inflight.keys()), num_returns=1)
        if not ready:
            continue

        obj_done = ready[0]
        actor_idx, batch_idx = inflight.pop(obj_done)

        batch_result = ray.get(obj_done)
        results[batch_idx] = batch_result

        if next_batch < num_batches:
            payload = batch_payloads[next_batch]
            obj = submit_fn(actors[actor_idx], payload)
            inflight[obj] = (actor_idx, next_batch)
            next_batch += 1

    return results


def main():
    args = parse_args()

    candidate_config = load_config(args.candidate_config)
    verifier_config = load_config(args.verifier_config)

    if args.candidate_model_path:
        candidate_config["backend"]["model_path"] = args.candidate_model_path
    if args.verifier_model_path:
        verifier_config["backend"]["model_path"] = args.verifier_model_path

    logger = get_logger(verifier_config, __name__)

    ensure_parent_dir(args.output)
    records: List[Dict[str, Any]] = JsonUtil.read_jsonlines(args.input)
    for i, rec in enumerate(records):
        if "idx" not in rec:
            rec["idx"] = i

    logger.info(f"Loaded {len(records)} records from {args.input}")

    state: List[Dict[str, Any]] = []
    for rec in records:
        if "ground_truth" not in rec:
            ground_truth = rec.get("answer", "")
            rec["ground_truth"] = ground_truth
            if not ground_truth:
                logger.warning(f"Record idx={rec.get('idx')} has no 'ground_truth' key")

        initial_answer = None
        refine_rounds = rec.get("refine_rounds")
        if isinstance(refine_rounds, list) and len(refine_rounds) > 0:
            first_round = refine_rounds[0] or {}
            initial_answer = first_round.get("answer")

        s = {
            "idx": rec["idx"],
            "question": rec.get("question", ""),
            "raw": rec,
            "rounds": [],
            "answer": None,
            "label": None,
            "status": "active",
            "initial_answer": initial_answer,
        }
        state.append(s)

    idx2pos: Dict[int, int] = {s["idx"]: i for i, s in enumerate(state)}

    ray.init(
        address=args.ray_address or None,
        include_dashboard=False,
    )
    logger.info("Ray initialized for refine pipeline.")

    candidate_gen_kwargs = build_gen_kwargs(
        args.candidate_temperature,
        args.candidate_top_p,
        args.candidate_max_new_tokens,
    )
    verifier_gen_kwargs = build_gen_kwargs(
        args.verifier_temperature,
        args.verifier_top_p,
        args.verifier_max_new_tokens,
    )

    num_candidates = max(1, args.num_candidate_workers)
    num_verifiers = max(1, args.num_verifier_workers)
    batch_size = max(1, args.batch_size)

    try:
        for round_id in range(args.max_refine_rounds):
            logger.info(f"Refine Round {round_id} / {args.max_refine_rounds}")

            active_indices = [i for i, s in enumerate(state) if s["status"] == "active"]
            if not active_indices:
                logger.info("All samples are already done, stop refining.")
                break

            logger.info(f"Round {round_id}: {len(active_indices)} active samples")

            logger.info(f"Round {round_id}: starting candidate phase")
            if round_id == 0:
                indices_need_candidate = []
                for si in active_indices:
                    s = state[si]
                    init_ans = s.get("initial_answer")
                    if init_ans:
                        while len(s["rounds"]) <= round_id:
                            s["rounds"].append({
                                "answer": None,
                                "cand_correct": None,
                                "cand_grade": None,
                                "label": None,
                                "feedback": None,
                                "process": None,
                            })
                        s["rounds"][round_id]["answer"] = init_ans

                        gt = s["raw"].get("ground_truth", "")
                        grade_info = grade_answer_verl(init_ans, gt)
                        cand_correct = bool(grade_info.get("correct", False))
                        s["rounds"][round_id]["cand_correct"] = cand_correct
                        s["rounds"][round_id]["cand_grade"] = grade_info
                    else:
                        indices_need_candidate.append(si)
            else:
                indices_need_candidate = list(active_indices)

            cand_batch_payloads: List[List[Dict[str, Any]]] = []
            for start in range(0, len(indices_need_candidate), batch_size):
                end = min(start + batch_size, len(indices_need_candidate))
                batch_idxs = indices_need_candidate[start:end]
                payload: List[Dict[str, Any]] = []
                for si in batch_idxs:
                    s = state[si]
                    item = {
                        "idx": s["idx"],
                        "question": s["question"],
                    }
                    if round_id > 0:
                        prev_round = s["rounds"][-1]
                        item["prev_answer"] = prev_round.get("answer", "")
                        item["feedback"] = prev_round.get("feedback", "")
                    payload.append(item)
                if payload:
                    cand_batch_payloads.append(payload)
            if cand_batch_payloads:
                candidate_actors = [
                    CandidateActor.remote(
                        candidate_config,
                        enable_thinking=args.enable_thinking_candidate,
                    )
                    for _ in range(num_candidates)
                ]

                def cand_submit_fn(actor, payload):
                    if round_id == 0:
                        return actor.generate_initial_batch.remote(payload, **candidate_gen_kwargs)
                    else:
                        return actor.generate_refine_batch.remote(payload, **candidate_gen_kwargs)

                cand_results_batches = schedule_batches_balanced(
                    actors=candidate_actors,
                    batch_payloads=cand_batch_payloads,
                    submit_fn=cand_submit_fn,
                )

                for batch in cand_results_batches:
                    for item in batch:
                        idx = item["idx"]
                        pos = idx2pos[idx]
                        s = state[pos]

                        while len(s["rounds"]) <= round_id:
                            s["rounds"].append({
                                "answer": None,
                                "cand_correct": None,
                                "cand_grade": None,
                                "label": None,
                                "feedback": None,
                                "process": None,
                            })

                        ans = item["answer"]
                        s["rounds"][round_id]["answer"] = ans

                        gt = s["raw"].get("ground_truth", "")
                        grade_info = grade_answer_verl(ans, gt)
                        cand_correct = bool(grade_info.get("correct", False))
                        s["rounds"][round_id]["cand_correct"] = cand_correct
                        s["rounds"][round_id]["cand_grade"] = grade_info

                for act in candidate_actors:
                    ray.kill(act, no_restart=True)

            logger.info(f"Round {round_id}: candidate phase finished.")

            logger.info(f"Round {round_id}: starting verifier phase")

            active_indices = [i for i, s in enumerate(state) if s["status"] == "active"]
            if not active_indices:
                logger.info("No active samples before verifier phase, break.")
                break

            verifier_actors = [
                VerifierActor.remote(
                    verifier_config,
                    verifier_type=args.verifier_type,
                    enable_thinking=args.enable_thinking_verifier,
                )
                for _ in range(num_verifiers)
            ]

            ver_batch_payloads: List[List[Dict[str, Any]]] = []
            for start in range(0, len(active_indices), batch_size):
                end = min(start + batch_size, len(active_indices))
                batch_idxs = active_indices[start:end]
                payload: List[Dict[str, Any]] = []
                for si in batch_idxs:
                    s = state[si]
                    cur_round = s["rounds"][round_id]
                    ans = cur_round.get("answer")
                    if not ans:
                        continue
                    payload.append({
                        "idx": s["idx"],
                        "question": s["question"],
                        "answer": ans,
                    })
                if payload:
                    ver_batch_payloads.append(payload)

            def ver_submit_fn(actor, payload):
                return actor.evaluate_batch.remote(payload, **verifier_gen_kwargs)

            ver_results_batches = schedule_batches_balanced(
                actors=verifier_actors,
                batch_payloads=ver_batch_payloads,
                submit_fn=ver_submit_fn,
            )

            for batch in ver_results_batches:
                for item in batch:
                    idx = item["idx"]
                    pos = idx2pos[idx]
                    s = state[pos]
                    cur_round = s["rounds"][round_id]

                    cur_round["label"] = item["label"]
                    cur_round["feedback"] = item["feedback"]
                    cur_round["process"] = item.get("process", None)

            for act in verifier_actors:
                ray.kill(act, no_restart=True)

            logger.info(f"Round {round_id}: verifier phase finished.")


            active_indices = [i for i, s in enumerate(state)]

            cand_correct_count = 0
            cand_total_count = 0

            tp = tn = fp = fn = 0
            uncertain_count = 0
            ver_total = 0

            prev_correct_count = prev_wrong_count = 0
            wrong_to_correct = correct_to_wrong = 0

            for i in active_indices:
                s = state[i]
                if round_id >= len(s["rounds"]):
                    continue
                rinfo = s["rounds"][round_id]
                cand_corr = rinfo.get("cand_correct")
                if cand_corr is not None:
                    cand_total_count += 1
                    if cand_corr:
                        cand_correct_count += 1

                label = rinfo.get("label")
                if label is not None:
                    ver_total += 1
                    pred_pos = (label == "correct")
                    gt_pos = bool(cand_corr)

                    if label == "uncertain":
                        uncertain_count += 1

                    if gt_pos and pred_pos:
                        tp += 1
                    elif (not gt_pos) and (not pred_pos):
                        tn += 1
                    elif (not gt_pos) and pred_pos:
                        fp += 1
                    elif gt_pos and (not pred_pos):
                        fn += 1

                if round_id > 0 and round_id - 1 < len(s["rounds"]):
                    prev_corr = s["rounds"][round_id - 1].get("cand_correct")
                    if prev_corr is not None and cand_corr is not None:
                        if prev_corr:
                            prev_correct_count += 1
                        else:
                            prev_wrong_count += 1

                        if (not prev_corr) and cand_corr:
                            wrong_to_correct += 1
                        if prev_corr and (not cand_corr):
                            correct_to_wrong += 1

            cand_accuracy = float(cand_correct_count) / cand_total_count if cand_total_count > 0 else 0.0
            ver_accuracy = float(tp + tn) / ver_total if ver_total > 0 else 0.0

            wrong_to_correct_rate = (
                float(wrong_to_correct) / prev_wrong_count if prev_wrong_count > 0 else 0.0
            )
            correct_to_wrong_rate = (
                float(correct_to_wrong) / prev_correct_count if prev_correct_count > 0 else 0.0
            )

            metrics = {
                "round": round_id,
                "num_samples_with_answer": cand_total_count,
                "candidate": {
                    "correct": cand_correct_count,
                    "wrong": cand_total_count - cand_correct_count,
                    "accuracy": cand_accuracy,
                },
                "candidate_transition": {
                    "has_prev_round": round_id > 0,
                    "prev_correct": prev_correct_count,
                    "prev_wrong": prev_wrong_count,
                    "correct_after_refine": cand_correct_count,
                    "wrong_after_refine": cand_total_count - cand_correct_count,
                    "wrong_to_correct": wrong_to_correct,
                    "wrong_to_correct_rate": wrong_to_correct_rate,
                    "correct_to_wrong": correct_to_wrong,
                    "correct_to_wrong_rate": correct_to_wrong_rate,
                },
                "verifier": {
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "uncertain": uncertain_count,
                    "total": ver_total,
                    "accuracy": ver_accuracy,
                },
            }

            metrics_path = f"{args.metrics_output_dir.rstrip('/')}/{args.exp_name}_metrics_round_{round_id:02d}.json"
            ensure_parent_dir(metrics_path)
            JsonUtil.write_json(metrics_path, metrics, mode="w")
            logger.info(f"Round {round_id}: wrote metrics to {metrics_path}")

            round_path = f"{args.round_output_dir.rstrip('/')}/{args.exp_name}_round_{round_id:02d}.jsonl"
            ensure_parent_dir(round_path)
            round_records: List[Dict[str, Any]] = []
            for s in state:
                rec = dict(s["raw"])
                rec["idx"] = s["idx"]
                if s["answer"] is not None:
                    rec["answer"] = s["answer"]
                    rec["label"] = s["label"]
                elif round_id < len(s["rounds"]):
                    rec["answer"] = s["rounds"][round_id].get("answer")
                    rec["label"] = s["rounds"][round_id].get("label")
                else:
                    rec["answer"] = None
                    rec["label"] = None

                rec["refine_rounds"] = s["rounds"]
                round_records.append(JsonUtil.json_sanitize(rec))
            JsonUtil.write_jsonlines(round_path, round_records, mode="w")
            logger.info(f"Round {round_id}: wrote round file {round_path} ({len(round_records)} records)")

        for s in state:
            if s["answer"] is None and s["rounds"]:
                last = s["rounds"][-1]
                s["answer"] = last.get("answer")
                s["label"] = last.get("label")

        out_lines: List[Dict[str, Any]] = []
        for s in state:
            rec = dict(s["raw"])
            rec["idx"] = s["idx"]
            rec["answer"] = s["answer"]
            rec["label"] = s["label"]
            rec["refine_rounds"] = s["rounds"]
            out_lines.append(JsonUtil.json_sanitize(rec))

        JsonUtil.write_jsonlines(args.output, out_lines, mode="w")
        logger.info(f"[DONE] Wrote {len(out_lines)} refined records to {args.output}")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
