# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
import numpy as np

from agentflow.agent.planner.interfaces import Plan, Subtask
from agentflow.agent.executor.interfaces import VerificationSubtaskReport, ExecutionReport

Stage = str
GroupKey = int


@dataclass
class _SubtaskSlotState:
    """单个 subtask 的 best-of 聚合单元"""
    subtask: Subtask
    best_report: Optional[VerificationSubtaskReport] = None
    best_reward: float = float("-inf")
    tried_reports: List[VerificationSubtaskReport] = field(default_factory=list)

    def update(self, rep: VerificationSubtaskReport, reward: Optional[float]) -> None:
        self.tried_reports.append(rep)
        r = float(reward) if reward is not None else float("-inf")
        if r > self.best_reward:
            self.best_reward = r
            self.best_report = rep

    @property
    def covered(self) -> bool:
        return len(self.tried_reports) > 0


@dataclass
class _SamplePlanState:
    """
    单样本的完整 RL 轨迹（与 base_idx 绑定）：
    - plan: 冻结的 Plan
    - slots: subtask 槽位，顺序与 plan.subtasks 一致
    - id2idx: subtask.id -> 槽位索引
    - gt_by_id: 外部真值缓存 {subtask_id: True/False/None}
    - next_ptr: 下一个“尚未覆盖”的 subtask 索引
    - rr_cursor: 复练游标（policy="rr"）
    """
    plan: Plan
    slots: List[_SubtaskSlotState]
    id2idx: Dict[str, int]
    gt_by_id: Dict[str, Optional[bool]] = field(default_factory=dict)
    next_ptr: int = 0
    rr_cursor: int = 0

    def num_subtasks(self) -> int:
        return len(self.slots)

    def covered_count(self) -> int:
        return sum(int(s.covered) for s in self.slots)

    def ready(self) -> bool:
        return self.covered_count() == self.num_subtasks()

    def pop_next_subtask_index(self) -> Optional[int]:
        if self.next_ptr < self.num_subtasks():
            return self.next_ptr
        return None

    def choose_duplicate_index(self, policy: str = "best") -> Optional[int]:
        covered = [i for i, s in enumerate(self.slots) if s.covered]
        if not covered:
            return None
        if policy == "hard":
            return min(covered, key=lambda i: self.slots[i].best_reward)
        if policy == "best":
            return max(covered, key=lambda i: self.slots[i].best_reward)
        # rr
        idx = covered[self.rr_cursor % len(covered)]
        self.rr_cursor += 1
        return idx

    def note_first_cover(self, idx: int) -> None:
        if idx == self.next_ptr:
            while self.next_ptr < self.num_subtasks() and self.slots[self.next_ptr].covered:
                self.next_ptr += 1


@dataclass
class _BatchState:
    """
    一个 group 的缓存桶（不再假定 plan/pop 的长度一致；按 base_idx 分组管理）
    - samples_by_bidx: base_idx -> _SamplePlanState
    - review_cache_by_bidx: base_idx -> ExecutionReport（样本 ready 后缓存）
    - plans_raw/plan_rewards/subtask_labels_raw: 可选，保留原始输入用于日志与选择
    """
    samples_by_bidx: Dict[int, _SamplePlanState] = field(default_factory=dict)
    review_cache_by_bidx: Dict[int, ExecutionReport] = field(default_factory=dict)
    plans_raw: Optional[np.ndarray] = None
    plan_rewards: Optional[np.ndarray] = None
    subtask_labels_raw: Optional[np.ndarray] = None




class AgentRlStageCache:
    """
    关键性质
    ----------
    1) plan put: 支持 B×K（穿插、非简单重复）。按 base_idx 分组、选优(可用 plan_rewards)，为每个 base_idx 冻结一个 Plan。
    2) subtask pop: 输入 base_idx(B,) → 逐元素取样本状态并分配 subtask；若该样本未覆盖用尽则复练（仅复用该样本自身）。
    3) subtask put: 输入任意长度 N（可为 B×K），使用 (base_idx[i], subtask_ids[i]) 精确写回 best-of；完成覆盖即生成 review 缓存。
    4) review pop: 按输入 base_idx(B,) 返回 (plan, exec_report)；仅当该样本 ready 才 valid。
    5) 始终保证 plan.subtasks 与 sub-reports 一一对应（索引由 id2idx 管控）。
    """

    def __init__(self, *, max_num_subtasks: Optional[int] = None, resample_policy: str = "best"):
        assert resample_policy in {"rr", "hard", "best"}
        self.max_num_subtasks = max_num_subtasks
        self.resample_policy = resample_policy
        self._batches: Dict[GroupKey, _BatchState] = {}

    # ---------- utils ----------

    @staticmethod
    def _argsort(a: np.ndarray) -> np.ndarray:
        return np.argsort(a)

    @staticmethod
    def _to_bool_or_none(v: Any) -> Optional[bool]:
        if isinstance(v, bool) or v is None:
            return v
        s = str(v).strip().lower()
        if s in ("true", "t", "yes", "y", "1"):  return True
        if s in ("false", "f", "no", "n", "0"):  return False
        return None

    def _truncate(self, plan: Plan) -> List[Subtask]:
        subs = plan.subtasks
        if self.max_num_subtasks is not None:
            subs = subs[: self.max_num_subtasks]
        return subs



    def put_batch(self, stage: Stage, data: Dict[str, np.ndarray], meta: Dict, **kwargs):
        """
        通用约定：data 必含 'base_idx'
        - stage == "plan":
            data["plans"]           : object[np.ndarray] (len=N)  N=B×K ；元素可为 Plan / List[Plan] / None
            data["base_idx"]        : int[np.ndarray]    (len=N)  允许重复（GRPO 扩张）
            data["plan_rewards"]    : float[np.ndarray]  (len=N)  可选；用于组内选优
            data["subtask_labels"]  : object[np.ndarray] (len=N)  可选；每元素为 {subtask_id: bool|None}
            ——> 按 base_idx 分组，组内选一份 plan 冻结；subtask_labels 组内合并（同 id 取首个非 None）
        - stage == "subtask":
            data["reports"]         : object[np.ndarray] (len=N)  任意长度 N（B 或 B×K）
            data["rewards"]         : float[np.ndarray]  (len=N)  可选；best-of 聚合用
            data["subtask_ids"]     : object[np.ndarray] (len=N)  与 data["base_idx"] 一一对应
        """
        gid: int = int(meta["group_id"])
        if stage == "plan":
            plans   = data["plans"]            # [N]
            bidx    = data["base_idx"]         # [N] 允许重复
            scores  = data.get("plan_rewards", None)     # [N] or None
            labels  = data.get("subtask_labels", None)   # [N] or None
            N = int(plans.shape[0])
            assert bidx.shape[0] == N
            if scores is not None: assert scores.shape[0] == N
            if labels is not None: assert labels.shape[0] == N

            bs = _BatchState(
                plans_raw=plans.copy(),
                plan_rewards=(scores.copy() if scores is not None else None),
                subtask_labels_raw=(labels.copy() if labels is not None else None),
            )

            # 1) 分组：base_idx -> 候选条目索引列表
            groups: Dict[int, List[int]] = {}
            for i in range(N):
                bi = int(bidx[i])
                groups.setdefault(bi, []).append(i)

            # 2) 组内选优、冻结为 _SamplePlanState
            for bi, idx_list in groups.items():
                # 2.1 选择一个候选 plan
                choice_i = idx_list[0]
                if scores is not None:
                    def _score(t): 
                        v = scores[t]
                        try:
                            return float(v) if v is not None else -1e30
                        except Exception:
                            return -1e30
                    choice_i = max(idx_list, key=_score)

                chosen = plans[choice_i]
                # 若是 List[Plan]，默认取第 0 个（也可在上游先挑好再传入）
                if isinstance(chosen, (list, tuple)):
                    chosen = chosen[0] if len(chosen) > 0 else None
                if not isinstance(chosen, Plan):
                    # 找不到可用 plan：该样本跳过，后续 pop 时 valid=False
                    continue

                # 2.2 合并该组 subtask 的外部真值（同 id 取首个非 None）
                gt_map: Dict[str, Optional[bool]] = {}
                if labels is not None:
                    for t in idx_list:
                        lab = labels[t]
                        if isinstance(lab, dict):
                            for sid, val in lab.items():
                                sid = str(sid)
                                if sid not in gt_map or gt_map[sid] is None:
                                    gt_map[sid] = self._to_bool_or_none(val)

                # 2.3 冻结
                subs  = self._truncate(chosen)
                slots = [_SubtaskSlotState(s) for s in subs]
                id2idx = {s.id: i for i, s in enumerate(subs)}
                bs.samples_by_bidx[bi] = _SamplePlanState(
                    plan=chosen, slots=slots, id2idx=id2idx, gt_by_id=gt_map
                )

            self._batches[gid] = bs
            return

        # 非 plan：需已有 bucket
        bs = self._batches.get(gid)
        if bs is None:
            return

        if stage == "subtask":
            reps   = data["reports"]
            bidx_p = data["base_idx"]
            sids   = data["subtask_ids"]
            rws    = data.get("rewards", None)
            N = int(bidx_p.shape[0])
            assert reps.shape[0] == N and sids.shape[0] == N
            if rws is not None: assert rws.shape[0] == N

            # 支持 B 或 B×K：逐条对齐 (base_idx, subtask_id)
            for i in range(N):
                bi  = int(bidx_p[i])
                sid = sids[i]
                if sid is None:
                    continue
                sp = bs.samples_by_bidx.get(bi)
                if sp is None:
                    continue
                sid = str(sid)
                idx = sp.id2idx.get(sid, None)
                if idx is None:
                    # 写回的 subtask_id 不属于该样本当前 plan，忽略
                    continue

                rw = float(rws[i]) if rws is not None else None
                sp.slots[idx].update(reps[i], rw)
                sp.note_first_cover(idx)

                # 样本 ready → 一次性构建执行报告缓存（严格按 plan.subtasks 顺序）
                if sp.ready() and bi not in bs.review_cache_by_bidx:
                    ordered: List[VerificationSubtaskReport] = []
                    ok = True
                    for slot in sp.slots:
                        if slot.best_report is None:
                            ok = False; break
                        ordered.append(slot.best_report)
                    if ok:
                        bs.review_cache_by_bidx[bi] = ExecutionReport(
                            sequence_id=f"{gid}:{bi}",
                            subtask_reports=ordered,
                            meta={"group_id": gid, "base_idx": bi},
                        )
            return

        if stage == "review" or stage == "overall":
            return

        raise ValueError(f"Invalid stage: {stage}")

 

    def pop_batch(self, stage: Stage, meta: Dict, *, base_idx: np.ndarray) -> Dict[str, np.ndarray]:
        """
        所有返回数组与传入 base_idx 一一对齐（长度=B）：
        - stage=="plan"   ：返回 {}
        - stage=="subtask"：
            {
              "subtasks":     object[np.ndarray[Subtask]],
              "subtask_ids":  object[np.ndarray[str]],
              "subtask_gt":   object[np.ndarray[object]],  # True/False/None
              "is_duplicate": np.ndarray[bool],
              "valid_mask":   np.ndarray[bool],
              "plans":        object[np.ndarray[Plan]],
            }
        - stage=="review" ：
            {
              "plans":        object[np.ndarray[Plan]],
              "execution_reports": object[np.ndarray[ExecutionReport]],
              "valid_mask":   np.ndarray[bool],
            }
        """
        gid: int = int(meta["group_id"])
        B = int(base_idx.shape[0])

        if stage == "plan":
            return {}

        bs = self._batches.get(gid)
        if bs is None:
            if stage == "subtask":
                return {
                    "subtasks": np.array([None] * B, dtype=object),
                    "subtask_ids": np.array([None] * B, dtype=object),
                    "subtask_gt": np.array([None] * B, dtype=object),
                    "is_duplicate": np.zeros(B, dtype=bool),
                    "valid_mask": np.zeros(B, dtype=bool),
                    "plans": np.array([None] * B, dtype=object),
                }
            elif stage == "review":
                return {
                    "plans": np.array([None] * B, dtype=object),
                    "execution_reports": np.array([None] * B, dtype=object),
                    "valid_mask": np.zeros(B, dtype=bool),
                }
            elif stage == "overall":
                return {}
            raise ValueError(f"Invalid stage: {stage}")

        if stage == "subtask":
            subtasks = np.empty(B, dtype=object); subtasks.fill(None)
            sids     = np.empty(B, dtype=object); sids.fill(None)
            gts      = np.empty(B, dtype=object); gts.fill(None)
            dup      = np.zeros(B, dtype=bool)
            valid    = np.zeros(B, dtype=bool)
            plans    = np.empty(B, dtype=object); plans.fill(None)

            # 逐元素（按传入 base_idx 的顺序）调度
            for j in range(B):
                bi = int(base_idx[j])
                sp = bs.samples_by_bidx.get(bi)
                if sp is None or sp.num_subtasks() == 0:
                    continue
                plans[j] = sp.plan

                nxt = sp.pop_next_subtask_index()
                if nxt is None:
                    idx = sp.choose_duplicate_index(self.resample_policy)
                    if idx is None:
                        continue
                    dup[j] = True
                else:
                    idx = nxt

                st = sp.slots[idx].subtask
                sid = st.id
                subtasks[j] = st
                sids[j] = sid
                gts[j]  = sp.gt_by_id.get(sid, None)
                valid[j] = True

            return {
                "subtasks": subtasks,
                "subtask_ids": sids,
                "subtask_gt": gts,
                "is_duplicate": dup,
                "valid_mask": valid,
                "plans": plans,
            }

        if stage == "review":
            plans = np.empty(B, dtype=object); plans.fill(None)
            reps  = np.empty(B, dtype=object); reps.fill(None)
            valid = np.zeros(B, dtype=bool)

            for j in range(B):
                bi = int(base_idx[j])
                sp = bs.samples_by_bidx.get(bi)
                if sp is None or not sp.ready():
                    continue
                plans[j] = sp.plan
                rep = bs.review_cache_by_bidx.get(bi)
                if rep is None:
                    # 兜底重组（严格按 plan.subtasks 顺序）
                    ordered: List[VerificationSubtaskReport] = []
                    ok = True
                    for slot in sp.slots:
                        if slot.best_report is None:
                            ok = False; break
                        ordered.append(slot.best_report)
                    if not ok:
                        continue
                    rep = ExecutionReport(
                        sequence_id=f"{gid}:{bi}",
                        subtask_reports=ordered,
                        meta={"group_id": gid, "base_idx": bi},
                    )
                    bs.review_cache_by_bidx[bi] = rep
                reps[j] = rep
                valid[j] = True

            return {"plans": plans, "execution_reports": reps, "valid_mask": valid}

        if stage == "overall":
            return {}
        
        raise ValueError(f"Invalid stage: {stage}")



    def ready_for_review(self, group_id: int) -> bool:
        bs = self._batches.get(int(group_id))
        if bs is None:
            return False
        for sp in bs.samples_by_bidx.values():
            if sp is None or not sp.ready():
                return False
        return True

    def clear(self):
        self._batches.clear()
