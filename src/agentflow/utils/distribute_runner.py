from __future__ import annotations
import os
import queue
import time
import torch
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue, get_context
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Protocol

from agentflow.utils.log_util import get_logger

class Initializer(Protocol):
    def __call__(self, worker_id: int, /, **context: Any) -> Any: ...

class BatchExecutor(Protocol):
    def __call__(self, state: Any, /, *, batch: Any) -> Sequence[Any]: ...



@dataclass
class _Task:
    index: int
    payload: Any

@dataclass
class BatchResult:
    index: int
    ok: bool
    data: Optional[Sequence[Any]] = None
    error: Optional[str] = None

def _entrypoint(
    worker_id: int,
    tq: Queue,
    rq: Queue,
    stop,
    initializer: Initializer,
    ctx: Dict[str, Any],
    executor: BatchExecutor,
    worker_gpu_set: List[int]
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in worker_gpu_set)
        state = initializer(worker_id, **ctx)
            
        while not stop.is_set():
            try:
                task: Optional[_Task] = tq.get(timeout=0.2)
            except queue.Empty:
                continue
            if task is None:
                break
            try:
                out = executor(state, batch=task.payload)
                if not isinstance(out, Sequence):
                    raise TypeError("executor must return a Sequence")
                rq.put(BatchResult(index=task.index, ok=True, data=out))
            except Exception:
                rq.put(BatchResult(index=task.index, ok=False, error=traceback.format_exc()))
    except Exception:
        rq.put(BatchResult(index=-1, ok=False, error=f"[FATAL worker_id={worker_id}]\n{traceback.format_exc()}"))
        
        
def resolve_visible_devices() -> List[int]:
    """Parse CUDA_VISIBLE_DEVICES env into a list of ints.
    If not set, use all visible GPUs from nvidia-smi or assume [0].
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw.strip() == "":
        return [0]
    ids = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        try:
            ids.append(int(token))
        except ValueError:
            pass
    return ids


def split_even(gpus: List[int], num_workers: int, gpus_per_worker: int) -> List[List[int]]:
    if gpus_per_worker <= 0:
        raise ValueError("gpus_per_worker must be positive")
    need = num_workers * gpus_per_worker
    if len(gpus) != need:
        raise RuntimeError(f"Require exactly {need} GPUs, but got {len(gpus)}.")
    sets = []
    for w in range(num_workers):
        s = w * gpus_per_worker
        e = s + gpus_per_worker
        sets.append(gpus[s:e])  
    return sets

class OrderedProcessPool:
    def __init__(
        self,
        *,
        num_workers: int,
        initializer: Initializer,
        executor: BatchExecutor,
        initializer_context: Optional[Dict[str, Any]] = None,
        start_method: str = "spawn",
        queue_capacity: Optional[int] = None,
        daemon: bool = False,
        gpus_per_worker: int = 1, 
    ) -> None:
        self._ctx = get_context(start_method)
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        self._initializer = initializer
        self._executor = executor
        self._initializer_context = dict(initializer_context or {})
        cap = queue_capacity or max(8 * num_workers, 8)
        self._tq: Queue = self._ctx.Queue(maxsize=cap)
        self._rq: Queue = self._ctx.Queue(maxsize=12)
        self._stop = self._ctx.Event()
        self._ps: List[Process] = []
        self._submitted = 0
        self._closed = False
        self.logger = get_logger(name = __name__)
        visible = resolve_visible_devices()
        worker_gpu_sets = split_even(visible, num_workers, gpus_per_worker)
        
        for wid in range(num_workers):
            p = self._ctx.Process(
                target=_entrypoint,
                args=(wid, self._tq, self._rq, self._stop, self._initializer, self._initializer_context, self._executor, worker_gpu_sets[wid]),
                daemon=daemon,
            )
            p.start()
            self._ps.append(p)

    def submit_batches(self, batches: Sequence[Any]) -> List[int]:
        if self._closed:
            raise RuntimeError("pool is closed")
        idxs: List[int] = []
        self.logger.info(f"Submit batch of length {len(batches)}.")
        for payload in batches:
            i = self._submitted
            self._submitted += 1
            self._tq.put(_Task(index=i, payload=payload))
            idxs.append(i)
        return idxs

    def iterate(self, total: int, *, timeout_seconds: Optional[float] = None, poll_seconds: float = 0.5) -> Iterator[BatchResult]:
        next_idx = self._submitted - total
        pending: Dict[int, BatchResult] = {}
        got = 0
        start = time.time()
        while got < total:
            for wid, p in enumerate(self._ps):
                if not p.is_alive():
                    raise RuntimeError(f"[pool] worker#{wid} pid={p.pid} died exit={p.exitcode}; results still pending")
            try:
                res: BatchResult = self._rq.get(timeout=poll_seconds)
                self.logger.info(f"Task of index {res.index} finished.")
            except queue.Empty:
                if timeout_seconds and (time.time() - start) > timeout_seconds:
                    raise TimeoutError("iterate timed out")
                continue
            if res.index == -1 and not res.ok:
                raise RuntimeError(res.error or "worker fatal error")
            pending[res.index] = res
            while next_idx in pending:
                out = pending.pop(next_idx)
                got += 1
                yield out
                next_idx += 1

    def map(
        self,
        items: Sequence[Any],
        *,
        batch_size: int,
        pack: Optional[Callable[[Sequence[Any]], Any]] = None,
        unpack: Optional[Callable[[Sequence[Any]], Sequence[Any]]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> List[Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        do_pack = pack or (lambda xs: list(xs))
        do_unpack = unpack or (lambda ys: ys)
        slices: List[Tuple[int, int, Any]] = []
        n = len(items)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            slices.append((s, e, do_pack(items[s:e])))
        self.submit_batches([p for (_, _, p) in slices])
        out: List[Any] = [None] * n  # type: ignore[list-item]
        cur = 0
        for res in self.iterate(len(slices), timeout_seconds=timeout_seconds):
            s, e, _ = slices[cur]
            cur += 1
            if not res.ok:
                raise RuntimeError(res.error or f"batch {res.index} failed")
            data = list(do_unpack(res.data or []))
            if len(data) != (e - s):
                raise ValueError(f"executor returned {len(data)} for batch size {e - s}")
            out[s:e] = data
        return out

    def close(self, *, wait: bool = True, timeout_seconds: float = 10.0) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._ps:
            self._tq.put(None)
        self._stop.set()
        if wait:
            for p in self._ps:
                p.join(timeout=timeout_seconds)

    def __enter__(self) -> "OrderedProcessPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
