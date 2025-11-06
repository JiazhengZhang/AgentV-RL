# agentflow/execution/executor.py
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pebble import ProcessPool
from concurrent.futures import TimeoutError as PebbleTimeoutError
from functools import partial
from multiprocessing import get_context
from tqdm import tqdm

from .python_sandbox import (
    SandboxConfig, SandboxRuntime,
    ExecutionResult, _run_in_sandbox
)
from .execution_plan import ExecPlan



class PythonExecutor:
    def __init__(self,
                 config: Optional[SandboxConfig] = None,
                 max_workers: Optional[int] = None):
        self.config = config or SandboxConfig()
        self.max_workers = max_workers or 1
        self._headers: List[str] = []
        self._context: Dict[str, Any] = {}
        self._helpers: Dict[str, Any] = {}
        self._helper_code: List[str] = []

    def set_headers(self, headers: List[str]) -> None:
        self._headers = list(headers)

    def register_header(self, code: str) -> None:
        self._headers.append(code)

    def set_context(self, ctx: Dict[str, Any]) -> None:
        self._context.update(ctx)

    def inject_from_module(self, module: str,
                           names: Optional[List[str]] = None,
                           alias: Optional[Dict[str,str]] = None) -> None:
        helpers = SandboxRuntime.load_helpers_from_module(module, names=names, alias=alias)
        self._helpers.update(helpers)

    def inject_from_code(self, code: str,
                         export: Optional[List[str]] = None,
                         alias: Optional[Dict[str,str]] = None) -> None:
        self._helper_code.append(code)

    def inject_helpers(self, helpers: Dict[str, Any]) -> None:
        self._helpers.update(helpers)

    def run(self, plan: ExecPlan) -> ExecutionResult:
        ctx = get_context("spawn")
        with ProcessPool(max_workers=1, max_tasks=1, context = ctx) as pool:
            fn = partial(_run_in_sandbox,
                         code=plan.code,
                         capture_mode=plan.capture_mode,
                         answer_symbol=plan.answer_symbol,
                         answer_expr=plan.answer_expr,
                         config=self.config,
                         headers=self._headers+self._helper_code,
                         context=self._context,
                         helpers=self._helpers)
            future = pool.schedule(fn, timeout=self.config.time_limit_s)
            try:
                return future.result()
            except PebbleTimeoutError:
                return ExecutionResult(ok=False, result="", stdout="", error="Timeout", duration_s=self.config.time_limit_s)
            except Exception as e:
                return ExecutionResult(ok=False, result="", stdout="", error=f"{type(e).__name__}", duration_s=self.config.time_limit_s)

    def run_many(self, plans: List[ExecPlan], show_progress: bool=False) -> List[ExecutionResult]:
        if not plans:
            return []
        outs: List[ExecutionResult] = [None] * len(plans)
        ctx = get_context("spawn")
        with ProcessPool(max_workers=min(self.max_workers, len(plans)), context=ctx, max_tasks=128) as pool:
            fn_base = partial(_run_in_sandbox,
                         config=self.config,
                         headers=self._headers+self._helper_code,
                         context=self._context,
                         helpers=self._helpers,
                         limit_resource = True)
            futures: List[Tuple[int, Any]] = []
            for idx, p in enumerate(plans):
                try:
                    fn = partial(
                        fn_base,
                        code=p.code,
                        capture_mode=p.capture_mode,
                        answer_symbol=p.answer_symbol,
                        answer_expr=p.answer_expr
                    )
                    fut = pool.schedule(fn, timeout=self.config.time_limit_s)
                    futures.append((idx, fut))
                except Exception as e:
                    outs[idx] = ExecutionResult(
                        ok=False, result="", stdout="",
                        error=f"ScheduleFailed: {type(e).__name__}: {e}",
                        duration_s=None
                    )

            use_pbar = show_progress
            pbar = tqdm(total=len(plans), desc="Execute") if use_pbar else None

            for idx, fut in futures:
                try:
                    r = fut.result()
                except PebbleTimeoutError:
                    r = ExecutionResult(
                        ok=False, result="", stdout="",
                        error="Timeout",
                        duration_s=self.config.time_limit_s
                    )
                except Exception as e:
                    r = ExecutionResult(
                        ok=False, result="", stdout="",
                        error=f"Exception: {type(e).__name__}: {e}",
                        duration_s=None
                    )
                outs[idx] = r
                if pbar: pbar.update(1)
            if pbar: pbar.close()

        return [
            x if x is not None else ExecutionResult(ok=False, result="", stdout="", error="UnknownError", duration_s=None)
            for x in outs
        ]

