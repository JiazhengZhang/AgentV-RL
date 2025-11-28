# agentflow/tools/code/sandbox/python_executor_ray.py
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict

import ray
from ray.exceptions import GetTimeoutError

from .execution_plan import ExecPlan
from .python_sandbox import SandboxConfig, ExecutionResult
from .python_sandbox_ray import PythonSandboxActor


class RayPythonExecutor:
    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        max_workers: int = 1,
        actor: "ray.actor.ActorHandle" = None,
    ):
        self.config = config or SandboxConfig()
        self.max_workers = max_workers
        self._headers: List[str] = []
        self._context: Dict[str, Any] = {}
        self._helpers: Dict[str, Any] = {}
        self._helper_code: List[str] = []
        self._actor = actor


    def set_headers(self, headers: List[str]) -> None:
        self._headers = list(headers)

    def register_header(self, code: str) -> None:
        self._headers.append(code)
        if self._actor is not None:
            ray.get(self._actor.register_header.remote(code))

    def set_context(self, ctx: Dict[str, Any]) -> None:
        self._context.update(ctx)
        if self._actor is not None:
            ray.get(self._actor.update_context.remote(ctx))

    def inject_helpers(self, helpers: Dict[str, Any]) -> None:
        self._helpers.update(helpers)
        if self._actor is not None:
            ray.get(self._actor.register_helpers.remote(helpers))

    def inject_from_code(self, code: str, export=None, alias=None) -> None:
        self._helper_code.append(code)
        if self._actor is not None:
            ray.get(self._actor.register_header.remote(code))


    def _ensure_actor(self):
        if self._actor is not None:
            return
        self._actor = PythonSandboxActor.options(
            num_cpus=1,
            max_restarts=4,
            max_task_retries=-1,
        ).remote(
            config=self.config,
            headers=self._headers + self._helper_code,
            context=self._context,
            helpers=self._helpers,
            helper_code=self._helper_code,
        )

    def shutdown(self):
        if self._actor is not None:
            try:
                ray.kill(self._actor)
            except Exception:
                pass
            self._actor = None


    def run(self, plan: ExecPlan) -> ExecutionResult:
        self._ensure_actor()
        plan_dict = asdict(plan)

        timeout_s = float(self.config.time_limit_s) + 1.0
        try:
            fut = self._actor.run_one.remote(plan_dict)
            res_dict = ray.get(fut, timeout=timeout_s)
        except GetTimeoutError:
            self.shutdown()
            return ExecutionResult(
                ok=False,
                result="",
                stdout="",
                error="Timeout",
                duration_s=self.config.time_limit_s,
            )
        except Exception as e:
            self.shutdown()
            return ExecutionResult(
                ok=False,
                result="",
                stdout="",
                error=f"Exception: {type(e).__name__}: {e}",
                duration_s=None,
            )

        return ExecutionResult(**res_dict)

    def run_many(self, plans: List[ExecPlan], show_progress: bool = False) -> List[ExecutionResult]:
        if not plans:
            return []

        results: List[ExecutionResult] = []

        for plan in plans:
            res = self.run(plan)
            results.append(res)
        return results
