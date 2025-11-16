import ray
from typing import Any, Dict, List, Optional
from dataclasses import asdict

from .python_sandbox import (
    SandboxConfig, ExecutionResult, _run_in_sandbox
)
from .execution_plan import ExecPlan


@ray.remote(
    num_cpus=1,
    max_restarts=16,
    max_task_retries=2,
)
class PythonSandboxActor:
    def __init__(
        self,
        config: Dict[str, Any],
        headers: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        helpers: Optional[Dict[str, Any]] = None,
        helper_code: Optional[List[str]] = None,
    ):
        self.config = SandboxConfig(**config)
        self.headers = list(headers or [])
        self.context = dict(context or {})
        self.helpers = dict(helpers or {})
        self.helper_code = list(helper_code or [])


    def update_context(self, ctx: Dict[str, Any]) -> None:
        self.context.update(ctx)

    def register_header(self, code_piece: str) -> None:
        self.headers.append(code_piece)

    def register_helpers(self, helpers: Dict[str, Any]) -> None:
        self.helpers.update(helpers)


    def run_one(self, plan_dict: Dict[str, Any]) -> Dict[str, Any]:
        plan = ExecPlan(**plan_dict)

        res: ExecutionResult = _run_in_sandbox(
            code=plan.code,
            capture_mode=plan.capture_mode,
            answer_symbol=plan.answer_symbol,
            answer_expr=plan.answer_expr,
            config=self.config,
            headers=self.headers + self.helper_code,
            context=self.context,
            helpers=self.helpers,
            limit_resource=True,
        )
        return asdict(res)

    def run_many(self, plan_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for pd in plan_dicts:
            plan = ExecPlan(**pd)
            res: ExecutionResult = _run_in_sandbox(
                code=plan.code,
                capture_mode=plan.capture_mode,
                answer_symbol=plan.answer_symbol,
                answer_expr=plan.answer_expr,
                config=self.config,
                headers=self.headers + self.helper_code,
                context=self.context,
                helpers=self.helpers,
                limit_resource=True,
            )
            out.append(asdict(res))
        return out
