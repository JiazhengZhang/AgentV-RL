# agentflow/tools/python_execution_tool.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Any, Dict, List, Optional
from dataclasses import asdict

from agentflow.tools.base import BaseTool, ToolCallRequest, ToolCallResult
from agentflow.tools.code.sandbox.python_executor import PythonExecutor
from agentflow.tools.code.sandbox.python_executor_no_proc import PythonExecutorNoProc
from agentflow.tools.code.sandbox.execution_plan import ExecPlan
from agentflow.tools.code.sandbox.python_sandbox import SandboxConfig

class PythonExecutionTool(BaseTool):
    """Execute Python code in a controlled sandbox with extensible context & helpers."""
    name = "python"
    description = "Execute Python code safely with optional stdout capture, headers/context/helpers injection."

    def __init__(self,
                 *,
                 timeout_length: int = 5,
                 mem_limit_mb: int = 512,
                 allowed_imports: Optional[List[str]] = None,
                 truncate_len: int = 600,
                 config: Optional[Dict[str, Any]] = None,
                 max_rounds: int = 3,
                 headers: Optional[List[str]] = None,
                 context: Optional[Dict[str, Any]] = None,
                 helper_modules: Optional[List[Dict[str, Any]]] = None,
                 helpers: Optional[Dict[str, Any]] = None,
                 use_proc: bool = False,
                 use_tqdm : bool = False
        ):
        super().__init__(config=config, max_rounds=max_rounds)

        sconf = SandboxConfig(
            time_limit_s=float(timeout_length),
            mem_limit_mb=int(mem_limit_mb),
            allowed_imports=allowed_imports,
            truncate_len=int(truncate_len),
        )
        self.use_tqdm = use_tqdm
        self.use_proc = use_proc
        if use_proc:
            self.executor = PythonExecutor(config=sconf)
        else:
            self.executor = PythonExecutorNoProc(config=sconf)
        if headers: 
            self.executor.set_headers(headers)
        if context: 
            self.executor.set_context(context)

        if helper_modules:
            for spec in helper_modules:
                mod = spec.get("module")
                names = spec.get("names")
                alias = spec.get("alias")
                if mod:
                    self.executor.inject_from_module(mod, names=names, alias=alias)

        if helpers:
            self.executor.inject_helpers(helpers)

    def register_header(self, code_piece: str) -> None:
        self.executor.register_header(code_piece)

    def update_context(self, ctx: Dict[str, Any]) -> None:
        self.executor.set_context(ctx)

    def register_helpers_from_module(self, module: str, names: Optional[List[str]] = None,
                                     alias: Optional[Dict[str, str]] = None) -> None:
        self.executor.inject_from_module(module, names=names, alias=alias)

    def register_helpers_from_code(self, code: str, export: Optional[List[str]] = None,
                                   alias: Optional[Dict[str, str]] = None) -> None:
        self.executor.inject_from_code(code, export=export, alias=alias)

    def register_helpers(self, helpers: Dict[str, Any]) -> None:
        self.executor.inject_helpers(helpers)

    def run_one(self, call: ToolCallRequest, **kwargs: Any) -> ToolCallResult:
        if self._is_quota_exceeded(call):
            return self._make_exceeded_result(call)

        meta = call.meta or {}
        cap = meta.get("capture_mode", "stdout")
        symbol = meta.get("answer_symbol")
        expr = meta.get("answer_expr")
        plan = ExecPlan(code=str(call.content), capture_mode=cap, answer_symbol=symbol, answer_expr=expr)
        res = self.executor.run(plan)

        out = f"Console: {res.stdout}\nResult: {res.result}"
        rep = "Execution Success" if res.ok else f"Execution Failed\n{res.error}"
        meta_out = {
            "success": bool(res.ok),
            "error": res.error,
            "duration_s": res.duration_s
        }
        return ToolCallResult(
            tool_name=self.name,
            request_content=call.content,
            output=out + f"\nReport: {rep}",
            meta=meta_out,
            error=None,
            index=call.index,
            call=call,
        )

    def run_batch(self, calls: List[ToolCallRequest], **kwargs: Any) -> List[ToolCallResult]:
        def _runner(allowed_calls: List[ToolCallRequest]) -> List[ToolCallResult]:
            if not allowed_calls:
                return []
            plans = []
            for c in allowed_calls:
                meta = c.meta or {}
                plans.append(ExecPlan(code=str(c.content),
                                      capture_mode=meta.get("capture_mode", "stdout"),
                                      answer_symbol=meta.get("answer_symbol"),
                                      answer_expr=meta.get("answer_expr")))
            results = self.executor.run_many(plans, show_progress=self.use_tqdm)
            packed: List[ToolCallResult] = []
            for res, c in zip(results, allowed_calls):
                out = f"Console: {res.stdout}\nResult: {res.result}"
                rep = "Execution Success" if res.ok else f"Execution Failed\n{res.error}"
                meta_out = {"success": bool(res.ok), "error": res.error, "duration_s": res.duration_s}
                packed.append(ToolCallResult(
                    tool_name=self.name,
                    request_content=c.content,
                    output=out + f"\nReport: {rep}",
                    meta=meta_out,
                    error=None,
                    index=c.index,
                    call=c,
                ))
            return packed
        return self._apply_round_quota(calls, _runner)
