import ast
import copy
import io
import os


import pickle
import re
import resource
import sys
import time
import types
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import importlib

DEFAULT_PRELUDE = """
"""

_BANNED = [
    re.compile(r'(^|[^A-Za-z0-9_])input\s*\('),
    re.compile(r'(^|[^A-Za-z0-9_])open\s*\('),
    re.compile(r'(^|[^A-Za-z0-9_])os\.system\s*\('),
    re.compile(r'(^|[^A-Za-z0-9_])subprocess\.'),
    re.compile(r'(^|[^A-Za-z0-9_])socket\.'),
    re.compile(r'(^|[^A-Za-z0-9_])sys\.(?:exit|stdin|stdout|stderr)\b'),
]

ALLOWED_IMPORTS_DEFAULT = {
    "math", "numpy", "np", "sympy", "random", "time"
}

def _check_banned(text: str) -> None:
    for pat in _BANNED:
        if re.search(pat, text):
            raise RuntimeError(f"Forbidden pattern detected: {pat}")

def _ast_guard_imports(text: str, allowed: Optional[set]) -> None:
    if not allowed:
        return
    try:
        tree = ast.parse(text)
    except Exception:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                root = (n.name or '').split('.')[0]
                if root not in allowed:
                    raise RuntimeError(f"Import not allowed: {n.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or '').split('.')[0]
            if root and root not in allowed:
                raise RuntimeError(f"ImportFrom not allowed: {node.module}")

@dataclass
class SandboxConfig:
    time_limit_s: float = 5.0
    mem_limit_mb: int = 8
    allowed_imports: Optional[List[str]] = field(default_factory=lambda: sorted(ALLOWED_IMPORTS_DEFAULT))
    capture_stdout: bool = True
    seed: Optional[int] = 0
    prelude: str = DEFAULT_PRELUDE
    truncate_len: int = 500

@dataclass
class ExecutionResult:
    ok: bool
    result: Any
    stdout: str
    error: Optional[str]
    duration_s: float

class SandboxRuntime:
    def __init__(self,
                 config: SandboxConfig,
                 headers: Optional[List[str]] = None,
                 context: Optional[Dict[str, Any]] = None,
                 helpers: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self._globals: Dict[str, Any] = {}
        self._init_prelude()
        if headers:
            for h in headers:
                self.exec_header(h)
        if context:
            self.inject_context(context)
        if helpers:
            self.register_helpers(helpers)

    def inject_context(self, ctx: Dict[str, Any]) -> None:
        self._globals.update(ctx)

    def register_helpers(self, helpers: Dict[str, Any]) -> None:
        self._globals.update(helpers)

    def register_header(self, code: str) -> None:
        self.exec_header(code)

    @staticmethod
    def load_helpers_from_module(module_path: str,
                                 names: Optional[List[str]] = None,
                                 alias: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        mod = importlib.import_module(module_path)
        exported: Dict[str, Any] = {}
        if names:
            for n in names:
                obj = getattr(mod, n)
                exported[(alias or {}).get(n, n)] = obj
        else:
            for n, obj in vars(mod).items():
                if n.startswith("_"):
                    continue
                if callable(obj):
                    exported[n] = obj
        return exported

    @staticmethod
    def load_helpers_from_code(code: str,
                               export: Optional[List[str]] = None,
                               alias: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        ns: Dict[str, Any] = {}
        exec(code, ns, ns)
        out: Dict[str, Any] = {}
        if export:
            for n in export:
                out[(alias or {}).get(n, n)] = ns[n]
        else:
            for n, obj in ns.items():
                if n.startswith("_"):
                    continue
                if callable(obj):
                    out[n] = obj
        return out

    def _init_prelude(self) -> None:
        self._globals.clear()
        exec(self.config.prelude, self._globals, self._globals)

    def exec_header(self, code_piece: str) -> None:
        _check_banned(code_piece)
        _ast_guard_imports(code_piece, set(self.config.allowed_imports or []))
        exec(code_piece, self._globals, self._globals)

    def exec_block(self, code: str) -> None:
        _check_banned(code)
        _ast_guard_imports(code, set(self.config.allowed_imports or []))
        exec(code, self._globals, self._globals)

    def eval_expr(self, expr: str) -> Any:
        _check_banned(expr)
        _ast_guard_imports(expr, set(self.config.allowed_imports or []))
        return eval(expr, self._globals, self._globals)

def _apply_resource_limits(mem_limit_mb: int):
    try:
        bytes_limit = mem_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
        resource.setrlimit(resource.RLIMIT_DATA, (bytes_limit, bytes_limit))
    except Exception:
        pass

def _run_in_sandbox(code: str,
                    capture_mode: str,        
                    answer_symbol: Optional[str],
                    answer_expr: Optional[str],
                    config: SandboxConfig,
                    headers: List[str],
                    context: Dict[str, Any],
                    helpers: Dict[str, Any],
                    limit_resource: bool = True
                    ) -> ExecutionResult:
    start = time.time()
    try:
        if limit_resource:
            _apply_resource_limits(config.mem_limit_mb)
        if config.seed is not None:
            try:
                import random, numpy as _np
                random.seed(config.seed)
                _np.random.seed(config.seed)
            except Exception:
                pass

        rt = SandboxRuntime(config=config,
                            headers=headers,
                            context=context,
                            helpers=helpers)

        stdout_buf = io.StringIO()
        if config.capture_stdout:
            out_ctx = redirect_stdout(stdout_buf)
        else:
            # dummy context manager
            class _NullCtx:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            out_ctx = _NullCtx()

        with out_ctx:
            if capture_mode == 'stdout':
                rt.exec_block(code)
                result = stdout_buf.getvalue()
            elif capture_mode == 'symbol' and answer_symbol:
                rt.exec_block(code)
                result = rt._globals.get(answer_symbol)
            elif capture_mode == 'expr' and answer_expr:
                rt.exec_block(code)
                result = rt.eval_expr(answer_expr)
            else:
                lines = [ln for ln in code.splitlines() if ln.strip()!='']
                if len(lines) > 1:
                    rt.exec_block("\n".join(lines[:-1]))
                    result = rt.eval_expr(lines[-1])
                else:
                    result = rt.eval_expr(code)

        try:
            pickle.dumps(result)
        except Exception:
            result = repr(result)

        out = stdout_buf.getvalue() if config.capture_stdout else ""

        if isinstance(result, str) and len(result) > config.truncate_len:
            result = result[:config.truncate_len] + "...(trunc)"
        if len(out) > config.truncate_len:
            out = out[:config.truncate_len] + "...(trunc)"

        return ExecutionResult(ok=True, result=result, stdout=out, error=None, duration_s=time.time()-start)

    except Exception as e:
        return ExecutionResult(ok=False, result="",
                               stdout="",
                               error=f"{type(e).__name__}: {e}",
                               duration_s=time.time()-start)
