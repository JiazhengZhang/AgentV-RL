# tests/test_python_execution_tool_ray.py

import sys
import os
import ray

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT_DIR)

from agentflow.tools.code.python_execution_ray import PythonExecutionToolRay, create_python_actor
from agentflow.tools.base import ToolCallRequest



def setup_module():
    """Start Ray before all tests."""
    if not ray.is_initialized():
        ray.init(
            include_dashboard=False,
        )
        
    actor = create_python_actor()


def teardown_module():
    """Shutdown Ray after all tests."""
    if ray.is_initialized():
        ray.shutdown()


def test_python_tool_print_basic():
    actor = create_python_actor()
    tool = PythonExecutionToolRay(actor = actor,timeout_length=5)
    request = ToolCallRequest(
        index=0,
        name="python",
        content="print(1+2+3)"
    )
    result = tool.run_one(request)
    print(result)
    assert "6" in result.output


def test_python_tool_forbidden_calls():
    actor = create_python_actor()
    tool = PythonExecutionToolRay(actor = actor, timeout_length=3)
    request = ToolCallRequest(
        index=0,
        name="python",
        content="input('x')"
    )
    result = tool.run_one(request)
    print(result)
    assert result.meta["success"] is False
    assert "Forbidden" in (result.meta["error"] or "")


def test_python_tool_timeout():
    actor = create_python_actor()
    tool = PythonExecutionToolRay(actor = actor, timeout_length=2)
    request = ToolCallRequest(
        index=0,
        name="python",
        content="import time\nprint('start')\ntime.sleep(10)\nprint('end')"
    )
    result = tool.run_one(request)
    print(result)
    assert result.meta["success"] is False
    assert result.meta["error"] == "Timeout"


def test_python_tool_batch():
    actor = create_python_actor()
    tool = PythonExecutionToolRay(actor = actor, timeout_length=5)

    tool.register_helpers_from_code(
        """
import math
import numpy as np
def calculate(a,b):
    return a+b
def foo():
    return np.sum([1,2,3])
"""
    )

    requests = [
        ToolCallRequest(
            index=0,
            name="python",
            content="print(calculate(10,20))"
        ),
        ToolCallRequest(
            index=1,
            name="python",
            content="print(foo())"
        )
    ]

    results = tool.run_batch(requests)
    print(results)

    assert results[0].meta["success"] is True
    assert "30" in results[0].output

    assert results[1].meta["success"] is True
    assert "6" in results[1].output


if __name__ == "__main__":
    setup_module()
    
    test_python_tool_print_basic()
    test_python_tool_forbidden_calls()
    test_python_tool_timeout()
    test_python_tool_batch()
    teardown_module()
