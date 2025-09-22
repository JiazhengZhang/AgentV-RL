import json
import jsonlines
import re

from typing import List, Dict, Any, Optional
from dataclasses import is_dataclass, asdict
from datetime import date, datetime
from enum import Enum
from pathlib import Path
import collections.abc as cabc


class JsonUtil:

    @staticmethod
    def read_json(file_path:str,mode="r"):
        """Read json"""
        with open(file_path, mode, encoding='utf-8') as file:
            return json.load(file)

    @staticmethod
    def write_json(file_path:str,data:List[Dict],mode="w"):
        """Write json"""
        with open(file_path, mode, encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

            
    @staticmethod
    def parse_json(text: str) -> Optional[List[Any]]:
        """
        从文本中提取并解析所有 JSON 顶层对象/数组。
        - 支持花括号 {..} 和方括号 [..] 的任意嵌套
        - 正确忽略字符串中的括号和转义符
        - 严格 JSON（双引号、无注释、无尾逗号）。如需宽松 JSON，请参考注释中的 json5 方案。

        返回:
            List[Any] | None
        """
        objs: List[Any] = []
        in_str = False           # 是否在字符串里
        escape = False           # 字符串内：上一字符是否为反斜杠
        stack: List[str] = []    # 括号栈，元素为'{'或'['
        start_idx: Optional[int] = None  # 当前顶层 JSON 片段的起始位置（包含起始括号）

        def is_open(c: str) -> bool:
            return c in '{['

        def is_close(c: str) -> bool:
            return c in '}]'

        def match(open_c: str, close_c: str) -> bool:
            return (open_c == '{' and close_c == '}') or (open_c == '[' and close_c == ']')

        for i, ch in enumerate(text):
            if in_str:
                # 字符串内部的处理（只关心结束引号与转义）
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
                # 其他字符在字符串内不影响括号匹配
                continue

            # 不在字符串里
            if ch == '"':
                in_str = True
                continue

            if is_open(ch):
                if not stack:
                    # 新的顶层 JSON 片段开始
                    start_idx = i
                stack.append(ch)
                continue

            if is_close(ch):
                if not stack:
                    # 孤立的右括号，跳过
                    continue
                top = stack.pop()
                if not match(top, ch):
                    # 括号不匹配，重置（防止错误片段污染后续识别）
                    stack.clear()
                    start_idx = None
                    continue

                if not stack and start_idx is not None:
                    # 一个完整的顶层 JSON 片段结束
                    candidate = text[start_idx:i+1]
                    # 尝试严格 JSON 解析
                    try:
                        objs.append(json.loads(candidate))
                    except json.JSONDecodeError:
                        # 如果你需要支持宽松 JSON（如尾逗号/单引号/注释），
                        # 可以启用 json5 作为回退（第三方库：pip install json5）:
                        # import json5
                        # try:
                        #     objs.append(json5.loads(candidate))
                        # except Exception:
                        #     pass
                        pass
                    start_idx = None

        return objs or None
     

    @staticmethod
    def read_jsonlines(file_path,mode="r"):
        """Read jsonlines

        """
        data = []
        with open(file_path, mode, encoding='utf-8') as f:
            for line in jsonlines.Reader(f):
                data.append(line)
        return data

    @staticmethod
    def write_jsonlines(file_path, data,mode="a"):
        """write jsonlines

        """
        with open(file_path, mode, encoding="utf-8") as w:
            if isinstance(data,list):
                for i in data:
                    json.dump(i, w, ensure_ascii=False)
                    w.write('\n')
            else:
                json.dump(data, w, ensure_ascii=False)
                w.write('\n')
                
    @staticmethod      
    def json_sanitize(obj: Any, *, on_unknown: str = "drop", max_depth: int = 64) -> Any | None:
        """
        将任意对象清洗为“可被 json 序列化”的结构。
        - dict: 递归清洗，键统一转 str；不可序列化的键/值会被丢弃
        - list/tuple/set: 递归清洗；set 转 list；不可序列化项被丢弃
        - dataclass -> asdict；Enum -> value(可再递归)；Path/bytes/datetime 等做常见转换
        - 具有 .tolist() / .item() 的对象（如 numpy/pandas/torch）会优先尝试这些转换
        - 其他未知对象：on_unknown="drop" 丢弃；"stringify" 则转 str
        - 若顶层整体不可序列化，返回 None
        """
        _SKIP = object()

        def _inner(o: Any, depth: int) -> Any:
            if depth > max_depth:
                return _SKIP

            # 基础可序列化类型
            if isinstance(o, (str, int, float, bool)) or o is None:
                return o

            # 常见转换
            if isinstance(o, (bytes, bytearray)):
                return o.decode("utf-8", errors="replace")
            if isinstance(o, (date, datetime)):
                return o.isoformat()
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, Enum):
                return _inner(o.value, depth + 1)

            # dataclass
            if is_dataclass(o):
                o = asdict(o)

            # torch/numpy/pandas 等：duck-typing 转基础类型
            if hasattr(o, "detach") and callable(getattr(o, "detach")) and hasattr(o, "cpu"):
                try:
                    return _inner(o.detach().cpu(), depth + 1)
                except Exception:
                    pass
            if hasattr(o, "tolist"):
                try:
                    return _inner(o.tolist(), depth + 1)
                except Exception:
                    pass
            if hasattr(o, "item"):
                try:
                    return _inner(o.item(), depth + 1)
                except Exception:
                    pass

            # 映射类型
            if isinstance(o, cabc.Mapping):
                out = {}
                for k, v in o.items():
                    try:
                        sk = k if isinstance(k, str) else str(k)
                    except Exception:
                        continue
                    sv = _inner(v, depth + 1)
                    if sv is not _SKIP:
                        out[sk] = sv
                return out

            # 序列类型（包含 set / tuple / list）
            if isinstance(o, (list, tuple, set, frozenset)) or (isinstance(o, cabc.Sequence) and not isinstance(o, (str, bytes, bytearray))):
                out = []
                for it in list(o):  # set/frozenset 转列表
                    si = _inner(it, depth + 1)
                    if si is not _SKIP:
                        out.append(si)
                return out

            # 其他未知对象
            if on_unknown == "stringify":
                try:
                    return str(o)
                except Exception:
                    return _SKIP
            return _SKIP

        cleaned = _inner(obj, 0)
        return None if cleaned is _SKIP else cleaned

def load_dataset(path: str):
    if path.endswith(".json"):
        data = JsonUtil.read_json(path)
    elif path.endswith(".jsonl"):
        data = JsonUtil.read_jsonlines(path)
    else:
        raise ValueError("Dataset format not supported")
    return data