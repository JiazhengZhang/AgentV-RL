import os
from pathlib import Path

def ensure_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)