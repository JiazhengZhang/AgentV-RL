import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from multiprocessing import current_process
from datetime import datetime
from typing import Optional, Dict, Any

def _coerce_level(level: Any, default=logging.INFO) -> int:
    """支持 'INFO' / 'DEBUG' / 20 之类写法。"""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        lvl = level.strip().upper()
        return getattr(logging, lvl, default)
    return default

def get_logger(config: Optional[Dict[str, Any]] = None, name: str = "log") -> logging.Logger:
    """Get a logger for current context

    Args:
        config (Dict): a dict-like object that contains "logging" key for logger config
        name (str, optional): name of the logger. Defaults to "my_project".
        level (_type_, optional): logging level. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: _description_
    """
    cfg = (config or {}).get("logging", {}) or {}

    level = _coerce_level(cfg.get("level", "DEBUG"))
    log_to_file = bool(cfg.get("log_to_file", False))
    log_file_dir = cfg.get("log_file_dir", "./logs")
    log_file_name = cfg.get("log_file_name", "default.log")
    max_bytes = int(cfg.get("max_bytes", 100 * 1024 * 1024))
    backup_count = int(cfg.get("backup_count", 5))

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(processName)s:%(process)d] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        pid = os.getpid()
        proc_name = current_process().name

        base, ext = os.path.splitext(log_file_name)
        if not ext:
            ext = ".log"

        log_base_name = os.path.splitext(os.path.basename(log_file_name))[0]
        if log_base_name == "default":
            log_base_name = datetime.now().strftime("%Y-%m-%d-%H-%M")

        target_dir = os.path.join(log_file_dir, log_base_name)
        os.makedirs(target_dir, exist_ok=True)

        final_filename = f"{base}_{pid}_{proc_name}{ext}"
        log_file_path = os.path.join(target_dir, final_filename)

        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def log_config(logger: logging.Logger, config: Dict[str, Any], parent_key: str = "") -> None:
    """
    Log the configuration with the given logger
    """
    for key, value in config.items():
        current_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            logger.info(f"{current_key}:")
            log_config(logger, value, current_key)
        else:
            logger.info(f"{current_key}: {value}")
            
            
def print_args(args):
    """Print key-val pairs of args"""
    for key, value in vars(args).items():
        print(f"{key}: {value}")