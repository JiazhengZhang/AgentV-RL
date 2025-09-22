import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Dict, Any

def get_logger(config: Dict, name: str = "log") -> logging.Logger:
    """Get a logger for current context

    Args:
        config (Dict): configuration
        name (str, optional): name of the logger. Defaults to "my_project".
        level (_type_, optional): logging level. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: _description_
    """
    level = logging.DEBUG
    level = config["logging"]["level"]
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(processName)s:%(process)d] [%(levelname)s] [%(name)s] %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_to_file = config["logging"]["log_to_file"]
    if log_to_file:
        log_file_dir  = config["logging"]["log_file_dir"]
        log_file_name = config["logging"]["log_file_name"]
        pid = os.getpid()
        from multiprocessing import current_process
        process_name = current_process().name
        base, ext = os.path.splitext(log_file_name)
        log_file_name = f"{base}_{pid}_{process_name}_{ext}"
        
        log_base_name = os.path.splitext(os.path.basename(config["logging"]["log_file_name"]))[0]
        if log_base_name == "default":
            log_base_name = datetime.now().strftime('%Y-%m-%d-%H-%M')
        
        log_file_dir = os.path.join(log_file_dir,log_base_name)
        

        log_file_path = os.path.join(log_file_dir, log_file_name)
        os.makedirs(log_file_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=100*1024*1024,
            backupCount=5,
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