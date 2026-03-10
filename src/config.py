import importlib.util
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


def configure_logging(log_level: str, log_file: Optional[Path]) -> None:
    """
    Configure logging for the script.

    Parameters
    ----------
    log_level : str
        Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : Path, optional
        Path to the log file. If not provided, logs will be printed to stdout.

    Returns
    -------
    None
    """
    # By default, log format does not include timestamp for easier reading in
    # stdout
    log_format = "%(levelname)s - %(message)s"
    handlers = []

    # Add a file handler with rotation if log_file is provided
    if log_file:
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        # Configure log to have a max size of 1MB and keep 1 backup file before
        # overwriting
        file_handler = RotatingFileHandler(
            log_file, mode="a", maxBytes=1024 * 1024, backupCount=1
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    # Else, print logs to stdout
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    logging.basicConfig(level=log_level, handlers=handlers)
    # Suppress noisy logging from urllib3 called by dxpy when making API calls
    logging.getLogger("urllib3.util.retry").setLevel(logging.WARNING)


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load the configuration file

    Parameters
    ----------
    config_path : Path
        Path to the Python configuration file

    Returns
    -------
    dict
        Configuration dictionary loaded from the specified file
    """
    spec = importlib.util.spec_from_file_location(
        config_path.name.replace(".py", ""), config_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[config_path.name] = module
    spec.loader.exec_module(module)
    config = module.CONFIG
    logging.debug(
        "Config version: %s",
        config.get("version", "version key not found in config"),
    )
    return config
