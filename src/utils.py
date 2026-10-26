"""
utils.py
--------
Utility functions for configuration loading, random seed setting,
reproducibility, directory management, and metadata serialization.
"""

import os
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np

# Canonical 4 vessel categories from DeepShip
CANONICAL_CLASSES = ["Cargo", "Passengership", "Tanker", "Tug"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CANONICAL_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CANONICAL_CLASSES)}


def get_project_root() -> Path:
    """Returns the absolute root directory of the project."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads YAML configuration from disk with environment variable overrides.
    
    Args:
        config_path: Optional path to config YAML. Defaults to config/config.yaml.
        
    Returns:
        Dict containing project configuration settings.
    """
    if config_path is None:
        config_path = str(get_project_root() / "config" / "config.yaml")

    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file.resolve()}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Check for environment variable overrides
    env_data_dir = os.environ.get("DEEPSHIP_DATA_DIR")
    if env_data_dir:
        config["dataset"]["raw_data_dir"] = env_data_dir

    return config


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds across Python, NumPy, and TensorFlow (if installed)
    to guarantee reproducible data splits and model initializations.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensure_dir(dir_path: str or Path) -> Path:
    """Ensures a directory exists, creating it if necessary."""
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, file_path: str or Path) -> None:
    """Serializes data to a formatted JSON file."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: str or Path) -> Any:
    """Loads and deserializes JSON from a file."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
