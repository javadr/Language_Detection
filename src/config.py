#!/usr/bin/env python3

from __future__ import annotations

# ----------------------
# Standard library modules
# ----------------------
import sys
import builtins
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

# ----------------------
# Third-party modules
# ----------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from rich import print

# ----------------------
# Local application modules
# ----------------------


__all__ = ["load_config", "get_logger"]

# Redirect logs to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)


def get_logger(name: str | Path = __file__):
    name = Path(name).name
    logger = logging.getLogger(name)
    return logger


def _project_root() -> Path:
    """Resolve the project root directory.

    Assumes this file lives in <root>/src/... or similar.
    """
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class BaseConfig:
    """Global application configuration."""

    # Reproducibility
    seed: int = 42

    # Paths
    working_dir: Path = _project_root()
    data_path: Path = working_dir / "data"
    result_path: Path = working_dir / "results"
    saved_models_path: Path = result_path / "saved-models"
    images_path: Path = result_path / "images"


@dataclass(frozen=True, slots=True)
class NNConfig:
    """Neural network training configuration."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    n_epochs: int = 20
    batch_size: int = 512
    lr: float = 1e-3  # learning rate
    wd: float = 1e-5  # weight decay
    dropout: float = 0.6
    test_size: float = 0.20

    # Learning rate scheduler
    lr_scheduler_step_size: int = 10
    lr_scheduler_gamma: float = 0.86


@dataclass(frozen=True, slots=True)
class CVConfig:
    """CountVectorizer configuration."""

    analyzer: str = "char"
    ngram_range: tuple[int, int] = (2, 3)  # bigram and trigram
    min_df: int = 5


@dataclass(frozen=True, slots=True)
class AppConfig:
    base: BaseConfig = BaseConfig()
    nn: NNConfig = NNConfig()
    cv: CVConfig = CVConfig()


def load_config() -> AppConfig:
    """Initialize global configuration and runtime defaults.

    - Sets random seeds
    - Configures matplotlib style
    - Ensures `display` is available outside Jupyter
    """
    warnings.simplefilter("ignore")
    plt.style.use("seaborn-v0_8")

    config = AppConfig()
    np.random.seed(config.base.seed)

    # ------------------------------------------------------------------
    # Ensure `display` exists outside Jupyter
    # ------------------------------------------------------------------
    if not hasattr(builtins, "display"):
        builtins.display = print

    return config


logger = get_logger(Path(__file__).name)
app_config = load_config()

# Neural Network Model
class Net(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout=app_config.nn.dropout):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, output_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_size * 4, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


# To ensure that the required directories are created.
for dir in (app_config.base.saved_models_path, app_config.base.images_path):
    dir.mkdir(exist_ok=True, parents=True)



if __name__ == "__main__":
    logger.info(app_config.base.working_dir)
