#!/usr/bin/env python3

# Standard library modules
import logging

from pathlib import Path
from dataclasses import dataclass

# Third-party modules
import torch
from torch import nn



logging.basicConfig(level=logging.INFO)


def get_logger(name: str | Path = __file__):
    name = Path(name).name
    logger = logging.getLogger(name)
    return logger



@dataclass
class CFG:
    # Random seed
    seed: int = 42
    # Paths
    working_dir: Path = Path(__file__).parent.parent
    data_path: Path = working_dir / "data"
    result_path: Path = working_dir / "results"
    saved_models_path: Path = result_path / "saved-models"
    images_path: Path = result_path / "images"
    # Training
    n_epochs: int = 20
    batch_size: int = 512
    lr: float = 1e-3  # learning rate
    wd: float = 1e-5  # weight decay
    dropout: float = 0.6
    test_size: float = 0.20
    # Learning Rate Scheduler
    lr_scheduler_step_size: int = 10
    lr_scheduler_gamma: float = 0.86
    # CountVectorizer
    analyzer: str = "char"
    # stop_words: str = "english" # just work with "word" analyzer
    ngram_range: tuple[int, int] = (2, 3)  # bigram and trigram
    min_df: int = 5



# Neural Network Model
class Net(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout=CFG.dropout):
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
for dir in (CFG.saved_models_path, CFG.images_path):
    dir.mkdir(exist_ok=True, parents=True)


logger = get_logger(Path(__file__).name)

if __name__ == "__main__":
    logger.info(CFG.working_dir)
