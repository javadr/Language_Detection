#!/usr/bin/env python3

from __future__ import annotations

# ----------------------
# Standard library modules
# ----------------------
from pathlib import Path
import re

# ----------------------
# Third-party modules
# ----------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import DataLoader, Dataset

# ----------------------
# Local application modules
# ----------------------
from config import app_config, get_logger
from constants import ISO639_LANGUAGE_NAMES

# __all__ = ["", ""]


logger = get_logger(__file__)


def set_random_seed(seed: int = app_config.base.seed):
    # To make a reproducible output
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Data:
    # Create bag of words
    _cv = CountVectorizer(
        analyzer=app_config.cv.analyzer,
        # stop_words=CFG.stop_words,
        ngram_range=app_config.cv.ngram_range,
        min_df=app_config.cv.min_df,
    )

    def __init__(self, df: pd.DataFrame, le: LabelEncoder = None, cv: CountVectorizer = None):
        # Drop duplicates and NAs
        df = df.drop_duplicates().dropna().reset_index(drop=True)
        self.df = df
        self.raw_X = df["Text"]
        self.raw_y = df["Language"]

        # Use the given LabelEncoder, or create a new one if none provided
        self.le = le if le else LabelEncoder()
        if le:
            # Transform using existing encoder
            self.y = self.le.transform(self.raw_y)
        else:
            # Fit encoder on main dataset
            self.y = self.le.fit_transform(self.raw_y)

        self.X = self.preprocess_text(self.raw_X)
        fit = False
        # Vectorizer
        if cv is None:
            cv = self.__class__._cv
            fit = True
        self.CountVectorizerFit(fit=fit)

    @property
    def cv(self):
        return self.__class__._cv

    @cv.setter
    def cv(self, value):
        self._cv = value

    @property
    def vocabulary_size(self):
        return len(self.cv.vocabulary_)

    @property
    def label_size(self):
        return len(self.le.classes_)

    @property
    def label_names(self):
        return self.le.classes_.tolist()

    @staticmethod
    def readcsv(file_: Path | str, **kwargs) -> pd.DataFrame:
        df = pd.read_csv(
            app_config.base.data_path / f"{file_}",
            encoding="utf8",
            **kwargs,
        )
        return df

    @staticmethod
    def preprocess_text(text: pd.Series) -> pd.DataFrame | pd.Series:
        """Sanitatize Data by removing unnecessary characters."""
        extras = "«»\n\r\t!\"$%&/{}[]()=?\\`´*+~#-_.:,;<>|1234567890°-'۰۱۲۳۴۵۶۷۸۹،×„•′؟؛"
        rx = "[" + re.escape("".join(extras)) + "]"

        def sanitize(text: str) -> str:
            text = re.sub(" +", " ", re.sub(rx, " ", text)).strip()
            return text.lower().replace(" ", "_")

        return text.apply(sanitize)  # .tolist()

    def CountVectorizerFit(self, fit=True):
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=app_config.nn.test_size,
            random_state=app_config.base.seed,
            stratify=self.y,
        )

        self.X_train = self.cv.fit_transform(X_train) if fit else self.cv.transform(X_train)
        self.X_test = self.cv.transform(X_test)

    def train_test_split(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def fit_transform(self, fit=False):
        if fit:
            return self.cv.fit_transform(self.X)
        else:
            return self.cv.transform(self.X)


# Columns: Text, Language
data = Data(Data.readcsv("language_detection.csv.bz2"))

# Columns: Language, Text
# Language names are in iso639 format
# Test dataset, using same encoder and keeping only labels seen in main data
benchmark_df = Data.readcsv(
    "lanidenn_testset.csv.bz2",
    delimiter="^",
    header=None,
    names=["Language", "Text"],
)
# Map ISO codes to descriptive names, default to "Unknown" for unmapped codes
benchmark_df["Language"] = benchmark_df["Language"].map(
    lambda iso: ISO639_LANGUAGE_NAMES.get(iso, "Unknown"),
)
# Filter ldata to only labels present in main dataset
benchmark_df = benchmark_df[benchmark_df["Language"].isin(data.label_names)].reset_index(drop=True)

# Create Data instance using main dataset's LabelEncoder
# Use same LabelEncoder and CountVectorizer
benchmark_data = Data(benchmark_df, le=data.le, cv=data.cv)


class LanguageDetectionDataset(Dataset):
    def __init__(self, X, Y, train=True):
        self.X = X
        self.Y = Y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self.X[idx], self.Y[idx]


X_train, X_test, y_train, y_test = data.train_test_split()
train_dataset = LanguageDetectionDataset(
    X_train.toarray(),
    y_train,
)
val_dataset = LanguageDetectionDataset(
    X_test.toarray(),
    y_test,
)
test_dataset = LanguageDetectionDataset(
    benchmark_data.fit_transform(
        fit=False,
    ).toarray(),  # toarray() is used to convert the sparse matrix to a dense matrix
    benchmark_data.y,
)


def collate_fn(batch):
    x, y = zip(*batch)  # Unzips into two tuples
    # Stack along axis=0 to create shape (N, vocabulary_size=25643)
    np_stacked = np.stack(x, axis=0)
    # Convert to PyTorch tensor
    x = torch.from_numpy(np_stacked).to(dtype=torch.float, device=device)
    y = torch.tensor(y, dtype=torch.long).to(device)  # if labels are scalars, this works
    return x, y


kwargs = {
    "batch_size": app_config.nn.batch_size,
    "collate_fn": collate_fn,
    "num_workers": 0,
}

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    **kwargs,
)
val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    **kwargs,
)
test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    **kwargs,
)


if __name__ == "__main__":
    logger.info(data.le.classes_)
    logger.info([i.shape for i in data.train_test_split()])
    logger.info(f"train_dataset elements: {len(train_dataset)}")
    logger.info(f"test_dataset elements : {len(test_dataset)}")
    logger.info(f"train_loader batches: {len(train_loader)}")
    logger.info(f"test_loader batches : {len(test_loader)}")
    logger.info(f"data.vocabulary_size: {data.vocabulary_size}")
    logger.info(f"data.label_size: {data.label_size}")
