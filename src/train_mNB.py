#!/usr/bin/env python3

# Standard library modules
from dataclasses import dataclass
from pathlib import Path
import re
from tkinter import Y

# Third-party modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.naive_bayes import MultinomialNB
from timebudget import timebudget
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score


# Local application modules
from config import CFG
from data import data

X_train, X_test, y_train, y_test = data.train_test_split()


modelMNB = MultinomialNB()
with timebudget("Training model MultinomialNB"):
    modelMNB.fit(X_train, y_train)

# Prediction
y_pred = modelMNB.predict(X_test)


ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred, target_names=np.unique(data.le.inverse_transform(y_test)))


# visualising the confusion matrix


labels = pd.DataFrame(cm).map(lambda v: f"{v}" if v != 0 else "")
ticks = list(map(str, np.unique(data.le.inverse_transform(y_test))))
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=labels, fmt="s", xticklabels=ticks, yticklabels=ticks, linewidths=0)
plt.tight_layout()
plt.savefig(CFG.images_path / "confusion_matrix_MultinomialNB.pdf", dpi=600)
# plt.show()

if __name__ == "__main__":
    # Accuracy
    print("Accuracy is :", ac)

    # classification report
    print(cr)
    s = [
    "this is test for 4th.",
    "أي تغيير  يتلاعب  بطريقة  بشكل  بسلامة يعتبر",
    "امروز هوا بارانی است.",
    "Eftersom Wikipedia bygger på tidskriftsartiklar, forskningspublikationer och böcker finns risk för rundgång ",
    "loro tre erano amiche da secoli, ma Terry ed Ellie avevano un segreto entrambe.",
    "Ja",
]
    sv = data.cv.transform(s).toarray()
    l = modelMNB.predict(sv)
    for text, lang in zip(s, l):
        print(f"{data.le.inverse_transform([lang])[0]} <-> {text}")