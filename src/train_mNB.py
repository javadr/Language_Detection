#!/usr/bin/env python3

# ----------------------
# Standard library modules
# ----------------------

# ----------------------
# Third-party modules
# ----------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from timebudget import timebudget
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.sparse import vstack

# ----------------------
# Local application modules
# ----------------------
from config import app_config, get_logger
from data import data, benchmark_data


logger = get_logger(__file__)

# X_train, y_train
X_train, X_test, y_train, y_test = data.train_test_split()
X = vstack([X_train, X_test])
y = np.concatenate([y_train, y_test])
# X_test, y_test
X_train, X_test, y_train, y_test = benchmark_data.train_test_split()
X_test = vstack([X_train, X_test])
y_test = np.concatenate([y_train, y_test])

modelMNB = MultinomialNB()
with timebudget("Training model MultinomialNB"):
    modelMNB.fit(X, y)


# Prediction
y_pred = modelMNB.predict(X_test)

# Find labels actually present in y_test or y_pred
labels = np.unique(np.concatenate([y_test, y_pred]))
# Map those labels to class names
target_names = data.le.inverse_transform(labels)


ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Generate report
cr = classification_report(
    y_test,
    y_pred,
    labels=labels,
    target_names=target_names,
    zero_division=0,
)


# visualising the confusion matrix
labels = pd.DataFrame(cm).map(lambda v: f"{v}" if v != 0 else "")
ticks = data.le.inverse_transform(range(len(data.le.classes_)))
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=labels, fmt="s", xticklabels=ticks, yticklabels=ticks, linewidths=0, cbar=True)
plt.tight_layout()
plt.savefig(app_config.base.images_path / "confusion_matrix_MultinomialNB.pdf", dpi=600)
plt.show()

if __name__ == "__main__":
    # Accuracy
    logger.info(f"Accuracy is : {ac}")

    # classification report
    logger.info(cr)
    s = [
        "this is test for 4th.",
        "أي تغيير  يتلاعب  بطريقة  بشكل  بسلامة يعتبر",
        "امروز هوا بارانی است.",
        "Eftersom Wikipedia bygger på tidskriftsartiklar, forskningspublikationer och böcker finns risk för rundgång ",
        "loro tre erano amiche da secoli, ma Terry ed Ellie avevano un segreto entrambe.",
        "Ja, ich bin heute da!",
    ]
    sv = data.cv.transform(s).toarray()
    l = modelMNB.predict(sv)
    for text, lang in zip(s, l):
        logger.info(f"{data.le.inverse_transform([lang])[0]} <-> {text}")
