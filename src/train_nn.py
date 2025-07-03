#!/usr/bin/env python3

# Standard library modules
import time

# Third-party modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from torch.utils.data import DataLoader

# Local application modules
from .config import CFG, get_logger, Net
from .data import data, train_loader, val_loader
from . import utils


logger = get_logger(__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


input_size = data.vocabulary_size
output_size = data.label_size

modelNN = Net(input_size, output_size).to(device)
bestModel = modelNN

optimizer = Adam(modelNN.parameters(), lr=CFG.lr)
lr_scheduler = StepLR(
    optimizer,
    step_size=CFG.lr_scheduler_step_size,
    gamma=CFG.lr_scheduler_gamma,
)
criterion = nn.CrossEntropyLoss()


def fwd_pass(X, y, net, loss_function, optimizer, train=False) -> tuple[float, torch.Tensor, torch.Tensor]:
    if train:
        net.train()
        outputs = net(X)
        loss = loss_function(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        net.eval()
        with torch.no_grad():
            outputs = net(X)
            loss = loss_function(outputs, y)
    with torch.no_grad():
        preds = outputs.argmax(dim=1)

    return loss.item(), preds, y


def total_batches(
    loader: DataLoader,
    *,
    desc: str,
    train: bool,
    conf_matrix: bool = False,
) -> tuple[float | np.ndarray, float, float]:
    loss = 0
    all_preds, all_labels = [], []
    for X, y in tqdm(loader, desc=desc, total=len(loader), leave=False):
        loss, preds, targets = fwd_pass(
            X,
            y,
            modelNN,
            criterion,
            optimizer,
            train,
        )
        all_preds.append(preds.cpu())
        all_labels.append(targets.cpu())
        loss += loss

    loss /= len(train_loader)  # average over batches
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    if conf_matrix:
        conf_mat = confusion_matrix(y_true=all_labels, y_pred=all_preds)
        return conf_mat, acc, f1

    return loss, acc, f1


print_every = CFG.n_epochs // 10
train_losses, val_losses = [], []
accuracies, val_accuracies = [], []
f1s, val_f1s = [], []


start = time.time()
logger.info("Training Starts ...")


best_loss = float("inf")

for epoch in range(CFG.n_epochs + 1):
    ### Training loop

    train_loss, train_acc, train_f1 = total_batches(train_loader, desc="Training...", train=True)
    train_losses.append(train_loss)
    accuracies.append(train_acc)
    f1s.append(train_f1)

    ### Validation loop
    val_loss, val_acc, val_f1 = total_batches(val_loader, desc="Validating...", train=False)

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_f1s.append(val_f1)

    # Save the best model
    if best_loss > val_loss:
        best_loss = val_loss
        best_model = modelNN

    # Update the learning rate
    lr_scheduler.step()

    # Print the results
    if epoch <= 3 or epoch % print_every == 0:
        logger.info(
            f"\r====> Epoch [{epoch:3}] Train/Validation loss: {train_loss:.9f}/{val_loss:.9f} :: Accuracy: {train_acc * 100:7.5f}/{val_acc * 100:7.5f}, F1 Scores: {train_f1:.5f}/{val_f1:.5f}"
        )


logger.info(f"Training ends at {utils.timeSince(start)}")


utils.res_plot([train_losses, val_losses], title="Train vs Valid Losses")
utils.res_plot([accuracies, val_accuracies], ylabel="Accuracy", title="Train vs Valid Accuracies")

# torch.save(best_model, CFG.saved_models_path / "bestmodel_nn.pth")
torch.save(
    {
        "model_state_dict": best_model.state_dict(),
        "model_in_out": (input_size, output_size),
        "vectorizer": data.cv,
        "label_encoder": data.le,
    },
    CFG.saved_models_path / "bestmodel_nn.pth",
)

best_model.eval()
modelNN = best_model

conf_matrix, accuracy, f1score = total_batches(
    val_loader,
    desc="Testing...",
    train=False,
    conf_matrix=True,
)

logger.info(f"Accuracy is {accuracy}\nf1-score is {f1score}")

ticks = [str(data.le.inverse_transform([j])[0]) for j in range(data.label_size)]
df_cm = pd.DataFrame(conf_matrix, columns=ticks, index=ticks)
df_cm.index.name = "Actual"
df_cm.columns.name = "Predicted"
plt.figure(figsize=(8, 6))
sns.set(font_scale=0.8)
labels = pd.DataFrame(df_cm).map(lambda v: f"{v}" if v != 0 else "")
sns.heatmap(df_cm, annot=labels, fmt="s", linewidths=0.5)
plt.tight_layout()
plt.title("Confusion Matrix - NN")
plt.savefig(CFG.images_path / "confusion_matrix_NN.pdf", dpi=600)
plt.show()


if __name__ == "__main__":
    summary(
        Net(input_size, output_size),
        input_size=(input_size,),
        batch_size=-1,
        device="cpu",
    )
