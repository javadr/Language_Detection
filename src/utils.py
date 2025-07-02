#!/usr/bin/env python3

# Standard library modules
import time
import math

# Third-party modules
import matplotlib.pyplot as plt

# Local application modules
from config import CFG


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m:d}m {s:0.0f}s"


def res_plot(plotdata, xlabel="epoch", ylabel="losses", legend=["Train", "Valid"], title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    ax1.set_ylabel(ylabel)
    for ax in (ax1, ax2):
        n = 10 if ax == ax2 else 0
        x = range(CFG.n_epochs - n + 1, CFG.n_epochs + 1) if ax == ax2 else range(CFG.n_epochs + 1)
        title = f"{title} ({n} last values)" if ax == ax2 else title
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.plot(x, plotdata[0][-n:], "-o", label=legend[0])
        ax.plot(x, plotdata[1][-n:], "-o", label=legend[1])
        ax.legend() 
    plt.tight_layout()
    plt.savefig(CFG.images_path / f"{title}.pdf", dpi=600)
    plt.show()


ISO639_LANGUAGE_NAMES = {
    "ara": "Arabic",
    "dan": "Danish",
    "deu": "German",
    "gre": "Greek",
    "eng": "English",
    "spa": "Spanish",
    "fas": "Persian",
    "fra": "French",
    "hin": "Hindi",
    "ita": "Italian",
    "kan": "Kannada",
    "mal": "Malayalam",
    "dut": "Dutch",
    "por": "Portuguese",
    "rus": "Russian",
    "swe": "Swedish",
    "tam": "Tamil",
    "tur": "Turkish",
}


if __name__ == "__main__":
    start = time.time() - 1000
    print(timeSince(start))