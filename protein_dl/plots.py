"""Small plotting helpers used by the notebooks."""

import matplotlib.pyplot as plt
import numpy as np

from .data import Q3_CLASSES, Q3_NAMES


def plot_history(history, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(history["epoch"], history["train_loss"], "o-", label="train")
    axes[0].plot(history["epoch"], history["val_loss"], "o-", label="validation")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss"); axes[0].legend()
    axes[0].set_title("Loss (lower is better)")
    axes[1].plot(history["epoch"], history["val_q3"], "o-", color="tab:green")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("Q3 accuracy")
    axes[1].set_title("Validation Q3 accuracy (higher is better)")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_confusion(cm, title="", ax=None, normalize=True):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm[i, j] > (0.5 if normalize else cm.max() / 2) else "black")
    ax.set_xticks(range(len(Q3_NAMES)), Q3_NAMES)
    ax.set_yticks(range(len(Q3_NAMES)), Q3_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title)
    if show:
        plt.tight_layout(); plt.show()


def show_prediction(sequence, true_ids, pred_ids, width=60):
    """Print sequence, true and predicted structure aligned, `width` residues per line."""
    true_s = "".join(Q3_CLASSES[i] for i in true_ids)
    pred_s = "".join(Q3_CLASSES[i] for i in pred_ids)
    match = "".join("|" if a == b else " " for a, b in zip(true_s, pred_s))
    for s in range(0, len(sequence), width):
        print(f"seq   {sequence[s:s + width]}")
        print(f"true  {true_s[s:s + width]}")
        print(f"pred  {pred_s[s:s + width]}")
        print(f"      {match[s:s + width]}\n")
