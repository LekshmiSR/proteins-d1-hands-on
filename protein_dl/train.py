"""Training loop, evaluation and result saving shared by all notebooks.

Notebook 01 (MLP) writes this loop out by hand, cell by cell, so you can see
every step. Later notebooks call these functions so they can focus on the
architecture.

Contract: every model is called as  logits = model(tokens, lengths)
  tokens  [B, L]     integer amino-acid ids
  lengths [B]        true length of each protein
  logits  [B, L, 3]  one score per class for every residue
"""

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .data import IGNORE_INDEX, NUM_CLASSES, Q3_NAMES


def set_seed(seed=42):
    """Make runs repeatable (as far as GPUs allow)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(verbose=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            props = torch.cuda.get_device_properties(0)
            print(f"Using GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU (no GPU found). Training will be slow; in Colab use "
                  "Runtime > Change runtime type > T4 GPU.")
    return device


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def masked_cross_entropy(logits, labels):
    """Cross-entropy over residues only (padding has label IGNORE_INDEX and is skipped)."""
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=IGNORE_INDEX)


def train_one_epoch(model, loader, optimizer, device, scaler=None, clip=1.0):
    model.train()
    total_loss, total_res = 0.0, 0
    for tokens, labels, lengths in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(tokens, lengths)
            loss = masked_cross_entropy(logits.float(), labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        n = (labels != IGNORE_INDEX).sum().item()
        total_loss += loss.item() * n
        total_res += n
    return total_loss / total_res


@torch.no_grad()
def predict(model, loader, device):
    """Run the model on a loader. Returns per-protein lists of (y_true, y_pred)."""
    model.eval()
    trues, preds = [], []
    total_loss, total_res = 0.0, 0
    for tokens, labels, lengths in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        logits = model(tokens, lengths).float()
        n = (labels != IGNORE_INDEX).sum().item()
        total_loss += masked_cross_entropy(logits, labels).item() * n
        total_res += n
        pred = logits.argmax(dim=-1)
        for i, L in enumerate(lengths.tolist()):
            y = labels[i, :L].cpu().numpy()
            p = pred[i, :L].cpu().numpy()
            keep = y != IGNORE_INDEX
            trues.append(y[keep])
            preds.append(p[keep])
    return trues, preds, total_loss / total_res


def compute_metrics(trues, preds, loss=None):
    """Q3 accuracy, macro-F1, per-class scores, confusion matrix, per-protein accuracy."""
    y = np.concatenate(trues)
    p = np.concatenate(preds)
    prec, rec, f1, support = precision_recall_fscore_support(
        y, p, labels=list(range(NUM_CLASSES)), zero_division=0)
    per_protein = [float((t == q).mean()) for t, q in zip(trues, preds) if len(t) > 0]
    return {
        "loss": loss,
        "q3_accuracy": float((y == p).mean()),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "mean_per_protein_accuracy": float(np.mean(per_protein)),
        "per_class": {name: {"precision": float(prec[i]), "recall": float(rec[i]),
                             "f1": float(f1[i]), "support": int(support[i])}
                      for i, name in enumerate(Q3_NAMES)},
        "confusion_matrix": confusion_matrix(y, p, labels=list(range(NUM_CLASSES))).tolist(),
        "n_residues": int(len(y)),
        "n_proteins": len(trues),
    }


def evaluate(model, loader, device):
    trues, preds, loss = predict(model, loader, device)
    return compute_metrics(trues, preds, loss)


def train_model(model, train_dl, valid_dl, device, epochs=10, lr=1e-3, weight_decay=1e-2,
                use_amp=None, clip=1.0, verbose=True):
    """Train with AdamW, keep the weights of the epoch with the best validation accuracy.

    use_amp: mixed precision (faster on GPU, less memory). Default: on for CUDA.
    Returns a history dict.
    """
    model.to(device)
    if use_amp is None:
        use_amp = device.type == "cuda"
        # GeForce GTX 16xx cards are known to give NaNs with half precision: use full precision there.
        if use_amp and "GTX 16" in torch.cuda.get_device_name(device):
            use_amp = False
            print("GTX 16xx GPU detected: mixed precision turned off for numerical safety.")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine schedule: learning rate smoothly decays to ~0 by the last epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_q3": [], "seconds": []}
    best_acc, best_state = -1.0, None
    start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, device, scaler, clip)
        val = evaluate(model, valid_dl, device)
        scheduler.step()
        dt = time.time() - t0
        for k, v in zip(history, [epoch, train_loss, val["loss"], val["q3_accuracy"], dt]):
            history[k].append(v)
        flag = ""
        if val["q3_accuracy"] > best_acc:
            best_acc = val["q3_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  <- best so far"
        if verbose:
            print(f"epoch {epoch:2d}/{epochs} | train loss {train_loss:.4f} | "
                  f"val loss {val['loss']:.4f} | val Q3 {val['q3_accuracy']:.4f} | {dt:5.1f}s{flag}")
    model.load_state_dict(best_state)
    history["total_seconds"] = time.time() - start
    history["best_val_q3"] = best_acc
    if verbose:
        print(f"Done in {history['total_seconds'] / 60:.1f} min. Best validation Q3 = {best_acc:.4f} "
              f"(those weights are now loaded in the model).")
    return history


def get_drive_dir(save_to_drive, folder="protein-dl-results"):
    """In Colab, mount Google Drive and return a folder inside it (else None).

    Colab forgets every file when the session ends. Saving results to Drive lets
    notebook 05 compare the models you trained in different sessions.
    """
    if not save_to_drive:
        return None
    try:
        from google.colab import drive  # only exists inside Colab
    except ImportError:
        print("Not running in Colab: results are saved only in the local results/ folder.")
        return None
    drive.mount("/content/drive")
    path = os.path.join("/content/drive/MyDrive", folder)
    os.makedirs(path, exist_ok=True)
    print("Results will also be saved to Google Drive:", path)
    return path


def save_results(name, model, history, val_metrics, test_metrics, config=None,
                 results_dir="results", drive_dir=None):
    """Write results/<name>.json (and optionally a copy to Google Drive)."""
    out = {
        "model": name,
        "n_parameters": count_parameters(model),
        "config": config or {},
        "history": history,
        "valid": val_metrics,
        "test_cb513": test_metrics,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    paths = [results_dir] + ([drive_dir] if drive_dir else [])
    for d in paths:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}.json"), "w") as f:
            json.dump(out, f, indent=1)
        print(f"saved {os.path.join(d, name + '.json')}")
    return out
