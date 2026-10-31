"""Trains the leaf-disease classifier and runs the transfer-learning ablation.

With 40 images per class a single held-out split would be ~8 images per class,
far too noisy to report, so every number here comes from stratified k-fold
cross-validation. Folds are split at the *image* level because crops drawn from
one leaf strip are near-duplicates and would otherwise leak across folds.

Usage:
    python train.py                       # full ablation, 5 folds, 3 modes
    python train.py --modes finetune      # just the mode being shipped
    python train.py --folds 3 --epochs 12 # quicker sweep
"""
import os
import sys
import json
import time
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CLASS_NAMES, LeafCropDataset, discover_images, load_strip, tta_crops
from model import (
    MODES, build_model, freeze_batchnorm_where_params_frozen, set_block_trainable,
)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get(
    "LEAF_DISEASE_DATA", os.path.join(BASE, "../../../data/rice+leaf+diseases")
)
MODELS_DIR = os.path.join(BASE, "models")
PROGRESS_PATH = os.path.join(MODELS_DIR, "fold_progress.json")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "leaf_disease_cnn.pt")
SEED = 42


def _save_checkpoint(state, mode, acc, args):
    torch.save(
        {"state_dict": state, "arch": args.arch, "mode": mode,
         "class_names": CLASS_NAMES, "cv_accuracy": acc},
        CHECKPOINT_PATH,
    )


def _loader(paths, labels, repeats, batch_size):
    ds = LeafCropDataset(paths, labels, repeats=repeats)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)


@torch.no_grad()
def predict_strips(net, paths, device, n_crops):
    """One averaged prediction per image, over evenly spaced crops."""
    net.eval()
    probs = []
    for p in paths:
        crops = tta_crops(load_strip(p), n_crops=n_crops).to(device)
        probs.append(torch.softmax(net(crops), dim=1).mean(0).cpu().numpy())
    return np.stack(probs)


def train_one(mode, tr_paths, tr_labels, args, device):
    net, head_params, block_params = build_model(mode, arch=args.arch, dropout=args.dropout)
    net.to(device)

    groups = [{"params": head_params, "lr": args.head_lr}]
    if block_params:
        groups.append({"params": block_params, "lr": args.block_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    loader = _loader(tr_paths, tr_labels, args.repeats, args.batch_size)
    staged = mode == "finetune" and block_params

    for epoch in range(args.epochs):
        if staged:
            # Head warm-up first: an untrained head backpropagating into
            # pretrained conv weights is what destroys the features.
            set_block_trainable(block_params, epoch >= args.warmup_epochs)
        net.train()
        # train() just reset every module, so re-apply BN freezing.
        freeze_batchnorm_where_params_frozen(net)
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

    return net


def _load_folds(mode, args):
    """Per-fold results already computed for this mode, keyed by fold number.

    Training the full ablation takes tens of CPU-minutes and has segfaulted
    mid-run on this platform, so each fold is cached as soon as it finishes and
    a rerun picks up where the crash left off.
    """
    if not os.path.exists(PROGRESS_PATH):
        return {}
    with open(PROGRESS_PATH) as f:
        progress = json.load(f)
    entry = progress.get(mode, {})
    # Settings that change what a fold means invalidate the cache.
    if entry.get("signature") != _signature(args):
        return {}
    return {int(k): v for k, v in entry.get("folds", {}).items()}


def _signature(args):
    return {k: getattr(args, k) for k in
            ("arch", "folds", "epochs", "warmup_epochs", "batch_size", "repeats",
             "tta_crops", "head_lr", "block_lr", "dropout", "weight_decay")}


def _save_fold(mode, fold, record, args):
    progress = {}
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            progress = json.load(f)
    entry = progress.get(mode)
    if not entry or entry.get("signature") != _signature(args):
        entry = {"signature": _signature(args), "folds": {}}
    entry["folds"][str(fold)] = record
    progress[mode] = entry
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def cross_validate(mode, paths, labels, args, device):
    paths, labels = np.array(paths), np.array(labels)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)

    cached = _load_folds(mode, args)
    if cached:
        print(f"  [{mode}] resuming, {len(cached)} fold(s) cached")

    accs, f1s = [], []
    cm_total = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    best = {"acc": -1.0, "state": None}

    for fold, (tr, va) in enumerate(skf.split(paths, labels), start=1):
        if fold in cached:
            rec = cached[fold]
            accs.append(rec["acc"])
            f1s.append(rec["f1"])
            cm_total += np.array(rec["cm"], dtype=int)
            print(f"  [{mode}] fold {fold}/{args.folds}  acc={rec['acc']:.3f}  "
                  f"macroF1={rec['f1']:.3f}  (cached)")
            continue

        t0 = time.time()
        net = train_one(mode, paths[tr], labels[tr], args, device)

        probs = predict_strips(net, paths[va], device, args.tta_crops)
        pred, truth = probs.argmax(1), labels[va]
        acc = float((pred == truth).mean())
        f1 = float(f1_score(truth, pred, average="macro", zero_division=0))
        cm = confusion_matrix(truth, pred, labels=range(len(CLASS_NAMES)))

        accs.append(acc)
        f1s.append(f1)
        cm_total += cm
        if acc > best["acc"]:
            best = {"acc": acc, "state": copy.deepcopy(net.state_dict())}
            # Persist immediately: a later fold may segfault, and the best
            # weights so far are what gets served.
            if mode == args.ship_mode:
                _save_checkpoint(best["state"], mode, acc, args)

        _save_fold(mode, fold, {"acc": acc, "f1": f1, "cm": cm.tolist()}, args)
        print(f"  [{mode}] fold {fold}/{args.folds}  acc={acc:.3f}  macroF1={f1:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    return {
        "mode": mode,
        "arch": args.arch,
        "folds": args.folds,
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs)), 4),
        "macro_f1_mean": round(float(np.mean(f1s)), 4),
        "macro_f1_std": round(float(np.std(f1s)), 4),
        "fold_accuracies": [round(a, 4) for a in accs],
        "confusion_matrix": cm_total.tolist(),
    }, best["state"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--arch", default="resnet18",
                    choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--warmup-epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--tta-crops", type=int, default=5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--block-lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--ship-mode", default="finetune", choices=MODES,
                    help="mode whose best fold is saved as the served checkpoint")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths, labels = discover_images(DATA_ROOT)
    print(f"Found {len(paths)} images across {len(CLASS_NAMES)} classes on {device}")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Results are merged into whatever a previous run already finished, and
    # flushed after every mode: a full ablation is tens of CPU-minutes and
    # losing completed modes to an interrupted run is needless.
    report_path = os.path.join(MODELS_DIR, "ablation.json")
    report = {}
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f).get("results", {})
        if report:
            print(f"Resuming; already have: {', '.join(report)}")

    def flush():
        with open(report_path, "w") as f:
            json.dump({"class_names": CLASS_NAMES, "seed": SEED, "results": report},
                      f, indent=2)

    for mode in args.modes:
        print(f"\n=== {mode} ({args.arch}) ===", flush=True)
        result, state = cross_validate(mode, paths, labels, args, device)
        report[mode] = result
        flush()
        print(f"  -> acc {result['accuracy_mean']:.3f} +/- {result['accuracy_std']:.3f}  "
              f"macroF1 {result['macro_f1_mean']:.3f}")

        if mode == args.ship_mode:
            if state is not None:
                # Re-save so the metadata carries the mode's mean CV accuracy
                # rather than the single best fold's; the agent reports this.
                _save_checkpoint(state, mode, result["accuracy_mean"], args)
                print("  -> saved checkpoint (models/leaf_disease_cnn.pt)")
            else:
                print("  -> all folds cached; kept existing checkpoint")

    flush()

    print("\n--- transfer-learning ablation ---")
    for mode, r in report.items():
        print(f"{mode:9s} acc={r['accuracy_mean']:.3f}+/-{r['accuracy_std']:.3f}  "
              f"macroF1={r['macro_f1_mean']:.3f}")


if __name__ == "__main__":
    main()
