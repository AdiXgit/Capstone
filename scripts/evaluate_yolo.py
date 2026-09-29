#!/usr/bin/env python3
"""Evaluate the fine-tuned paddy disease classifier on its held-out val set and
write models/paddy_disease_metrics.json.

The gateway serves that file at /api/crop-health/model-metrics, and the dashboard
"Model Performance" page renders accuracy, per-class precision/recall, and the
confusion matrix from it. This is what turns "we trained a model" into a number
you can show.

Run:  python scripts/evaluate_yolo.py
      python scripts/evaluate_yolo.py --model models/paddy_disease_cls.pt --data data/paddy_disease
"""
from __future__ import annotations

import argparse
import json
import os
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO, "data", "paddy_disease")
MODELS_DIR = os.path.join(REPO, "models")
DEFAULT_MODEL = os.path.join(MODELS_DIR, "paddy_disease_cls.pt")
METRICS_PATH = os.path.join(MODELS_DIR, "paddy_disease_metrics.json")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")


def evaluate(model_path: str, data_dir: str, imgsz: int = 224) -> dict:
    from ultralytics import YOLO

    val_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(val_dir):
        raise SystemExit(f"No val split at {val_dir}. Run scripts/download_paddy_dataset.py first.")

    classes = sorted(d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d)))
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = [[0] * n for _ in range(n)]  # cm[true][pred]

    model = YOLO(model_path)
    total = correct = 0
    t0 = time.time()

    for cls in classes:
        cdir = os.path.join(val_dir, cls)
        imgs = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith(IMG_EXT)]
        for path in imgs:
            r = model.predict(path, imgsz=imgsz, verbose=False)[0]
            pred_name = r.names[int(r.probs.top1)]
            # map prediction back to one of our val classes by normalised name
            pred_norm = _norm(pred_name)
            pred_cls = next((c for c in classes if _norm(c) == pred_norm), None)
            if pred_cls is None:
                # unknown label (base model) — count as wrong, skip cm
                total += 1
                continue
            cm[idx[cls]][idx[pred_cls]] += 1
            total += 1
            if pred_cls == cls:
                correct += 1

    # per-class precision / recall / f1 from the confusion matrix
    per_class = []
    for c in classes:
        i = idx[c]
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(cm[r][i] for r in range(n)) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class.append({
            "class": c.replace("_", " "),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "support": sum(cm[i]),
        })

    top1 = correct / total if total else 0.0
    train_dir = os.path.join(data_dir, "train")
    train_count = sum(
        len([f for f in os.listdir(os.path.join(train_dir, c)) if f.lower().endswith(IMG_EXT)])
        for c in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, c))
    ) if os.path.isdir(train_dir) else 0

    return {
        "trained": True,
        "model": os.path.basename(model_path),
        "top1_accuracy": round(top1, 4),
        "target_accuracy": 0.75,
        "meets_target": top1 >= 0.75,
        "num_classes": n,
        "class_names": [c.replace("_", " ") for c in classes],
        "val_images": total,
        "train_images": train_count,
        "per_class": per_class,
        "confusion_matrix": cm,
        "eval_seconds": round(time.time() - t0, 1),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--imgsz", type=int, default=224)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}. Train it first "
                         "(scripts/train_yolo_classifier.py).")

    metrics = evaluate(args.model, args.data, args.imgsz)
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nTop-1 accuracy: {metrics['top1_accuracy'] * 100:.1f}%  "
          f"({'MEETS' if metrics['meets_target'] else 'BELOW'} 75% target)")
    print(f"Wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
