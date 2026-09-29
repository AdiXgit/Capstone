#!/usr/bin/env python3
"""Fine-tune a YOLOv8 classification model on the paddy leaf disease dataset.

Prereq: run scripts/download_paddy_dataset.py first so data/paddy_disease/
contains train/ and val/ class folders.

The trained weights are copied to models/paddy_disease_cls.pt, which
disease_detection.py picks up automatically (no code change needed).

Usage
-----
    python scripts/train_yolo_classifier.py                 # sensible defaults
    python scripts/train_yolo_classifier.py --epochs 30 --imgsz 224 --model yolov8s-cls.pt

On a CPU-only laptop keep epochs low (10-20) and imgsz 128-224; a GPU is much
faster. The pipeline works either way.
"""
import argparse
import os
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO, "data", "paddy_disease")
MODELS_DIR = os.path.join(REPO, "models")
TARGET = os.path.join(MODELS_DIR, "paddy_disease_cls.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov8n-cls.pt", help="Base classification model")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--data", default=DATA_DIR)
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(args.data, "train")):
        sys.exit(f"No training data at {args.data}/train. "
                 "Run scripts/download_paddy_dataset.py first.")

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install -r requirements-vision.txt")

    print(f"Training {args.model} on {args.data} for {args.epochs} epochs (imgsz={args.imgsz})")
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=os.path.join(REPO, "runs", "paddy_cls"),
        name="train",
        exist_ok=True,
    )

    # Locate best.pt from the run and publish it where the detector looks.
    best = os.path.join(str(results.save_dir), "weights", "best.pt")
    if not os.path.exists(best):
        sys.exit(f"Training finished but best.pt not found at {best}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    shutil.copy2(best, TARGET)
    print(f"\nSaved fine-tuned model → {TARGET}")
    print("disease_detection.py will now use it automatically.")

    # Evaluate on the val split and write metrics for the dashboard.
    try:
        import json
        from evaluate_yolo import evaluate, METRICS_PATH
        print("\nEvaluating on the val split…")
        metrics = evaluate(TARGET, args.data, args.imgsz)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Top-1 accuracy: {metrics['top1_accuracy'] * 100:.1f}%  "
              f"({'MEETS' if metrics['meets_target'] else 'BELOW'} 75% target) → {METRICS_PATH}")
    except Exception as e:
        print(f"(Metrics step skipped: {e}. Run scripts/evaluate_yolo.py manually.)")


if __name__ == "__main__":
    main()
