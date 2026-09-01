"""Standalone training script: trains and persists the crop-health stress classifier.

Usage:
    cd python/agents/crop_health_agent
    python3 train_models.py
"""
import os
import sys
import logging

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_store import CropHealthDataStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TRAIN] %(levelname)s %(message)s")
log = logging.getLogger("TRAIN")

BASE = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(BASE, "../../../data/Karnataka_Paddy_AI_ML_Dataset.xlsx")
CSV_PATH = os.path.join(BASE, "../../../data/Karnataka_Paddy_Main_Dataset.csv")
MODELS_DIR = os.path.join(BASE, "models")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    store = CropHealthDataStore(XLSX_PATH, CSV_PATH)

    model_path = os.path.join(MODELS_DIR, "stress_classifier.pkl")
    joblib.dump(store._clf, model_path)
    log.info("Saved stress classifier -> %s", model_path)

    acc = store.train_test_accuracy
    cm_path = os.path.join(MODELS_DIR, "confusion_matrix.csv")
    acc["confusion_matrix"].to_csv(cm_path)
    log.info("Saved confusion matrix -> %s", cm_path)

    log.info("Final: train_accuracy=%.3f  test_accuracy=%.3f (80/20 stratified split)",
              acc["train_accuracy"], acc["test_accuracy"])


if __name__ == "__main__":
    main()
