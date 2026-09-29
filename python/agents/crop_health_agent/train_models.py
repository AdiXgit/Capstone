"""Persist the crop-health stress rule report.

Revision: the stress level is a deterministic NDVI-vs-benchmark rule, not a learned
model (see data_store.py for the leakage analysis), so there is no pickled model to
train. This script just writes the honest rule report for provenance.

Usage:
    cd python/agents/crop_health_agent
    python3 train_models.py
"""
import os
import sys
import json
import logging

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
    out = os.path.join(MODELS_DIR, "stress_rule.json")
    with open(out, "w") as f:
        json.dump(store.stress_model, f, indent=2)
    log.info("Saved stress rule report -> %s", out)
    log.info("Method: %s | distribution: %s",
             store.stress_model["method"], store.stress_model["label_distribution"])


if __name__ == "__main__":
    main()
