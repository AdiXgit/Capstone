"""Unit tests for the crop-health stress classifier (data_store.py)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_store import CropHealthDataStore, STRESS_LEVELS, _label_from_ndvi
from stage_profiles import STAGE_PROFILES

BASE = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(BASE, "../../../data/Karnataka_Paddy_AI_ML_Dataset.xlsx")
CSV_PATH = os.path.join(BASE, "../../../data/Karnataka_Paddy_Main_Dataset.csv")


@pytest.fixture(scope="module")
def store():
    return CropHealthDataStore(XLSX_PATH, CSV_PATH)


def test_label_from_ndvi_covers_all_four_levels():
    q25, median, q75 = STAGE_PROFILES["Vegetative"]["ndvi"]
    assert _label_from_ndvi(q75 + 0.05, "Vegetative") == "Healthy"
    assert _label_from_ndvi(median + 0.001, "Vegetative") == "Mild"
    assert _label_from_ndvi(q25 + 0.001, "Vegetative") == "Moderate"
    assert _label_from_ndvi(q25 - 0.05, "Vegetative") == "Severe"


def test_classifier_trained_with_80_20_stratified_split(store):
    acc = store.train_test_accuracy
    assert 0.0 <= acc["train_accuracy"] <= 1.0
    assert 0.0 <= acc["test_accuracy"] <= 1.0
    # 1400 growth-stage records -> 20% held out is 280
    cm = acc["confusion_matrix"]
    assert cm.values.sum() == 280


def test_predict_stress_returns_valid_label(store):
    prof = STAGE_PROFILES["Flowering"]
    q25, median, q75 = prof["ndvi"]
    label = store.predict_stress(
        ndvi=q75, lai=5.0, das=55, height=90, tillers=15, stage="Flowering"
    )
    assert label in STRESS_LEVELS


def test_predict_stress_low_ndvi_is_worse_than_high_ndvi(store):
    prof = STAGE_PROFILES["Grain_Filling"]
    q25, median, q75 = prof["ndvi"]
    healthy = store.predict_stress(q75, 4.5, 80, 100, 16, "Grain_Filling")
    severe = store.predict_stress(max(0.0, q25 - 0.1), 4.5, 80, 100, 16, "Grain_Filling")
    order = {"Healthy": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
    assert order[severe] >= order[healthy]


def test_get_closest_growth_record_returns_expected_keys(store):
    rec = store.get_closest_growth_record("Mandya", "Kharif", 60)
    assert rec is not None
    for key in ("growth_stage", "days_after_sowing", "plant_height_cm",
                "tiller_count", "leaf_area_index", "ndvi"):
        assert key in rec
