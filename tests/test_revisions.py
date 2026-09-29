"""Tests for the revisions: leaf-validity gate, honest treatment mapping,
deterministic stress rule, and the new Market/Pest agents."""
import os

import numpy as np
import pytest
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")


# ── Leaf-validity (vegetation) gate ──────────────────────────────
def test_vegetation_gate_separates_leaf_from_nonleaf():
    from disease_detection import vegetation_index, LEAF_REJECT

    green = Image.new("RGB", (256, 256), (60, 160, 60))   # foliage-like
    gray = Image.new("RGB", (256, 256), (128, 128, 128))  # wall
    skin = Image.new("RGB", (256, 256), (210, 150, 120))  # face-like

    assert vegetation_index(green)["green_fraction"] >= LEAF_REJECT
    assert vegetation_index(gray)["green_fraction"] < LEAF_REJECT
    assert vegetation_index(skin)["green_fraction"] < LEAF_REJECT


def test_vegetation_gate_on_a_noisy_green_leaf():
    from disease_detection import vegetation_index, LEAF_REJECT
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 60, (256, 256, 3), dtype=np.uint8)
    arr[..., 1] += 120  # boost green channel → vegetation
    assert vegetation_index(Image.fromarray(arr))["green_fraction"] >= LEAF_REJECT


# ── Treatment mapping (distinct blast types, OOD handling) ────────
def test_treatment_distinguishes_blast_types():
    from disease_treatment import lookup
    assert lookup("Leaf_Blast")["label"] == "Leaf Blast"
    assert lookup("Neck_Blast")["label"] == "Neck Blast"
    assert lookup("Leaf_Blast")["matched"] and lookup("Neck_Blast")["matched"]


def test_treatment_healthy_alias_and_unknown():
    from disease_treatment import lookup
    assert lookup("Healthy")["label"] == "Healthy Leaf"
    unknown = lookup("some_random_object")
    assert unknown["matched"] is False


# ── Deterministic stress rule (leakage removed) ──────────────────
def test_stress_rule_is_monotonic_in_ndvi():
    from data_store import _label_from_ndvi
    from stage_profiles import STAGE_PROFILES
    q25, med, q75 = STAGE_PROFILES["Grain_Filling"]["ndvi"]
    order = {"Healthy": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
    assert order[_label_from_ndvi(q75 + 0.05, "Grain_Filling")] == 0
    assert order[_label_from_ndvi(max(0.0, q25 - 0.1), "Grain_Filling")] == 3


# ── New agents' domain logic (no server needed) ──────────────────
def test_market_store_advisory_shape():
    from market_agent import MarketStore
    a = MarketStore(os.path.join(DATA, "Karnataka_Market_Prices.csv")).advisory("")
    assert a["sell"] in ("SELL", "HOLD")
    assert a["direction"] in ("RISING", "FALLING", "STABLE")
    assert a["current"] > 0


def test_pest_store_history_shape():
    from pest_agent import PestStore
    h = PestStore(os.path.join(DATA, "Karnataka_Paddy_Main_Dataset.csv")).history("Mandya", "Kharif")
    assert set(h.keys()) == {"pest", "disease"}
