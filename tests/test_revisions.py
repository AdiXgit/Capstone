"""Tests for the revisions: leaf-validity gate, honest treatment mapping,
deterministic stress rule, and the new Market/Pest agents."""
import os

import numpy as np
import pytest
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")


# ── Botanical plant gate ─────────────────────────────────────────
def _gate(img):
    from disease_detection import vegetation_index, plant_gate
    return plant_gate(vegetation_index(img))


def _leafy(seed=0):
    """A textured, daylight-bright green patch that stands in for foliage."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 70, (256, 256, 3), dtype=np.uint8)
    arr[..., 1] = np.clip(arr[..., 1].astype(int) + 150, 0, 255).astype(np.uint8)
    arr[..., 0] = np.clip(arr[..., 0].astype(int) + 70, 0, 255).astype(np.uint8)
    arr[..., 2] = np.clip(arr[..., 2].astype(int) + 55, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def test_gate_accepts_textured_foliage():
    assert _gate(_leafy())["is_plant"] is True


@pytest.mark.parametrize("colour", [(128, 128, 128), (210, 150, 120), (120, 170, 235)])
def test_gate_rejects_flat_nonplant_surfaces(colour):
    """Walls, skin and sky have no leaf texture — the classifier must not run."""
    assert _gate(Image.new("RGB", (256, 256), colour))["is_plant"] is False


def test_gate_rejects_flat_green_surface():
    """A painted green wall is green but has zero texture — the old single-cue
    Excess-Green gate passed it."""
    g = _gate(Image.new("RGB", (256, 256), (90, 140, 85)))
    assert g["is_plant"] is False
    assert any("texture" in r for r in g["reasons"])


def test_gate_rejects_rgb_keyboard_regression():
    """Regression: an RGB-backlit keyboard scored green_fraction=0.38 under the old
    gate and was reported as 'Neck Blast, HIGH, 62%'. Saturated multi-hue lighting
    on a dark scene must be rejected."""
    rng = np.random.default_rng(1)
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(0, 256, 12):  # vivid rainbow key stripes on a dark desk
        h = rng.random()
        c = np.clip([abs(h * 6 - 3) - 1, 2 - abs(h * 6 - 2), 2 - abs(h * 6 - 4)], 0, 1)
        arr[:, i:i + 7] = (c * 230).astype(np.uint8)
    g = _gate(Image.fromarray(arr))
    assert g["is_plant"] is False
    assert any("multi-coloured" in r or "dark" in r for r in g["reasons"])


def test_gate_rejects_greyscale_screenshot():
    """A black-on-white document/UI capture has texture but no colour at all."""
    arr = np.full((256, 256, 3), 247, dtype=np.uint8)
    for i in range(20, 236, 26):
        arr[i:i + 7, 25:216] = 60
    g = _gate(Image.fromarray(arr))
    assert g["is_plant"] is False
    assert any("colour" in r for r in g["reasons"])


def test_gate_rejects_underexposed_frame():
    dark = np.asarray(_leafy()).astype(np.float32) * 0.25
    assert _gate(Image.fromarray(dark.astype(np.uint8)))["is_plant"] is False


# ── Prediction confidence gate ───────────────────────────────────
def test_confidence_grades_by_separation():
    from disease_detection import grade_confidence, _distribution_stats
    confident = grade_confidence(_distribution_stats(np.array([0.97, 0.02, 0.005, 0.005])))
    assert confident["level"] == "CONFIRMED" and confident["accepted"]

    # The exact distribution the keyboard produced: 0.62 / 0.20 / 0.14.
    diffuse = grade_confidence(_distribution_stats(np.array([0.62, 0.20, 0.14, 0.04])))
    assert diffuse["level"] == "ABSTAIN" and not diffuse["accepted"]


def test_provisional_severity_is_capped():
    """A merely-provisional call must not render as a red HIGH badge."""
    from disease_detection import _cap_severity
    assert _cap_severity("HIGH", "PROVISIONAL") == "MEDIUM"
    assert _cap_severity("HIGH", "CONFIRMED") == "HIGH"


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
