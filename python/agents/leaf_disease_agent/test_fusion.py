import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion import fuse
from treatment_map import TREATMENTS, lookup


def cnn(disease="Brown spot", confidence=0.9, inconclusive=False, cv=0.88):
    return {
        "disease": disease,
        "confidence": confidence,
        "inconclusive": inconclusive,
        "probabilities": {d: 0.1 for d in TREATMENTS},
        "model_cv_accuracy": cv,
    }


def test_image_only_when_no_weather():
    r = fuse(cnn())
    assert r["basis"] == "image_only"
    assert r["overall_risk"] == "Medium"  # Brown spot severity
    assert r["corroborated"] is False


def test_confident_detection_raises_risk_above_calm_rules():
    """A disease that is already present outranks a Low weather forecast."""
    r = fuse(cnn(disease="Bacterial leaf blight"), rule_risk="Low")
    assert r["overall_risk"] == "High"
    assert r["basis"] == "image+rules"
    assert r["corroborated"] is False


def test_agreement_escalates_and_corroborates():
    """Brown spot (Medium) plus elevated rules is genuine corroboration."""
    r = fuse(cnn(disease="Brown spot"), rule_risk="Medium", rule_pests=["Stem Borer"])
    assert r["corroborated"] is True
    assert r["overall_risk"] == "High"
    assert "Stem Borer" in r["advisory"]


def test_minor_disease_is_not_corroborated_by_unrelated_pest_pressure():
    """Leaf smut is cosmetic and its advisory says no treatment is warranted.

    The rulebook covers insect pests, so Medium pest pressure alongside it is a
    separate minor finding. Escalating to High would contradict the advisory
    text the farmer reads in the same response.
    """
    r = fuse(cnn(disease="Leaf smut"), rule_risk="Medium", rule_pests=["Stem Borer"])
    assert r["corroborated"] is False
    assert r["overall_risk"] == "Medium"  # from the rules, not the leaf
    assert "different concern" in r["advisory"]


def test_minor_disease_still_takes_high_from_high_rules():
    r = fuse(cnn(disease="Leaf smut"), rule_risk="High")
    assert r["overall_risk"] == "High"
    assert r["corroborated"] is False


def test_rules_cannot_lower_an_observed_high():
    r = fuse(cnn(disease="Bacterial leaf blight"), rule_risk="Medium")
    assert r["overall_risk"] == "High"


def test_inconclusive_falls_back_to_rules_without_diagnosing():
    r = fuse(cnn(confidence=0.4, inconclusive=True), rule_risk="Medium",
             rule_advisory="Monitor fields every 3 days.")
    assert r["detected_disease"] is None
    assert r["treatment"] is None
    assert r["overall_risk"] == "Medium"
    assert r["basis"] == "rules_only"
    assert "inconclusive" in r["advisory"].lower()


def test_inconclusive_with_no_rules_reports_unknown():
    r = fuse(cnn(confidence=0.4, inconclusive=True))
    assert r["overall_risk"] == "Unknown"
    assert r["basis"] == "none"
    assert r["confidence"] < 0.5


def test_corroboration_never_exceeds_confidence_cap():
    r = fuse(cnn(confidence=1.0, cv=1.0), rule_risk="High")
    assert r["confidence"] <= 0.95


def test_every_class_has_complete_advisory():
    for disease in TREATMENTS:
        entry = lookup(disease)
        assert entry["severity"] in ("Low", "Medium", "High")
        for field in ("pathogen", "signs", "action", "etl", "yield_impact"):
            assert entry[field].strip(), f"{disease} missing {field}"


def test_unknown_disease_rejected():
    with pytest.raises(ValueError):
        lookup("Rice Blast")
