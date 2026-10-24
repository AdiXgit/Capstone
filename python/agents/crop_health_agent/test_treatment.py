"""Unit tests for the (stress_level, growth_stage) -> KVK treatment mapping."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from treatment_map import (
    STRESS_LEVELS, TREATMENT_MAP, STAGE_ADVISORIES,
    get_treatment, crop_status_from_stress,
)
from stage_profiles import STAGE_ORDER


def test_treatment_map_covers_every_stress_level_x_stage_combo():
    for stage in STAGE_ADVISORIES:
        for level in STRESS_LEVELS:
            assert (level, stage) in TREATMENT_MAP
    assert len(TREATMENT_MAP) == len(STAGE_ADVISORIES) * len(STRESS_LEVELS)


def test_get_treatment_returns_nonempty_string_for_all_known_stages():
    for stage in STAGE_ORDER:
        for level in STRESS_LEVELS:
            treatment = get_treatment(level, stage)
            assert isinstance(treatment, str)
            assert len(treatment) > 0


def test_get_treatment_unknown_stage_falls_back_gracefully():
    treatment = get_treatment("Severe", "Not_A_Real_Stage")
    assert isinstance(treatment, str)
    assert "KVK" in treatment


def test_severe_treatment_mentions_kvk_contact():
    for stage in STAGE_ADVISORIES:
        assert "KVK" in get_treatment("Severe", stage)


def test_healthy_treatment_does_not_urge_urgent_action():
    for stage in STAGE_ADVISORIES:
        text = get_treatment("Healthy", stage).lower()
        assert "urgent" not in text


def test_crop_status_from_stress_mapping():
    assert crop_status_from_stress("Healthy") == "HEALTHY"
    assert crop_status_from_stress("Mild") == "HEALTHY"
    assert crop_status_from_stress("Moderate") == "STRESSED"
    assert crop_status_from_stress("Severe") == "DISEASED"
    assert crop_status_from_stress("Nonsense") == "UNKNOWN"
