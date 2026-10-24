from npk_profiles import get_npk_for_district
from fertilizer_model import get_fertilizer_recommendation, predict_yield
from soil_trend import get_soil_trend

def test_npk_profile_returns_valid_data():
    result = get_npk_for_district("Mandya")
    assert "N" in result and "P" in result and "K" in result
    assert result["N"] > 0

def test_fertilizer_recommendation_kharif():
    result = get_fertilizer_recommendation("medium", "Kharif")
    assert result["urea_kg_per_ha"] > 0
    assert "timing_advice" in result

def test_fertilizer_recommendation_rabi_reduces_urea():
    kharif = get_fertilizer_recommendation("medium", "Kharif")
    rabi = get_fertilizer_recommendation("medium", "Rabi")
    assert rabi["urea_kg_per_ha"] < kharif["urea_kg_per_ha"]

def test_soil_trend_returns_interpretation():
    result = get_soil_trend("Mandya")
    assert "interpretation" in result
    assert "ph_slope" in result

def test_predict_yield_returns_positive():
    result = predict_yield(
        district="Mandya", nitrogen=218, phosphorus=20, potassium=185,
        ph=6.2, organic_carbon=0.35, rainfall=900, irrigation=65,
        temperature=25, season="Kharif"
    )
    assert result["predicted_yield_kg_per_ha"] > 0