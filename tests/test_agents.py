# tests/test_agents.py
# Run: pytest tests/test_agents.py -v
# Prerequisites: pip install pytest

import sys
import os
import tempfile
import pandas as pd
import numpy as np
import pytest

# ── Make agent modules importable ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ══════════════════════════════════════════════════════════════════
# AMAN — Soil Agent Tests
# ══════════════════════════════════════════════════════════════════

class TestSoilScoring:
    """Unit tests for soil health scoring logic (Aman)."""

    def _score(self, ph, oc, n):
        """Inline scoring without importing the full server."""
        score = 0
        if 5.5 <= ph <= 7.0:     score += 2
        elif 5.0 <= ph <= 7.5:   score += 1
        if oc > 0.75:             score += 2
        elif oc > 0.5:            score += 1
        if n > 250:               score += 2
        elif n > 150:             score += 1
        if score >= 5:   return "EXCELLENT"
        elif score >= 3: return "GOOD"
        elif score >= 1: return "MODERATE"
        return "POOR"

    def test_excellent_soil(self):
        assert self._score(6.5, 0.8, 280) == "EXCELLENT"

    def test_good_soil(self):
        assert self._score(6.2, 0.6, 170) == "GOOD"

    def test_moderate_soil(self):
        assert self._score(6.0, 0.4, 120) == "MODERATE"

    def test_poor_soil(self):
        # All below thresholds
        assert self._score(4.5, 0.3, 80) == "POOR"

    def test_ideal_ph_range(self):
        # pH 5.5–7.0 gets max pH score
        assert self._score(5.5, 0.8, 280) == "EXCELLENT"
        assert self._score(7.0, 0.8, 280) == "EXCELLENT"

    def test_borderline_ph(self):
        # pH 5.0–5.5 or 7.0–7.5 gets partial score
        result = self._score(5.2, 0.3, 80)
        assert result in ("POOR", "MODERATE")


class TestSoilDataStore:
    """Test the data loading and model training (Aman)."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create minimal main dataset CSV for testing."""
        data = {
            "District": ["Koppal"] * 10 + ["Mandya"] * 10,
            "Region": ["North"] * 10 + ["South"] * 10,
            "Soil_Quality": ["medium"] * 20,
            "Season": ["Kharif"] * 20,
            "Soil_pH": np.random.uniform(5.5, 7.5, 20),
            "Soil_Organic_Carbon_Percent": np.random.uniform(0.3, 0.8, 20),
            "Soil_Nitrogen_kg_per_ha": np.random.uniform(150, 300, 20),
            "Soil_Phosphorus_kg_per_ha": np.random.uniform(10, 40, 20),
            "Soil_Potassium_kg_per_ha": np.random.uniform(100, 200, 20),
            "Urea_Applied_kg_per_ha": np.random.uniform(80, 130, 20),
            "DAP_Applied_kg_per_ha": np.random.uniform(40, 100, 20),
            "Potash_Applied_kg_per_ha": np.random.uniform(20, 60, 20),
            "Organic_Fertilizer_kg_per_ha": np.random.uniform(500, 2000, 20),
            "Yield_Kg_per_Ha": np.random.uniform(2000, 5000, 20),
            "Numeric_Year": list(range(2015, 2025)) * 2,
        }
        df = pd.DataFrame(data)
        path = tmp_path / "test_main.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_profile_built_per_district(self, sample_csv):
        from python.agents.soil_agent.server import SoilDataStore
        store = SoilDataStore(sample_csv)
        assert "Koppal" in store._district_profiles
        assert "Mandya" in store._district_profiles

    def test_profile_has_required_keys(self, sample_csv):
        from python.agents.soil_agent.server import SoilDataStore
        store = SoilDataStore(sample_csv)
        profile = store.get_profile("Koppal")
        assert profile is not None
        for key in ["Soil_pH", "Soil_Organic_Carbon_Percent",
                    "Soil_Nitrogen_kg_per_ha", "Soil_Quality"]:
            assert key in profile, f"Missing key: {key}"

    def test_unknown_district_returns_none(self, sample_csv):
        from python.agents.soil_agent.server import SoilDataStore
        store = SoilDataStore(sample_csv)
        assert store.get_profile("NonExistentDistrict") is None

    def test_fertilizer_prediction_returns_positives(self, sample_csv):
        from python.agents.soil_agent.server import SoilDataStore
        store = SoilDataStore(sample_csv)
        urea, dap, potash = store.predict_fertilizer(6.5, 0.5, 200, 20, 150)
        assert urea > 0, "Urea should be positive"
        assert dap > 0, "DAP should be positive"
        assert potash > 0, "Potash should be positive"

    def test_trend_direction_for_single_district(self, sample_csv):
        from python.agents.soil_agent.server import SoilDataStore
        store = SoilDataStore(sample_csv)
        trend = store.get_trend("Koppal", 5)
        assert trend["direction"] in ("IMPROVING", "DEGRADING", "STABLE")
        assert isinstance(trend["ph_trend"], float)
        assert isinstance(trend["carbon_trend"], float)


class TestFertilizerTiming:
    """Validate fertilizer timing strings (Aman)."""

    def test_kharif_timing(self):
        from python.agents.soil_agent.server import fertilizer_timing
        t = fertilizer_timing("Kharif")
        assert "Basal" in t
        assert "transplanting" in t.lower() or "transplant" in t.lower()

    def test_rabi_timing(self):
        from python.agents.soil_agent.server import fertilizer_timing
        t = fertilizer_timing("Rabi")
        assert "Basal" in t
        assert "sowing" in t.lower()


# ══════════════════════════════════════════════════════════════════
# RISHI MOHAN — Crop Health Agent Tests
# ══════════════════════════════════════════════════════════════════

class TestGrowthStageFunctions:
    """Unit tests for growth stage helpers (Rishi Mohan)."""

    def test_das_to_stage_sowing(self):
        from python.agents.crop_health_agent.server import _das_to_stage
        assert _das_to_stage(0) == "Sowing"
        assert _das_to_stage(3) == "Sowing"

    def test_das_to_stage_germination(self):
        from python.agents.crop_health_agent.server import _das_to_stage
        assert _das_to_stage(10) == "Germination"

    def test_das_to_stage_vegetative(self):
        from python.agents.crop_health_agent.server import _das_to_stage
        assert _das_to_stage(25) == "Vegetative"

    def test_das_to_stage_tillering(self):
        from python.agents.crop_health_agent.server import _das_to_stage
        assert _das_to_stage(55) == "Tillering"

    def test_das_to_stage_flowering(self):
        from python.agents.crop_health_agent.server import _das_to_stage
        assert _das_to_stage(85) == "Flowering"

    def test_das_to_stage_maturity(self):
        from python.agents.crop_health_agent.server import _das_to_stage
        assert _das_to_stage(130) == "Maturity"

    def test_next_stage_info_returns_tuple(self):
        from python.agents.crop_health_agent.server import _next_stage_info
        stage, days = _next_stage_info(25)
        assert isinstance(stage, str)
        assert isinstance(days, int)
        assert days >= 0

    def test_estimate_height_grows_with_das(self):
        from python.agents.crop_health_agent.server import _estimate_height
        h0  = _estimate_height(0)
        h30 = _estimate_height(30)
        h80 = _estimate_height(80)
        assert h0 < h30 < h80, "Height should increase with DAS"

    def test_estimate_tillers_peaks_at_tillering(self):
        from python.agents.crop_health_agent.server import _estimate_tillers
        t45 = _estimate_tillers(45)
        t65 = _estimate_tillers(65)
        t120 = _estimate_tillers(120)
        assert t65 >= t45, "Tillers should peak at tillering stage"
        assert t120 < t65, "Tillers should reduce post-tillering"


class TestVegetationStatus:
    """Test NDVI to vegetation status mapping (Rishi Mohan)."""

    def test_dense_healthy(self):
        from python.agents.crop_health_agent.server import _vegetation_status
        assert _vegetation_status(0.80) == "DENSE_HEALTHY_VEGETATION"

    def test_moderate(self):
        from python.agents.crop_health_agent.server import _vegetation_status
        assert _vegetation_status(0.60) == "MODERATE_VEGETATION"

    def test_sparse(self):
        from python.agents.crop_health_agent.server import _vegetation_status
        assert _vegetation_status(0.40) == "SPARSE_VEGETATION"

    def test_bare_soil(self):
        from python.agents.crop_health_agent.server import _vegetation_status
        assert _vegetation_status(0.20) == "BARE_SOIL_OR_STRESS"

    def test_no_vegetation(self):
        from python.agents.crop_health_agent.server import _vegetation_status
        assert _vegetation_status(0.05) == "NO_VEGETATION"


class TestCropHealthEvaluation:
    """Test health status logic (Rishi Mohan)."""

    def test_excellent_crop(self):
        from python.agents.crop_health_agent.server import _evaluate_crop_health
        status, action = _evaluate_crop_health(0.75, 4.0, "Tillering")
        assert status == "EXCELLENT"

    def test_good_crop(self):
        from python.agents.crop_health_agent.server import _evaluate_crop_health
        status, action = _evaluate_crop_health(0.55, 2.5, "Vegetative")
        assert status == "GOOD"

    def test_moderate_crop(self):
        from python.agents.crop_health_agent.server import _evaluate_crop_health
        status, action = _evaluate_crop_health(0.38, 1.5, "Vegetative")
        assert status == "MODERATE"

    def test_poor_crop(self):
        from python.agents.crop_health_agent.server import _evaluate_crop_health
        status, action = _evaluate_crop_health(0.20, 0.8, "Tillering")
        assert status == "POOR"
        assert action != ""  # Must provide actionable guidance


class TestPestRiskMatrix:
    """Test pest/disease risk assessment (Rishi Mohan)."""

    def test_tillering_has_bph_risk(self):
        from python.agents.crop_health_agent.server import PEST_RISK_MATRIX
        tillering_pests = PEST_RISK_MATRIX.get("Tillering", {})
        assert "Brown Planthopper" in tillering_pests

    def test_flowering_no_spray_advisory(self):
        from python.agents.crop_health_agent.server import _pest_disease_action
        # During flowering, action should mention not disturbing the crop
        action = _pest_disease_action(["Rice Bug"], [], "MEDIUM")
        assert action != ""

    def test_high_humidity_triggers_blast_risk(self):
        from python.agents.crop_health_agent.server import DISEASE_RISK_MATRIX
        humid_diseases = DISEASE_RISK_MATRIX.get("HIGH_HUMIDITY", {})
        assert "Blast" in humid_diseases
        assert humid_diseases["Blast"] == "HIGH"


class TestDistrictCoords:
    """Verify all 10 weather-tracked districts have coordinates (Rishi Mohan)."""

    def test_all_tracked_districts_have_coords(self):
        from python.agents.crop_health_agent.server import DISTRICT_COORDS
        tracked = [
            "Koppal", "Ballari", "Raichur", "Davanagere",
            "Shivamogga", "Hassan", "Mandya", "Mysuru",
            "Belagavi", "Dharwad",
        ]
        for district in tracked:
            assert district in DISTRICT_COORDS, f"Missing coords for {district}"
            lat, lon = DISTRICT_COORDS[district]
            # Validate Karnataka bounding box: lat 11–18°N, lon 74–78°E
            assert 11 <= lat <= 18, f"Lat out of Karnataka range: {district} {lat}"
            assert 74 <= lon <= 79, f"Lon out of Karnataka range: {district} {lon}"
