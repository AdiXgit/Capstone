import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../data/Karnataka_Paddy_Main_Dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

FEATURES = ["Soil_Nitrogen_kg_per_ha", "Soil_Phosphorus_kg_per_ha",
            "Soil_Potassium_kg_per_ha", "Soil_pH",
            "Soil_Organic_Carbon_Percent", "district_enc",
            "season_enc", "region_enc", "soil_quality_enc",
            "Rainfall_mm", "Irrigation_Percent"]

TARGETS = {
    "urea_kg_per_ha":   "Urea_Applied_kg_per_ha",
    "dap_kg_per_ha":    "DAP_Applied_kg_per_ha",
    "potash_kg_per_ha": "Potash_Applied_kg_per_ha",
}

os.makedirs(MODEL_DIR, exist_ok=True)


def load_models():
    models = {}
    for model_name in TARGETS:
        path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        if os.path.exists(path):
            models[model_name] = joblib.load(path)
    enc_path = os.path.join(MODEL_DIR, "encoders.pkl")
    encoders = joblib.load(enc_path) if os.path.exists(enc_path) else None
    return models, encoders


def predict_fertilizer(district: str, nitrogen: float, phosphorus: float,
                        potassium: float, ph: float, organic_carbon: float,
                        season: str, region: str = "South",
                        soil_quality: str = "medium",
                        rainfall: float = 900.0,
                        irrigation: float = 65.0) -> dict:

    models, encoders = load_models()

    if encoders is None or not models:
        raise RuntimeError("Models not trained yet. Run train_models.py first.")

    def safe_encode(le, val, fallback=0):
        return int(le.transform([val])[0]) if val in le.classes_ else fallback

    district_enc     = safe_encode(encoders["district"], district.strip().title())
    season_enc       = safe_encode(encoders["season"],   season.strip().title())
    region_enc       = safe_encode(encoders["region"],   region.strip().title())
    soil_quality_enc = safe_encode(encoders["quality"],  soil_quality.strip().lower())

    input_row = pd.DataFrame(
        [[nitrogen, phosphorus, potassium, ph, organic_carbon,
          district_enc, season_enc, region_enc,
          soil_quality_enc, rainfall, irrigation]],
        columns=FEATURES
    )

    result = {}
    for model_name in TARGETS:
        if model_name in models:
            result[model_name] = round(float(models[model_name].predict(input_row)[0]), 2)
        else:
            result[model_name] = 0.0

    if season.lower() == "kharif":
        timing = (
            "Apply 50% Urea as basal dose at transplanting, "
            "remaining 50% at active tillering (25-30 days). "
            "Apply full DAP and Potash as basal."
        )
    else:
        timing = (
            "Apply 40% Urea as basal at sowing, 30% at tillering, "
            "30% at panicle initiation. "
            "Apply full DAP and Potash as basal dose."
        )

    result["timing_advice"] = timing
    result["season"]        = season
    return result