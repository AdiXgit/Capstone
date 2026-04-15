import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../data/Karnataka_Paddy_Main_Dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

YIELD_FEATURES = [
    "Soil_Nitrogen_kg_per_ha", "Soil_Phosphorus_kg_per_ha",
    "Soil_Potassium_kg_per_ha", "Soil_pH",
    "Soil_Organic_Carbon_Percent", "Rainfall_mm",
    "Irrigation_Percent", "Temperature_Celsius",
    "district_enc", "season_enc", "region_enc", "soil_quality_enc"
]

KVK_FERTILIZER_GUIDELINES = {
    "high":   {"urea_kg_per_ha": 100, "dap_kg_per_ha": 50,  "potash_kg_per_ha": 30},
    "medium": {"urea_kg_per_ha": 130, "dap_kg_per_ha": 65,  "potash_kg_per_ha": 40},
    "low":    {"urea_kg_per_ha": 160, "dap_kg_per_ha": 80,  "potash_kg_per_ha": 50},
}

os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    le_district = LabelEncoder()
    le_season   = LabelEncoder()
    le_region   = LabelEncoder()
    le_quality  = LabelEncoder()

    df["district_enc"]     = le_district.fit_transform(df["District"])
    df["season_enc"]       = le_season.fit_transform(df["Season"])
    df["region_enc"]       = le_region.fit_transform(df["Region"])
    df["soil_quality_enc"] = le_quality.fit_transform(df["Soil_Quality"])

    encoders = {
        "district": le_district,
        "season":   le_season,
        "region":   le_region,
        "quality":  le_quality,
    }
    return df, encoders


def train_models():
    df, encoders = load_and_prepare_data()

    print("=" * 55)
    print("SOIL AGENT — YIELD PREDICTION MODEL")
    print("=" * 55)
    print(f"Dataset:   {len(df)} records")
    print(f"Districts: {df['District'].nunique()}")
    print(f"Target:    Yield_Kg_per_Ha")
    print()

    X = df[YIELD_FEATURES]
    y = df["Yield_Kg_per_Ha"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    preds  = rf.predict(X_test)
    mae    = mean_absolute_error(y_test, preds)
    r2     = r2_score(y_test, preds)

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="neg_mean_absolute_error")
    cv_mae    = -cv_scores.mean()

    importances = pd.Series(rf.feature_importances_, index=YIELD_FEATURES)
    top = importances.sort_values(ascending=False)

    print(f"  MAE:             {mae:.2f} kg/ha")
    print(f"  R² Score:        {r2:.4f}")
    print(f"  CV MAE (5-fold): {cv_mae:.2f} kg/ha")
    print()
    print("  Feature Importances:")
    for feat, imp in top.items():
        print(f"    {feat:<35} {imp*100:.1f}%")

    joblib.dump(rf,       os.path.join(MODEL_DIR, "yield_model.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))

    print()
    print("=" * 55)
    print("Model saved to", MODEL_DIR)
    print("=" * 55)
    return rf


def load_models():
    model_path = os.path.join(MODEL_DIR, "yield_model.pkl")
    enc_path   = os.path.join(MODEL_DIR, "encoders.pkl")
    model    = joblib.load(model_path) if os.path.exists(model_path) else None
    encoders = joblib.load(enc_path)   if os.path.exists(enc_path)   else None
    return model, encoders


def get_fertilizer_recommendation(soil_quality: str, season: str) -> dict:
    quality = soil_quality.strip().lower()
    if quality not in KVK_FERTILIZER_GUIDELINES:
        quality = "medium"

    base = KVK_FERTILIZER_GUIDELINES[quality].copy()

    if season.strip().lower() == "rabi":
        base["urea_kg_per_ha"]   = round(base["urea_kg_per_ha"]   * 0.85, 1)
        base["dap_kg_per_ha"]    = round(base["dap_kg_per_ha"]    * 0.90, 1)
        base["potash_kg_per_ha"] = round(base["potash_kg_per_ha"] * 0.90, 1)
        timing = (
            "Apply 40% Urea as basal at sowing, 30% at tillering, "
            "30% at panicle initiation. "
            "Apply full DAP and Potash as basal dose."
        )
    else:
        timing = (
            "Apply 50% Urea as basal dose at transplanting, "
            "remaining 50% at active tillering (25-30 days). "
            "Apply full DAP and Potash as basal."
        )

    base["timing_advice"] = timing
    base["season"]        = season
    return base


def predict_yield(district: str, nitrogen: float, phosphorus: float,
                  potassium: float, ph: float, organic_carbon: float,
                  rainfall: float, irrigation: float, temperature: float,
                  season: str, region: str = "South",
                  soil_quality: str = "medium") -> dict:

    model, encoders = load_models()

    if model is None or encoders is None:
        raise RuntimeError("Model not trained yet. Run train_models.py first.")

    def safe_encode(le, val, fallback=0):
        return int(le.transform([val])[0]) if val in le.classes_ else fallback

    district_enc     = safe_encode(encoders["district"], district.strip().title())
    season_enc       = safe_encode(encoders["season"],   season.strip().title())
    region_enc       = safe_encode(encoders["region"],   region.strip().title())
    soil_quality_enc = safe_encode(encoders["quality"],  soil_quality.strip().lower())

    input_row = pd.DataFrame(
        [[nitrogen, phosphorus, potassium, ph, organic_carbon,
          rainfall, irrigation, temperature,
          district_enc, season_enc, region_enc, soil_quality_enc]],
        columns=YIELD_FEATURES
    )

    predicted_yield = round(float(model.predict(input_row)[0]), 2)

    if predicted_yield >= 3500:
        interpretation = "Excellent yield potential"
    elif predicted_yield >= 2500:
        interpretation = "Good yield potential"
    elif predicted_yield >= 1500:
        interpretation = "Average yield — consider soil improvement"
    else:
        interpretation = "Low yield — immediate intervention needed"

    return {
        "predicted_yield_kg_per_ha": predicted_yield,
        "interpretation":            interpretation,
        "district":                  district,
        "season":                    season,
    }