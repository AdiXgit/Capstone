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

FEATURES  = ["Soil_Nitrogen_kg_per_ha", "Soil_Phosphorus_kg_per_ha",
             "Soil_Potassium_kg_per_ha", "Soil_pH",
             "Soil_Organic_Carbon_Percent", "district_enc"]

TARGETS   = {
    "urea_kg_per_ha":   "Urea_Applied_kg_per_ha",
    "dap_kg_per_ha":    "DAP_Applied_kg_per_ha",
    "potash_kg_per_ha": "Potash_Applied_kg_per_ha",
}

os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    le = LabelEncoder()
    df["district_enc"] = le.fit_transform(df["District"])
    return df, le


def train_models():
    df, le = load_and_prepare_data()
    models = {}

    for model_name, col in TARGETS.items():
        X = df[FEATURES]
        y = df[col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        mae = mean_absolute_error(y_test, rf.predict(X_test))
        print(f"  {model_name}: MAE = {mae:.2f}")

        joblib.dump(rf, os.path.join(MODEL_DIR, f"{model_name}.pkl"))
        models[model_name] = rf

    joblib.dump(le, os.path.join(MODEL_DIR, "district_encoder.pkl"))
    print("Models saved to", MODEL_DIR)
    return models


def load_models():
    models = {}
    for model_name in TARGETS:
        path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        if os.path.exists(path):
            models[model_name] = joblib.load(path)
    le_path = os.path.join(MODEL_DIR, "district_encoder.pkl")
    le = joblib.load(le_path) if os.path.exists(le_path) else None
    return models, le


def predict_fertilizer(district: str, nitrogen: float, phosphorus: float,
                        potassium: float, ph: float, organic_carbon: float,
                        season: str) -> dict:
    models, le = load_models()

    if le is None or not models:
        raise RuntimeError("Models not trained yet. Run train_models.py first.")

    district_title = district.strip().title()
    if district_title not in le.classes_:
        district_title = le.classes_[0]

    district_enc = le.transform([district_title])[0]

    input_row = pd.DataFrame([[nitrogen, phosphorus, potassium,
                           ph, organic_carbon, district_enc]],
                          columns=FEATURES)

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
    result["season"] = season
    return result