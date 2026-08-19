# python/agents/soil_agent/train_models.py
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

# Must match the YIELD_FEATURES list inside fertilizer_model.py exactly
YIELD_FEATURES = [
    "Soil_Nitrogen_kg_per_ha", "Soil_Phosphorus_kg_per_ha",
    "Soil_Potassium_kg_per_ha", "Soil_pH",
    "Soil_Organic_Carbon_Percent", "Rainfall_mm",
    "Irrigation_Percent", "Temperature_Celsius",
    "district_enc", "season_enc", "region_enc", "soil_quality_enc"
]

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

    preds = rf.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

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


if __name__ == "__main__":
    train_models()