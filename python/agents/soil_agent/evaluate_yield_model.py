"""
Honest evaluation of the Soil Agent's yield model.

Compares three ways of testing the SAME Random Forest:
  1. Random 5-fold CV       - rows shuffled; same district/year can be in train AND test
  2. Leave-districts-out    - whole districts held out  -> "a district the model never saw"
  3. Leave-years-out        - whole years held out      -> "a new season"

If 2 and 3 are much lower than 1, the headline R2 was optimistic.
Nothing is retrained or overwritten - this only evaluates.

Run from the soil_agent folder:
    python evaluate_yield_model.py
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "../../../data/Karnataka_Paddy_Main_Dataset.csv")
MODEL_PATH = os.path.join(HERE, "models/yield_model.pkl")

# Leave as None to auto-detect; set the exact column name if detection picks the wrong one
YIELD_COL = None
YEAR_COL = None

NUMERIC_FEATURES = [
    "Soil_Nitrogen_kg_per_ha", "Soil_Phosphorus_kg_per_ha",
    "Soil_Potassium_kg_per_ha", "Soil_pH",
    "Soil_Organic_Carbon_Percent", "Rainfall_mm",
    "Irrigation_Percent", "Temperature_Celsius",
]
CATEGORICAL = {"District": "district_enc", "Season": "season_enc",
               "Region": "region_enc", "Soil_Quality": "soil_quality_enc"}
FEATURES = NUMERIC_FEATURES + list(CATEGORICAL.values())


def find_column(df, keyword, override):
    if override:
        return override
    matches = [c for c in df.columns if keyword in c.lower()]
    if not matches:
        raise SystemExit(f"No column containing '{keyword}'. Columns are:\n{df.columns.tolist()}\n"
                         f"Set {keyword.upper()}_COL at the top of this file.")
    return matches[0]


def get_model():
    """Use the exact settings of the saved model if possible, so we test YOUR model."""
    try:
        saved = joblib.load(MODEL_PATH)
        if isinstance(saved, dict):
            saved = next((v for v in saved.values() if hasattr(v, "predict")), None)
        if saved is not None and hasattr(saved, "get_params"):
            print(f"Using settings from saved model: {type(saved).__name__}")
            return clone(saved)
    except Exception:
        pass
    print("Saved model not found - using RandomForestRegressor(n_estimators=200)")
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)


def evaluate(model, X, y, splits, name):
    r2s, maes = [], []
    all_true, all_pred = [], []
    for train_idx, test_idx in splits:
        m = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = m.predict(X.iloc[test_idx])
        r2s.append(r2_score(y.iloc[test_idx], pred))
        maes.append(mean_absolute_error(y.iloc[test_idx], pred))
        all_true.extend(y.iloc[test_idx]); all_pred.extend(pred)
    return {
        "Method": name,
        "R2 (mean)": np.mean(r2s),
        "R2 (min fold)": np.min(r2s),
        "MAE kg/ha": np.mean(maes),
        "R2 pooled": r2_score(all_true, all_pred),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    yield_col = find_column(df, "yield", YIELD_COL)
    year_col = find_column(df, "year", YEAR_COL)
    print(f"Rows: {len(df)} | target: '{yield_col}' | year column: '{year_col}'")

    missing = [c for c in NUMERIC_FEATURES + list(CATEGORICAL) if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected columns: {missing}")

    df = df.dropna(subset=NUMERIC_FEATURES + list(CATEGORICAL) + [yield_col]).reset_index(drop=True)
    for col, enc in CATEGORICAL.items():
        df[enc] = LabelEncoder().fit_transform(df[col].astype(str))

    X, y = df[FEATURES], df[yield_col]
    n_districts = df["District"].nunique()
    n_years = df[year_col].nunique()
    print(f"{n_districts} districts, {n_years} years "
          f"({df[year_col].min()}-{df[year_col].max()}), mean yield {y.mean():.0f} kg/ha\n")

    model = get_model()
    k_d = min(5, n_districts)
    k_y = min(5, n_years)

    results = [
        evaluate(model, X, y, KFold(5, shuffle=True, random_state=42).split(X),
                 "1. Random 5-fold"),
        evaluate(model, X, y, GroupKFold(k_d).split(X, y, df["District"]),
                 f"2. Leave-districts-out ({k_d} folds)"),
        evaluate(model, X, y, GroupKFold(k_y).split(X, y, df[year_col]),
                 f"3. Leave-years-out ({k_y} folds)"),
    ]
    res = pd.DataFrame(results)

    print("=" * 84)
    print(res.round(3).to_string(index=False))
    print("=" * 84)

    base = res.loc[0, "R2 (mean)"]
    print("\nWHAT THIS MEANS")
    for i, label in [(1, "new district"), (2, "new year")]:
        drop = base - res.loc[i, "R2 (mean)"]
        if drop > 0.10:
            verdict = "BIG drop - the random-split R2 is optimistic for this case"
        elif drop > 0.03:
            verdict = "small drop - normal, report both numbers"
        else:
            verdict = "holds up well"
        print(f"  {label:<13}: R2 {res.loc[i, 'R2 (mean)']:.3f} vs {base:.3f} random -> {verdict}")

    # Feature importance from a model on all data
    fitted = clone(model).fit(X, y)
    if hasattr(fitted, "feature_importances_"):
        imp = pd.Series(fitted.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("\nFEATURE IMPORTANCE (what the model relies on)")
        for f, v in imp.items():
            print(f"  {f:<30} {v:.3f} {'#' * int(v * 60)}")
        if imp.get("district_enc", 0) > 0.3:
            print("  ! Relies heavily on district ID - explains a drop for unseen districts.")


if __name__ == "__main__":
    main()