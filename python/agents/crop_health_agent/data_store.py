import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from stage_profiles import STAGE_ORDER, STAGE_PROFILES, stage_index, stage_ndvi_median

log = logging.getLogger("CROP_HEALTH.data_store")

STRESS_LEVELS = ["Healthy", "Mild", "Moderate", "Severe"]
FEATURE_COLUMNS = ["ndvi", "lai", "das", "height", "tillers", "ndvi_deviation", "stage_idx"]


def _label_from_ndvi(ndvi, stage):
    """4-level stress label from NDVI deviation against the stage's quartile baseline."""
    q25, median, q75 = STAGE_PROFILES.get(stage, {}).get("ndvi", (0, 0, 0))
    if median == 0:
        return "Healthy" if ndvi == 0 else "Mild"
    if ndvi >= q75:
        return "Healthy"
    if ndvi >= median:
        return "Mild"
    if ndvi >= q25:
        return "Moderate"
    return "Severe"


class CropHealthDataStore:
    """Loads the Karnataka paddy datasets, engineers the NDVI-deviation feature,
    and trains a Gradient Boosting stress classifier (4 classes, 80/20 stratified split).
    """

    def __init__(self, xlsx_path: str, main_csv_path: str):
        log.info("Loading crop health datasets...")
        xl = pd.ExcelFile(xlsx_path)
        self.gs_df = xl.parse("Crop_Growth_Stages")
        self.sat_df = xl.parse("Satellite_Indices")
        self.main_df = pd.read_csv(main_csv_path, usecols=[
            "District", "Season", "Pest_Incidence", "Disease_Incidence",
            "Seed_Variety", "Humidity_Percent", "Temperature_Celsius", "Pesticide_Sprays",
        ])
        log.info("  Growth stages: %d  Satellite: %d  Main: %d",
                  len(self.gs_df), len(self.sat_df), len(self.main_df))
        self._build_profiles()
        self.train_test_accuracy = self._train_stress_classifier()

    # ── NDVI-deviation feature engineering ──────────────────────

    def _engineer_features(self):
        rows = []
        for _, r in self.gs_df.iterrows():
            stage = r["Growth_Stage"]
            ndvi = r["NDVI"]
            median = stage_ndvi_median(stage)
            rows.append({
                "ndvi": ndvi,
                "lai": r["Leaf_Area_Index"],
                "das": r["Days_After_Sowing"],
                "height": r["Plant_Height_cm"],
                "tillers": r["Tiller_Count"],
                "ndvi_deviation": ndvi - median,  # deviation from expected healthy NDVI at this stage
                "stage_idx": stage_index(stage),
                "label": _label_from_ndvi(ndvi, stage),
            })
        return pd.DataFrame(rows)

    def _train_stress_classifier(self):
        df = self._engineer_features()
        X = df[FEATURE_COLUMNS].values
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        self._clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42
        )
        self._clf.fit(X_train, y_train)

        y_pred = self._clf.predict(X_test)
        train_acc = accuracy_score(y_train, self._clf.predict(X_train))
        test_acc = accuracy_score(y_test, y_pred)
        log.info("  Stress classifier — train acc: %.3f  held-out test acc: %.3f", train_acc, test_acc)
        log.info("\n%s", classification_report(y_test, y_pred, labels=STRESS_LEVELS, zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=STRESS_LEVELS)
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in STRESS_LEVELS],
                              columns=[f"pred_{l}" for l in STRESS_LEVELS])
        log.info("  Confusion matrix (rows=true, cols=predicted):\n%s", cm_df.to_string())

        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "confusion_matrix": cm_df,
            "classification_report": classification_report(
                y_test, y_pred, labels=STRESS_LEVELS, zero_division=0, output_dict=True),
        }

    # ── Profiles ─────────────────────────────────────────────────

    def _build_profiles(self):
        self._profiles = {}
        grp = self.gs_df.groupby(["District", "Season", "Growth_Stage"]).agg({
            "NDVI": ["mean", "std"], "Leaf_Area_Index": ["mean"], "Plant_Height_cm": ["mean"],
            "Tiller_Count": ["mean"], "Days_After_Sowing": ["mean", "min", "max"],
        }).round(4)
        for (dist, season, stage), row in grp.iterrows():
            self._profiles[(dist, season, stage)] = {
                "ndvi_mean": row[("NDVI", "mean")], "ndvi_std": row[("NDVI", "std")],
                "lai_mean": row[("Leaf_Area_Index", "mean")],
                "height": row[("Plant_Height_cm", "mean")],
                "tillers": row[("Tiller_Count", "mean")],
                "das_mean": row[("Days_After_Sowing", "mean")],
                "das_min": row[("Days_After_Sowing", "min")],
                "das_max": row[("Days_After_Sowing", "max")],
            }
        log.info("  Built %d district x season x stage profiles", len(self._profiles))

    # ── Query methods ────────────────────────────────────────────

    def get_stage_profile(self, district, season, stage):
        """Observed profile for one district x season x stage.

        Falls back to the season-wide average for that stage, so a district with
        no record still gets a real observation rather than the global median.
        """
        key = (district, season, stage)
        if key in self._profiles:
            return self._profiles[key]
        peers = [v for (_, s, st), v in self._profiles.items() if s == season and st == stage]
        if not peers:
            peers = [v for (_, _, st), v in self._profiles.items() if st == stage]
        if not peers:
            return None
        return {k: float(np.mean([p[k] for p in peers])) for k in peers[0]}

    def predict_stress(self, ndvi, lai, das, height, tillers, stage):
        median = stage_ndvi_median(stage)
        X = np.array([[ndvi, lai, das, height, tillers, ndvi - median, stage_index(stage)]])
        return self._clf.predict(X)[0]

    def get_satellite(self, district, year, month):
        sub = self.sat_df[(self.sat_df["District"] == district) &
                           (self.sat_df["Year"] == year) & (self.sat_df["Month"] == month)]
        if sub.empty:
            sub = self.sat_df[(self.sat_df["District"] == district) & (self.sat_df["Month"] == month)]
        if sub.empty:
            return None
        r = sub.iloc[-1]
        return {"ndvi_mean": float(r["NDVI_Mean"]), "ndvi_std": float(r["NDVI_StdDev"]),
                "evi_mean": float(r["EVI_Mean"]), "lst_c": float(r["Land_Surface_Temp_C"]),
                "soil_moisture": float(r["Soil_Moisture_Percent"])}

    def get_closest_growth_record(self, district, season, das):
        sub = self.gs_df[(self.gs_df["District"] == district) & (self.gs_df["Season"] == season)]
        if sub.empty:
            sub = self.gs_df[self.gs_df["Season"] == season]
        if sub.empty:
            return None
        sub = sub.copy()
        sub["_d"] = (sub["Days_After_Sowing"] - das).abs()
        r = sub.nsmallest(1, "_d").iloc[0]
        return {"growth_stage": r["Growth_Stage"], "days_after_sowing": int(r["Days_After_Sowing"]),
                "plant_height_cm": float(r["Plant_Height_cm"]), "tiller_count": int(r["Tiller_Count"]),
                "leaf_area_index": float(r["Leaf_Area_Index"]), "ndvi": float(r["NDVI"])}

    def get_pest_risk(self, district, season):
        sub = self.main_df[(self.main_df["District"] == district) & (self.main_df["Season"] == season)]
        if sub.empty:
            sub = self.main_df[self.main_df["Season"] == season]
        if sub.empty:
            return {"pest": "Medium", "disease": "Medium", "sprays": 4.0}
        return {"pest": sub["Pest_Incidence"].mode()[0],
                "disease": sub["Disease_Incidence"].mode()[0],
                "sprays": float(sub["Pesticide_Sprays"].mean())}
