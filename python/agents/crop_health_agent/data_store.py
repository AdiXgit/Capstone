"""Crop-health data store.

REVISION (leakage fix): the stress level was previously "predicted" by a
GradientBoosting model that took NDVI (and NDVI-deviation) as inputs — but the
label itself is defined by NDVI vs the stage benchmark, so the task was circular
and reported a meaningless ~100% accuracy. We verified this honestly:

    features WITHOUT ndvi  → 5-fold CV accuracy 0.36  (== the 0.36 majority baseline)
    features WITH  ndvi_dev → 5-fold CV accuracy 0.97  (pure leakage; label IS the rule)

i.e. the agronomic measurements carry no independent signal for this label and
NDVI trivially reconstructs it. So the honest design is a **transparent,
deterministic NDVI-vs-benchmark rule** — interpretable and correct — rather than
a black box that launders the same rule as "machine learning". The real learned
model in this project is the YOLOv8 leaf-image classifier (see Model Performance).
"""
import logging

import numpy as np
import pandas as pd

from stage_profiles import STAGE_ORDER, STAGE_PROFILES, stage_index, stage_ndvi_median

log = logging.getLogger("CROP_HEALTH.data_store")

STRESS_LEVELS = ["Healthy", "Mild", "Moderate", "Severe"]


def _label_from_ndvi(ndvi, stage):
    """4-level stress label from NDVI vs the stage's data-derived quartile baseline.

    This is the definition of "stress" in this system — a transparent agronomic
    rule, not a learned target.
    """
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
    """Loads the Karnataka paddy datasets and applies the NDVI-benchmark stress rule."""

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
        self.stress_model = self._rule_report()

    # ── stress rule reporting (honest; no learned-accuracy claim) ──

    def _rule_report(self):
        """Describe the deterministic rule and how the dataset labels distribute."""
        labels = [_label_from_ndvi(r["NDVI"], r["Growth_Stage"]) for _, r in self.gs_df.iterrows()]
        dist = {lvl: int(pd.Series(labels).eq(lvl).sum()) for lvl in STRESS_LEVELS}
        log.info("  Stress = deterministic NDVI-vs-benchmark rule (no ML). "
                 "Label distribution over %d records: %s", len(labels), dist)
        return {
            "method": "deterministic_ndvi_benchmark_rule",
            "records": len(labels),
            "label_distribution": dist,
            "note": "NDVI vs stage Q25/median/Q75. Not a learned model — the real ML "
                    "model in this project is the YOLOv8 image classifier.",
        }

    # ── profiles ──────────────────────────────────────────────────

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

    # ── query methods ─────────────────────────────────────────────

    def get_stage_profile(self, district, season, stage):
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
        """Deterministic NDVI-benchmark rule. The lai/das/height/tillers args are
        kept for call-site compatibility but do not affect the label (they carry no
        independent signal for it — see the module docstring)."""
        return _label_from_ndvi(ndvi, stage)

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
