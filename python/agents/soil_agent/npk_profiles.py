import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../data/Karnataka_Paddy_Main_Dataset.csv")


def build_npk_profiles() -> dict:
    df = pd.read_csv(DATA_PATH)

    profiles = (
        df.groupby("District")[[
            "Soil_Nitrogen_kg_per_ha",
            "Soil_Phosphorus_kg_per_ha",
            "Soil_Potassium_kg_per_ha"
        ]]
        .mean()
        .round(2)
        .to_dict(orient="index")
    )

    clean = {}
    for district, vals in profiles.items():
        clean[district] = {
            "N": vals["Soil_Nitrogen_kg_per_ha"],
            "P": vals["Soil_Phosphorus_kg_per_ha"],
            "K": vals["Soil_Potassium_kg_per_ha"],
        }
    return clean


def get_npk_for_district(district: str) -> dict:
    profiles = build_npk_profiles()
    district = district.strip().title()
    if district not in profiles:
        available = list(profiles.keys())[:5]
        raise ValueError(f"District '{district}' not found. Sample available: {available}")
    return profiles[district]
def build_district_agro_profiles() -> dict:
    df = pd.read_csv(DATA_PATH)
    numeric = (df.groupby("District")[["Rainfall_mm", "Irrigation_Percent", "Temperature_Celsius"]]
               .mean().round(1).to_dict(orient="index"))
    region = df.groupby("District")["Region"].agg(lambda s: s.mode()[0]).to_dict()
    quality = df.groupby("District")["Soil_Quality"].agg(lambda s: s.mode()[0]).to_dict()
    profiles = {}
    for district, vals in numeric.items():
        profiles[district] = {
            "rainfall": vals["Rainfall_mm"], "irrigation": vals["Irrigation_Percent"],
            "temperature": vals["Temperature_Celsius"], "region": region[district],
            "soil_quality": quality[district],
        }
    return profiles

def get_agro_profile_for_district(district: str) -> dict:
    profiles = build_district_agro_profiles()
    district = district.strip().title()
    if district not in profiles:
        raise ValueError(f"District '{district}' not found.")
    return profiles[district]