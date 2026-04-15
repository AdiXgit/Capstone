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