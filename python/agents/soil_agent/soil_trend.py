import pandas as pd
from scipy import stats
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../data/Karnataka_Paddy_Main_Dataset.csv")


def get_soil_trend(district: str) -> dict:
    df = pd.read_csv(DATA_PATH)

    district = district.strip().title()
    df_dist  = df[df["District"] == district]

    if df_dist.empty:
        raise ValueError(f"No data found for district: {district}")

    yearly = (
        df_dist.groupby("Numeric_Year")[["Soil_pH", "Soil_Organic_Carbon_Percent"]]
        .mean()
        .reset_index()
    )

    def regress(years, values):
        if len(years) < 2:
            return 0.0, round(float(values.iloc[-1]), 3)
        slope, _, _, _, _ = stats.linregress(years, values)
        return round(slope, 4), round(float(values.iloc[-1]), 3)

    ph_slope, ph_last = regress(yearly["Numeric_Year"], yearly["Soil_pH"])
    oc_slope, oc_last = regress(yearly["Numeric_Year"], yearly["Soil_Organic_Carbon_Percent"])

    ph_msg = "stable"
    if ph_slope >  0.01: ph_msg = "increasing (becoming more alkaline)"
    if ph_slope < -0.01: ph_msg = "decreasing (becoming more acidic)"

    oc_msg = "stable"
    if oc_slope >  0.001: oc_msg = "improving (organic matter rising)"
    if oc_slope < -0.001: oc_msg = "declining (consider green manure)"

    interpretation = (
        f"{district}: pH is {ph_msg} (slope={ph_slope}/yr, last={ph_last}). "
        f"Organic carbon is {oc_msg} (slope={oc_slope}/yr, last={oc_last}%)."
    )

    return {
        "district":      district,
        "ph_slope":      ph_slope,
        "ph_last_value": ph_last,
        "oc_slope":      oc_slope,
        "oc_last_value": oc_last,
        "interpretation": interpretation,
    }