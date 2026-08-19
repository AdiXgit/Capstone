import pandas as pd
from scipy.stats import linregress
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../data/Karnataka_Market_Prices.csv")

VALID_VARIETIES = ["BPT-5204", "IR-64", "JGL-1798", "Jyothi", "MTU-1010"]


def load_market_data(variety):
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    variety = variety.strip()

    if variety not in VALID_VARIETIES:
        raise ValueError(f"Unknown variety '{variety}'. Valid options: {VALID_VARIETIES}")

    df_v = df[df["Variety"] == variety].sort_values("Date").reset_index(drop=True)
    if df_v.empty:
        raise ValueError(f"No market data found for variety: {variety}")
    return df_v


def get_market_advisory(variety, district=None):
    df = load_market_data(variety)

    current_row = df.iloc[-1]
    current_price = float(current_row["Price_Rs_per_Quintal"])
    current_demand = current_row["Market_Demand"]
    current_supply = current_row["Supply_Status"]

    avg_30 = float(df.tail(30)["Price_Rs_per_Quintal"].mean())
    avg_90 = float(df.tail(90)["Price_Rs_per_Quintal"].mean())

    recent = df.tail(30).reset_index(drop=True)
    if len(recent) >= 2:
        slope, _, _, _, _ = linregress(range(len(recent)), recent["Price_Rs_per_Quintal"])
    else:
        slope = 0

    if slope > 2:
        trend = "rising"
    elif slope < -2:
        trend = "falling"
    else:
        trend = "stable"

    monthly_avg = df.groupby(df["Date"].dt.month)["Price_Rs_per_Quintal"].mean()
    best_month_num = int(monthly_avg.idxmax())
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    best_month = months[best_month_num - 1]

    pct_above_90 = ((current_price - avg_90) / avg_90) * 100 if avg_90 else 0

    advisory_parts = []

    if pct_above_90 > 5:
        advisory_parts.append(f"Price is {pct_above_90:.1f}% above the 90-day average.")
    elif pct_above_90 < -5:
        advisory_parts.append(f"Price is {abs(pct_above_90):.1f}% below the 90-day average.")

    if current_demand == "High" and current_supply in ("Deficit", "Normal"):
        advisory_parts.append("High demand with limited supply — favourable time to sell.")
    elif current_demand == "Low" and current_supply == "Surplus":
        advisory_parts.append("Low demand and surplus supply — consider holding if storage allows.")
    elif trend == "rising":
        advisory_parts.append("Prices trending upward. Holding may yield better returns.")
    else:
        advisory_parts.append("Prices stable or declining. Selling soon avoids storage risk.")

    advisory = " ".join(advisory_parts)

    return {
        "variety": variety,
        "current_price": round(current_price, 2),
        "avg_30day": round(avg_30, 2),
        "avg_90day": round(avg_90, 2),
        "price_trend": trend,
        "market_demand": current_demand,
        "supply_status": current_supply,
        "best_month": best_month,
        "advisory": advisory,
    }