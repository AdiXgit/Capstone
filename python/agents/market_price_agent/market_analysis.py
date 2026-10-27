import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import os

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../data/Karnataka_Market_Prices.csv"
)

VALID_VARIETIES = [
    "BPT-5204", "IR-64", "JGL-1798", "Jyothi", "MTU-1010"
]

# Empirical one-month-ahead absolute-error quantiles from a
# 24-month rolling-origin backtest, pooled across all five varieties.
# These are empirical prediction bands, not SARIMAX confidence intervals.
SARIMA_BAND_RS_80 = 67.0
SARIMA_BAND_RS_90 = 84.0


def load_market_data(variety):
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    variety = variety.strip()

    if variety not in VALID_VARIETIES:
        raise ValueError(
            f"Unknown variety '{variety}'. Valid options: {VALID_VARIETIES}"
        )

    df_v = (
        df[df["Variety"] == variety]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if df_v.empty:
        raise ValueError(
            f"No market data found for variety: {variety}"
        )

    return df_v


def fit_sarima(df_v, forecast_months=3):
    """
    Fits a SARIMA(1,1,1)(1,1,1,12) model on monthly-resampled prices.

    Returns:
        forecast: Forecasted monthly prices, or None on failure.
        expected_current: In-sample estimate for the latest month, or None.
        status: "ok", "insufficient_history", or "fit_failed".
    """
    monthly = (
        df_v.set_index("Date")["Price_Rs_per_Quintal"]
        .resample("ME")
        .mean()
        .dropna()
    )

    if len(monthly) < 24:
        return None, None, "insufficient_history"

    try:
        model = SARIMAX(
            monthly,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=forecast_months)

        in_sample = fitted.get_prediction(
            start=len(monthly) - 1,
            end=len(monthly) - 1
        )

        expected_current = float(
            in_sample.predicted_mean.iloc[0]
        )

        return forecast, expected_current, "ok"

    except Exception:
        return None, None, "fit_failed"


def get_market_advisory(variety, district=None):
    df = load_market_data(variety)

    current_row = df.iloc[-1]
    current_price = float(
        current_row["Price_Rs_per_Quintal"]
    )

    current_demand = current_row["Market_Demand"]
    current_supply = current_row["Supply_Status"]

    avg_30 = float(
        df.tail(30)["Price_Rs_per_Quintal"].mean()
    )

    avg_90 = float(
        df.tail(90)["Price_Rs_per_Quintal"].mean()
    )

    recent = df.tail(30).reset_index(drop=True)

    # Short-term trend analysis using linear regression
    if len(recent) >= 2:
        slope, _, _, _, _ = linregress(
            range(len(recent)),
            recent["Price_Rs_per_Quintal"]
        )
    else:
        slope = 0

    if slope > 2:
        trend = "rising"
    elif slope < -2:
        trend = "falling"
    else:
        trend = "stable"

    # Seasonal analysis
    monthly_avg = df.groupby(
        df["Date"].dt.month
    )["Price_Rs_per_Quintal"].mean()

    best_month_num = int(monthly_avg.idxmax())

    months = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    best_month = months[best_month_num - 1]

    # Compare current price against the 90-day average
    pct_above_90 = (
        ((current_price - avg_90) / avg_90) * 100
        if avg_90 else 0
    )

    # SARIMA forecasting
    forecast, seasonal_expected, sarima_status = fit_sarima(df)

    seasonal_deviation_pct = None
    forecast_next_month = None

    if (
        sarima_status == "ok"
        and seasonal_expected is not None
        and seasonal_expected != 0
    ):
        seasonal_deviation_pct = (
            (current_price - seasonal_expected)
            / seasonal_expected
        ) * 100

    if sarima_status == "ok" and forecast is not None:
        forecast_next_month = round(
            float(forecast.iloc[0]), 2
        )

    # Empirical forecast bands
    forecast_low_80 = forecast_high_80 = None
    forecast_low_90 = forecast_high_90 = None

    if forecast_next_month is not None:
        forecast_low_80 = round(
            forecast_next_month - SARIMA_BAND_RS_80, 2
        )
        forecast_high_80 = round(
            forecast_next_month + SARIMA_BAND_RS_80, 2
        )
        forecast_low_90 = round(
            forecast_next_month - SARIMA_BAND_RS_90, 2
        )
        forecast_high_90 = round(
            forecast_next_month + SARIMA_BAND_RS_90, 2
        )

    # Build farmer-facing advisory
    advisory_parts = []

    if pct_above_90 > 5:
        advisory_parts.append(
            f"Price is {pct_above_90:.1f}% above the 90-day average."
        )
    elif pct_above_90 < -5:
        advisory_parts.append(
            f"Price is {abs(pct_above_90):.1f}% below the 90-day average."
        )

    # Seasonal price deviation
    if seasonal_deviation_pct is not None:
        if abs(seasonal_deviation_pct) > 5:
            direction = (
                "above" if seasonal_deviation_pct > 0 else "below"
            )
            advisory_parts.append(
                f"Price is {abs(seasonal_deviation_pct):.1f}% "
                f"{direction} what SARIMA expects for this time "
                f"of year (after accounting for the seasonal cycle)."
            )
        else:
            advisory_parts.append(
                "Price is in line with the expected seasonal pattern."
            )

    # Market demand and supply conditions
    if (
        current_demand == "High"
        and current_supply in ("Deficit", "Normal")
    ):
        advisory_parts.append(
            "High demand with limited supply — favourable market conditions."
        )

    elif (
        current_demand == "Low"
        and current_supply == "Surplus"
    ):
        advisory_parts.append(
            "Low demand and surplus supply — softer market conditions."
        )

    # Prefer SARIMA forecast when available
    elif forecast_next_month is not None:
        diff = forecast_next_month - current_price

        if diff > 50:
            advisory_parts.append(
                f"SARIMA forecasts the price rising to about "
                f"Rs {forecast_next_month:,.0f}/quintal next month "
                f"(likely between Rs {forecast_low_80:,.0f} and "
                f"Rs {forecast_high_80:,.0f})."
            )

        elif diff < -50:
            advisory_parts.append(
                f"SARIMA forecasts the price falling to about "
                f"Rs {forecast_next_month:,.0f}/quintal next month "
                f"(likely between Rs {forecast_low_80:,.0f} and "
                f"Rs {forecast_high_80:,.0f})."
            )

        else:
            advisory_parts.append(
                "SARIMA does not forecast a clear price move next month."
            )

    # Fall back to the linear-regression trend if no forecast exists
    elif trend == "rising":
        advisory_parts.append(
            "Short-term price trend is upward."
        )

    else:
        advisory_parts.append(
            "Short-term price trend is stable or declining."
        )

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
        "sarima_status": sarima_status,
        "sarima_seasonal_expected_price": (
            round(seasonal_expected, 2)
            if seasonal_expected is not None else None
        ),
        "sarima_seasonal_deviation_pct": (
            round(seasonal_deviation_pct, 1)
            if seasonal_deviation_pct is not None else None
        ),
        "sarima_next_month_forecast": forecast_next_month,
        "forecast_low_80": forecast_low_80,
        "forecast_high_80": forecast_high_80,
        "forecast_low_90": forecast_low_90,
        "forecast_high_90": forecast_high_90,
        "advisory": advisory,
    }