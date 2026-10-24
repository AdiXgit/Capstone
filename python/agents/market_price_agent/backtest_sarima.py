"""
SARIMA backtest — how accurate are the Market Agent's price forecasts?

Method (rolling-origin / walk-forward):
  For each of the last N months:
    1. Fit SARIMA using ONLY the months before it (no peeking at the future)
    2. Forecast that month
    3. Compare with the actual price
  Then compare SARIMA against two simple baselines. If SARIMA can't beat them,
  it isn't adding value.
    - Naive:           "next month = this month"
    - Seasonal naive:  "next month = same month last year"

Run from the market_price_agent folder:
    python backtest_sarima.py
"""

import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../data/Karnataka_Market_Prices.csv")
ORDER = (1, 1, 1)                 # same as the Market Agent
SEASONAL_ORDER = (1, 1, 1, 12)
TEST_MONTHS = 24                  # backtest over the last 2 years


def monthly_series(df, variety):
    s = (df[df["Variety"] == variety]
         .set_index("Date")["Price_Rs_per_Quintal"]
         .resample("MS").mean()
         .interpolate())
    return s


def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def mae(actual, pred):
    return float(np.mean(np.abs(np.array(actual) - np.array(pred))))


def direction_accuracy(prev, actual, pred):
    """% of months where the forecast got up/down right — what the hold/sell advice depends on."""
    prev, actual, pred = map(np.array, (prev, actual, pred))
    return float(np.mean(np.sign(actual - prev) == np.sign(pred - prev)) * 100)


def backtest_variety(series, test_months=TEST_MONTHS):
    rows = []
    start = len(series) - test_months
    for i in range(start, len(series)):
        history = series.iloc[:i]
        actual = series.iloc[i]
        try:
            fit = SARIMAX(history, order=ORDER, seasonal_order=SEASONAL_ORDER,
                          enforce_stationarity=False, enforce_invertibility=False
                          ).fit(disp=False)
            sarima = float(fit.forecast(1).iloc[0])
        except Exception:
            sarima = float(history.iloc[-1])      # fall back to naive if a fit fails
        rows.append({
            "month": series.index[i],
            "prev": float(history.iloc[-1]),
            "actual": float(actual),
            "sarima": sarima,
            "naive": float(history.iloc[-1]),
            "seasonal_naive": float(history.iloc[-12]),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    varieties = sorted(df["Variety"].unique())
    print(f"Backtesting SARIMA{ORDER}x{SEASONAL_ORDER} over the last {TEST_MONTHS} months "
          f"for {len(varieties)} varieties...\n")

    summary, all_rows = [], []
    for v in varieties:
        s = monthly_series(df, v)
        r = backtest_variety(s)
        r["variety"] = v
        all_rows.append(r)
        summary.append({
            "Variety": v,
            "SARIMA MAPE %": mape(r.actual, r.sarima),
            "Naive MAPE %": mape(r.actual, r.naive),
            "Seasonal MAPE %": mape(r.actual, r.seasonal_naive),
            "SARIMA MAE Rs": mae(r.actual, r.sarima),
            "Direction %": direction_accuracy(r.prev, r.actual, r.sarima),
        })
        print(f"  {v:<10} done")

    res = pd.DataFrame(summary)
    all_df = pd.concat(all_rows)
    overall = {
        "Variety": "OVERALL",
        "SARIMA MAPE %": mape(all_df.actual, all_df.sarima),
        "Naive MAPE %": mape(all_df.actual, all_df.naive),
        "Seasonal MAPE %": mape(all_df.actual, all_df.seasonal_naive),
        "SARIMA MAE Rs": mae(all_df.actual, all_df.sarima),
        "Direction %": direction_accuracy(all_df.prev, all_df.actual, all_df.sarima),
    }
    res = pd.concat([res, pd.DataFrame([overall])], ignore_index=True)

    print("\n" + "=" * 78)
    print("RESULTS  (lower MAPE/MAE = better; Direction = % of months up/down called right)")
    print("=" * 78)
    print(res.round(2).to_string(index=False))

    best_baseline = min(overall["Naive MAPE %"], overall["Seasonal MAPE %"])
    gain = best_baseline - overall["SARIMA MAPE %"]
    print("\nVERDICT")
    if gain > 0:
        print(f"  SARIMA beats the best simple baseline by {gain:.2f} percentage points of MAPE.")
    else:
        print(f"  SARIMA does NOT beat the best simple baseline (worse by {-gain:.2f} points).")
        print("  -> Report this honestly; consider the simpler model or tuning the SARIMA orders.")
    print(f"  Typical forecast error: about {overall['SARIMA MAPE %']:.1f}% "
          f"(~Rs {overall['SARIMA MAE Rs']:.0f}/quintal).")
    print(f"  Up/down direction correct {overall['Direction %']:.0f}% of the time "
          f"(50% = coin flip) - this is what the hold/sell advice relies on.")

    out = os.path.join(os.path.dirname(__file__) or ".", "sarima_backtest_results.csv")
    all_df.to_csv(out, index=False)
    print(f"\nMonth-by-month forecasts saved to {os.path.basename(out)}")


if __name__ == "__main__":
    main()
    