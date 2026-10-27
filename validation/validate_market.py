"""
Validation of the Market Price Agent (SARIMA(1,1,1)(1,1,1,12) on monthly mean prices).
Run from the repo root:   python3 validation/validate_market.py --repo . --out validation/results
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as tdist, linregress
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=".")
ap.add_argument("--out", default="validation_results")
ap.add_argument("--test-months", type=int, default=24)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

d = pd.read_csv(os.path.join(a.repo, "data/Karnataka_Market_Prices.csv"), parse_dates=["Date"])
VARS = sorted(d["Variety"].unique())
TEST, HMAX, STORAGE = a.test_months, 3, 30.0          # Rs/quintal/month storage cost used by the Income Agent
monthly = {v: d[d.Variety == v].set_index("Date")["Price_Rs_per_Quintal"].resample("MS").mean().dropna() for v in VARS}

# ---------------- rolling-origin backtest ----------------
rows, fails = [], {"sarima": 0, "ets": 0}
for v in VARS:
    s = monthly[v]; n = len(s)
    for t in range(n - TEST, n):
        train = s.iloc[:t]
        try:
            fc = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False).get_forecast(HMAX)
            sm, c80, c95 = fc.predicted_mean.values, fc.conf_int(alpha=0.2).values, fc.conf_int(alpha=0.05).values
        except Exception:
            fails["sarima"] += 1; sm = np.repeat(train.iloc[-1], HMAX); c80 = c95 = np.full((HMAX, 2), np.nan)
        try:
            em = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit().forecast(HMAX).values
        except Exception:
            fails["ets"] += 1; em = np.repeat(train.iloc[-1], HMAX)
        for h in range(1, HMAX + 1):
            if t + h - 1 >= n: continue
            rows.append(dict(v=v, t=t, h=h, actual=s.iloc[t + h - 1], prev=s.iloc[t - 1],
                             sarima=sm[h - 1], ets=em[h - 1], naive=train.iloc[-1], snaive=s.iloc[t + h - 13],
                             lo80=c80[h - 1, 0], hi80=c80[h - 1, 1], lo95=c95[h - 1, 0], hi95=c95[h - 1, 1]))
R = pd.DataFrame(rows)
MODELS = ["sarima", "ets", "snaive", "naive"]
scale = {v: float(np.mean(np.abs(monthly[v].iloc[12:len(monthly[v]) - TEST].values - monthly[v].iloc[:len(monthly[v]) - TEST - 12].values))) for v in VARS}
R["scale"] = R["v"].map(scale)

err = {}
for h in range(1, HMAX + 1):
    r = R[R.h == h]; err[h] = {}
    for m in MODELS:
        e = r.actual - r[m]
        err[h][m] = dict(MAPE_pct=round(float(np.mean(np.abs(e) / r.actual) * 100), 2), MAE=round(float(np.mean(np.abs(e))), 1),
                         RMSE=round(float(np.sqrt(np.mean(e ** 2))), 1), MASE=round(float(np.mean(np.abs(e) / r.scale)), 3), n=int(len(r)))
print("== Forecast error by horizon")
for h in err:
    print(f" h={h}")
    for m in MODELS: print("   ", f"{m:7s}", err[h][m])

# direction accuracy (h=1): did the model call up/down vs last month's actual price correctly?
r1 = R[R.h == 1]; act_up = (r1.actual > r1.prev).values
direction = {m: round(float(np.mean(((r1[m] > r1.prev).values) == act_up)), 3) for m in ["sarima", "ets", "snaive"]}
direction["share_of_months_price_rose"] = round(float(act_up.mean()), 3)
print("\nDirection accuracy h=1:", direction)

per_var = {v: {m: round(float(np.mean(np.abs(r1[r1.v == v].actual - r1[r1.v == v][m]) / r1[r1.v == v].actual) * 100), 2) for m in ["sarima", "ets", "snaive", "naive"]} for v in VARS}
print("\nPer-variety MAPE % (h=1):", json.dumps(per_var, indent=1))

# ---------------- Diebold-Mariano (absolute-error loss, HLN small-sample correction) ----------------
def dm(e1, e2, h=1):
    dd = np.abs(e1) - np.abs(e2); n = len(dd); var = np.var(dd, ddof=0)
    for k in range(1, h):
        var += 2 * np.cov(dd[k:], dd[:-k], ddof=0)[0, 1]
    stat = dd.mean() / np.sqrt(var / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * (1 - tdist.cdf(abs(stat), n - 1)))
dmres = {}
for h in range(1, HMAX + 1):
    r = R[R.h == h]; dmres[h] = {}
    for rival in ["snaive", "naive", "ets"]:
        s_, p_ = dm((r.actual - r.sarima).values, (r.actual - r[rival]).values, h)
        dmres[h][f"sarima_vs_{rival}"] = dict(DM_stat=round(s_, 2), p_value=round(p_, 4), sarima_better=bool(s_ < 0), significant_5pct=bool(p_ < 0.05))
print("\nDiebold-Mariano (negative stat = SARIMA has lower error):", json.dumps(dmres, indent=1))

# ---------------- interval calibration ----------------
cover = {}
for h in range(1, HMAX + 1):
    r = R[R.h == h].dropna(subset=["lo80"])
    cover[h] = {"80": dict(empirical=round(float(((r.actual >= r.lo80) & (r.actual <= r.hi80)).mean()), 3), mean_width=round(float((r.hi80 - r.lo80).mean()), 0)),
                "95": dict(empirical=round(float(((r.actual >= r.lo95) & (r.actual <= r.hi95)).mean()), 3), mean_width=round(float((r.hi95 - r.lo95).mean()), 0))}
print("\nInterval coverage (nominal 0.80 / 0.95):", json.dumps(cover, indent=1))

# ---------------- residual diagnostics (full-data fit) ----------------
diag = {}
for v in VARS:
    res = SARIMAX(monthly[v], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    rs = res.resid.iloc[13:]
    lb = acorr_ljungbox(rs, lags=[12, 24], return_df=True)
    diag[v] = dict(AIC=round(float(res.aic), 1), ljung_box_p_lag12=round(float(lb["lb_pvalue"].iloc[0]), 3), ljung_box_p_lag24=round(float(lb["lb_pvalue"].iloc[1]), 3),
                   residual_mean=round(float(rs.mean()), 1), residual_std=round(float(rs.std()), 1))
print("\nResidual diagnostics (Ljung-Box p>0.05 = no leftover autocorrelation):", json.dumps(diag, indent=1))

# ---------------- decision backtest: hold 1 month vs sell now ----------------
def decision(forecast_col, margin):
    gain = np.where((r1[forecast_col] - STORAGE - r1.prev) > margin, r1.actual - STORAGE - r1.prev, 0.0)
    hold = ((r1[forecast_col] - STORAGE - r1.prev) > margin).values
    return gain, hold
dec = {}
for name, col, mg in [("SARIMA rule, margin Rs0", "sarima", 0), ("SARIMA rule, margin Rs50", "sarima", 50), ("Seasonal-naive rule, margin Rs0", "snaive", 0)]:
    g, hld = decision(col, mg)
    dec[name] = dict(mean_gain_Rs_per_q=round(float(g.mean()), 1), hold_calls=int(hld.sum()), of=int(len(hld)),
                     hit_rate_when_holding=round(float((g[hld] > 0).mean()), 3) if hld.sum() else None,
                     mean_gain_when_holding=round(float(g[hld].mean()), 1) if hld.sum() else None)
dec["Always sell now"] = dict(mean_gain_Rs_per_q=0.0)
dec["Always hold"] = dict(mean_gain_Rs_per_q=round(float((r1.actual - STORAGE - r1.prev).mean()), 1))
dec["Perfect foresight (upper bound)"] = dict(mean_gain_Rs_per_q=round(float(np.maximum(0, r1.actual - STORAGE - r1.prev).mean()), 1))
print("\nDecision backtest (gain vs selling now, Rs/quintal, storage Rs30):", json.dumps(dec, indent=1))

# ---------------- the agent's rule-based trend label ----------------
def label(x):
    sl = linregress(range(len(x)), x)[0]
    return "rising" if sl > 2 else ("falling" if sl < -2 else "stable")
hit = tot = 0
for v in VARS:
    x = d[d.Variety == v].sort_values("Date")["Price_Rs_per_Quintal"].values
    for i in range(30, len(x) - 4, 2):
        l = label(x[i - 30:i])
        if l == "stable": continue
        up = x[i:i + 4].mean() > x[i - 4:i].mean()
        hit += (l == "rising") == up; tot += 1
trend = dict(label_accuracy_next4obs=round(hit / tot, 3), n=tot, note="overlapping windows; below 0.5 means worse than a coin flip")
print("\nTrend label vs next 4 observations:", trend)

json.dump(dict(forecast_error=err, direction=direction, per_variety_MAPE=per_var, diebold_mariano=dmres, interval_coverage=cover,
               residual_diagnostics=diag, decision_backtest=dec, trend_label=trend, failed_fits=fails),
          open(os.path.join(a.out, "market_validation.json"), "w"), indent=1)

# ---------------- figure ----------------
fig, ax = plt.subplots(2, 2, figsize=(14, 10), dpi=140)
cols = {"sarima": "#1f77b4", "ets": "#2ca02c", "snaive": "#ff7f0e", "naive": "#7f7f7f"}
a0 = ax[0, 0]; w = 0.2
for i, m in enumerate(MODELS): a0.bar(np.arange(HMAX) + i * w, [err[h][m]["MAPE_pct"] for h in range(1, HMAX + 1)], w, label=m, color=cols[m])
a0.set_xticks(np.arange(HMAX) + 1.5 * w); a0.set_xticklabels([f"{h}-month ahead" for h in range(1, HMAX + 1)]); a0.set_ylabel("MAPE %"); a0.set_title("Forecast error by horizon"); a0.legend()
a1 = ax[0, 1]
for i, m in enumerate(["sarima", "ets", "snaive"]): a1.bar(np.arange(len(VARS)) + i * 0.27, [per_var[v][m] for v in VARS], 0.27, label=m, color=cols[m])
a1.set_xticks(np.arange(len(VARS)) + 0.27); a1.set_xticklabels(VARS, fontsize=9); a1.set_ylabel("MAPE % (1-month)"); a1.set_title("Per-variety error"); a1.legend()
a2 = ax[1, 0]
for lvl, c in [("80", "#1f77b4"), ("95", "#d62728")]:
    a2.plot(range(1, HMAX + 1), [cover[h][lvl]["empirical"] for h in range(1, HMAX + 1)], "o-", color=c, label=f"{lvl}% interval"); a2.axhline(int(lvl) / 100, color=c, ls=":")
a2.set_ylim(0.5, 1.02); a2.set_xticks(range(1, HMAX + 1)); a2.set_xlabel("horizon (months)"); a2.set_ylabel("empirical coverage"); a2.set_title("Interval calibration (dotted = nominal)"); a2.legend()
a3 = ax[1, 1]; names = list(dec); vals = [dec[k]["mean_gain_Rs_per_q"] for k in names]
a3.barh(names[::-1], vals[::-1], color=["#9E9E9E" if "Perfect" in k else "#1f77b4" if "SARIMA" in k else "#BDBDBD" for k in names[::-1]])
a3.axvline(0, color="k", lw=0.8); a3.set_xlabel("mean gain vs selling now (Rs / quintal)"); a3.set_title("Would 'hold 1 month' have paid?")
plt.tight_layout(); plt.savefig(os.path.join(a.out, "market_validation.png"), bbox_inches="tight")
print("\nSaved:", os.path.join(a.out, "market_validation.json"), "and market_validation.png")
