"""
Validation of the Soil Agent yield model (Random Forest, 12 features).
Run from the repo root:   python3 validation/validate_soil.py --repo . --out validation/results
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=".")
ap.add_argument("--out", default="validation_results")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

df = pd.read_csv(os.path.join(a.repo, "data/Karnataka_Paddy_Main_Dataset.csv"))
for raw, enc in [("District", "district_enc"), ("Season", "season_enc"), ("Region", "region_enc"), ("Soil_Quality", "soil_quality_enc")]:
    df[enc] = LabelEncoder().fit_transform(df[raw])
NUM = ["Soil_Nitrogen_kg_per_ha", "Soil_Phosphorus_kg_per_ha", "Soil_Potassium_kg_per_ha", "Soil_pH",
       "Soil_Organic_Carbon_Percent", "Rainfall_mm", "Irrigation_Percent", "Temperature_Celsius"]
FEATS = NUM + ["district_enc", "season_enc", "region_enc", "soil_quality_enc"]   # the 12 features used by the agent
y = df["Yield_Kg_per_Ha"].values
RS = 42

# ---------------- models ----------------
def rf():  return RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=RS, n_jobs=-1)
def gb():  return GradientBoostingRegressor(random_state=RS)
def ridge():
    ct = ColumnTransformer([("n", StandardScaler(), NUM), ("c", OneHotEncoder(handle_unknown="ignore"), ["Season", "Region", "Soil_Quality"])])
    return make_pipeline(ct, RidgeCV(alphas=np.logspace(-2, 3, 12)))

def fit_predict(name, tr, te):
    if name == "Global mean":
        return np.full(len(te), y[tr].mean())
    if name == "District mean":
        m = df.iloc[tr].groupby("District")["Yield_Kg_per_Ha"].mean()
        return df.iloc[te]["District"].map(m).fillna(y[tr].mean()).values
    if name == "Ridge (linear)":
        return ridge().fit(df.iloc[tr], y[tr]).predict(df.iloc[te])
    if name == "Gradient Boosting":
        return gb().fit(df.iloc[tr][FEATS], y[tr]).predict(df.iloc[te][FEATS])
    if name == "Random Forest":
        return rf().fit(df.iloc[tr][FEATS], y[tr]).predict(df.iloc[te][FEATS])

MODELS = ["Global mean", "District mean", "Ridge (linear)", "Gradient Boosting", "Random Forest"]

# ---------------- protocols ----------------
def splits(proto):
    idx = np.arange(len(df))
    if proto == "random 5-fold":        return list(KFold(5, shuffle=True, random_state=RS).split(idx))
    if proto == "leave-years-out":      return list(GroupKFold(5).split(idx, y, df["Numeric_Year"]))
    if proto == "leave-districts-out":  return list(GroupKFold(5).split(idx, y, df["District"]))
    if proto == "forward-chaining":
        out = []
        for ty in [2021, 2022, 2023, 2024]:
            out.append((idx[df["Numeric_Year"] < ty], idx[df["Numeric_Year"] == ty]))
        return out
PROTOS = ["random 5-fold", "leave-years-out", "leave-districts-out", "forward-chaining"]

def metrics(yt, yp, fold_r2):
    return dict(R2=round(r2_score(yt, yp), 3), MAE=round(mean_absolute_error(yt, yp), 1),
                RMSE=round(float(np.sqrt(mean_squared_error(yt, yp))), 1),
                MAPE_pct=round(float(np.mean(np.abs(yt - yp) / yt) * 100), 2),
                fold_R2_mean=round(float(np.mean(fold_r2)), 3), fold_R2_min=round(float(np.min(fold_r2)), 3))

results, preds = {}, {}
for p in PROTOS:
    sp = splits(p)
    results[p] = {}
    for m in MODELS:
        pr = np.full(len(df), np.nan); fr2 = []
        for tr, te in sp:
            pr[te] = fit_predict(m, tr, te)
            fr2.append(r2_score(y[te], pr[te]))
        mask = ~np.isnan(pr)
        results[p][m] = metrics(y[mask], pr[mask], fr2)
        results[p][m]["n_test"] = int(mask.sum())
        preds[(p, m)] = pr
    print(f"\n== {p}")
    for m in MODELS: print(f"  {m:18s}", results[p][m])

# ---------------- paired significance: RF vs District mean ----------------
sig = {}
for p in PROTOS:
    pr_rf, pr_b = preds[(p, "Random Forest")], preds[(p, "District mean")]
    mk = ~np.isnan(pr_rf)
    e_rf, e_b = np.abs(y[mk] - pr_rf[mk]), np.abs(y[mk] - pr_b[mk])
    w = wilcoxon(e_rf, e_b)
    sig[p] = dict(mean_abs_err_RF=round(float(e_rf.mean()), 1), mean_abs_err_district_mean=round(float(e_b.mean()), 1),
                  wilcoxon_p=float(w.pvalue), RF_better=bool(e_rf.mean() < e_b.mean()))
print("\nPaired test RF vs District-mean:", json.dumps(sig, indent=1))

# ---------------- residuals (leave-years-out RF) ----------------
pr = preds[("leave-years-out", "Random Forest")]; res = y - pr
rd = pd.DataFrame({"District": df["District"], "Season": df["Season"], "abs": np.abs(res), "res": res})
resid = dict(bias_kg_ha=round(float(res.mean()), 1),
             MAE_by_season=rd.groupby("Season")["abs"].mean().round(1).to_dict(),
             worst_districts=rd.groupby("District")["abs"].mean().sort_values(ascending=False).head(5).round(1).to_dict(),
             best_districts=rd.groupby("District")["abs"].mean().sort_values().head(3).round(1).to_dict())
print("\nResiduals:", json.dumps(resid, indent=1))

# ---------------- interval calibration (OOB residuals, leave-years-out) ----------------
cover = {0.8: [], 0.9: []}; width = {0.8: [], 0.9: []}
for tr, te in splits("leave-years-out"):
    m = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=5, min_samples_leaf=2, oob_score=True, random_state=RS, n_jobs=-1)
    m.fit(df.iloc[tr][FEATS], y[tr])
    oob = np.abs(y[tr] - m.oob_prediction_)
    p_te = m.predict(df.iloc[te][FEATS])
    for lvl in cover:
        q = np.quantile(oob, lvl)
        cover[lvl] += list(np.abs(y[te] - p_te) <= q); width[lvl].append(2 * q)
interval = {str(l): dict(nominal=l, empirical_coverage=round(float(np.mean(c)), 3), mean_width_kg_ha=round(float(np.mean(width[l])), 0)) for l, c in cover.items()}
print("\nPrediction intervals (leave-years-out):", json.dumps(interval, indent=1))

# ---------------- permutation importance (held-out, leave-years-out) ----------------
imp = np.zeros(len(FEATS)); k = 0
for tr, te in splits("leave-years-out"):
    m = rf().fit(df.iloc[tr][FEATS], y[tr])
    pi = permutation_importance(m, df.iloc[te][FEATS], y[te], n_repeats=5, random_state=RS, n_jobs=-1)
    imp += pi.importances_mean; k += 1
imp /= k
importance = dict(sorted({f: round(float(v), 4) for f, v in zip(FEATS, imp)}.items(), key=lambda kv: -kv[1]))
print("\nPermutation importance (drop in R2):", json.dumps(importance, indent=1))

json.dump(dict(protocols=results, rf_vs_district_mean=sig, residuals=resid, intervals=interval, permutation_importance=importance),
          open(os.path.join(a.out, "soil_validation.json"), "w"), indent=1)

# ---------------- figure ----------------
fig, ax = plt.subplots(2, 2, figsize=(14, 10), dpi=140)
a0 = ax[0, 0]; w = 0.15
for i, m in enumerate(MODELS):
    a0.bar(np.arange(len(PROTOS)) + i * w, [results[p][m]["R2"] for p in PROTOS], w, label=m)
a0.set_xticks(np.arange(len(PROTOS)) + 2 * w); a0.set_xticklabels(PROTOS, fontsize=9); a0.set_ylabel("R\u00b2"); a0.set_ylim(-0.1, 1.25)
a0.set_title("R\u00b2 by validation protocol and model"); a0.legend(fontsize=8, ncol=3, loc="upper center")
a1 = ax[0, 1]
for p, c in [("leave-years-out", "#1f77b4"), ("leave-districts-out", "#d62728")]:
    pp = preds[(p, "Random Forest")]; a1.scatter(y, pp, s=7, alpha=0.4, c=c, label=p)
lim = [y.min(), y.max()]; a1.plot(lim, lim, "k--", lw=1); a1.set_xlabel("Actual yield (kg/ha)"); a1.set_ylabel("Predicted"); a1.legend(); a1.set_title("Random Forest: predicted vs actual")
a2 = ax[1, 0]; a2.barh(list(importance)[::-1], list(importance.values())[::-1], color="#2E7D32"); a2.set_title("Permutation importance (held-out years)"); a2.set_xlabel("drop in R\u00b2 when shuffled")
a3 = ax[1, 1]; dm = rd.groupby("District")["abs"].mean().sort_values(); a3.barh(dm.index, dm.values, color="#F57C00"); a3.tick_params(axis="y", labelsize=6)
a3.set_title("Mean absolute error by district (leave-years-out)"); a3.set_xlabel("kg/ha")
plt.tight_layout(); plt.savefig(os.path.join(a.out, "soil_validation.png"), bbox_inches="tight")
print("\nSaved:", os.path.join(a.out, "soil_validation.json"), "and soil_validation.png")
