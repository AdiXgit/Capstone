"""Does cross-variety pretraining actually help price forecasting?

Each of the five varieties is used as the target in turn. For every target:

    naive        next price = current price (the hurdle any forecaster must clear)
    ar1          OLS on lag-1 weekly return -- captures the -0.4 mean reversion
                 in closed form, so it isolates how much of the signal is
                 simply linear
    target_only  PriceGRU trained on that variety's ~365 training weeks
    transfer     same PriceGRU pretrained on the other four varieties, then
                 fine-tuned on the target

target_only vs transfer is the comparison the whole exercise exists for: same
architecture, same target data, same early-stopping set. Only the initial
weights differ.

Usage:
    python run_transfer_experiment.py
    python run_transfer_experiment.py --seeds 5
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from price_transfer import (
    RETURN_SCALE, VARIETIES, PriceGRU, build_windows, evaluate,
    load_panel, predict_returns, split_by_date, train_model,
)

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transfer_results.json")
METRICS = ("mae", "rmse", "mape", "direction_accuracy")


def fit_ar1(train_returns_lag1, train_returns):
    """Closed-form OLS: r[t] = a + b * r[t-1]."""
    A = np.stack([np.ones_like(train_returns_lag1), train_returns_lag1], axis=1)
    coef, *_ = np.linalg.lstsq(A, train_returns, rcond=None)
    return coef


def run_target(df, target, seeds, args):
    X, y, dates, last_price = build_windows(df, target)
    tr, va, te = split_by_date(dates)

    true_te = y[te] * RETURN_SCALE
    price_te = last_price[te]

    results = {}

    # ---- naive: predict no change -------------------------------------
    results["naive"] = evaluate(np.zeros_like(true_te), true_te, price_te)

    # ---- ar1: linear mean reversion, fit on train only ----------------
    lag1_all = X[:, -1, 0] * RETURN_SCALE
    coef = fit_ar1(lag1_all[tr], y[tr] * RETURN_SCALE)
    results["ar1"] = evaluate(coef[0] + coef[1] * lag1_all[te], true_te, price_te)
    results["ar1"]["coefficients"] = {"intercept": float(coef[0]), "lag1": float(coef[1])}

    # ---- source pool for pretraining ----------------------------------
    # Only the source varieties' TRAIN-period weeks. Including their test-period
    # weeks would be look-ahead across the panel.
    src_X, src_y, val_X, val_y = [], [], [], []
    for other in VARIETIES:
        if other == target:
            continue
        oX, oy, odates, _ = build_windows(df, other)
        otr, ova, _ = split_by_date(odates)
        src_X.append(oX[otr]); src_y.append(oy[otr])
        val_X.append(oX[ova]); val_y.append(oy[ova])
    src_X, src_y = np.concatenate(src_X), np.concatenate(src_y)
    val_X, val_y = np.concatenate(val_X), np.concatenate(val_y)

    per_seed = {"target_only": [], "transfer": []}
    pretrain_epochs = []

    for seed in range(seeds):
        # ---- target_only ----------------------------------------------
        model = PriceGRU(dropout=args.dropout)
        model, _ = train_model(model, X[tr], y[tr], X[va], y[va],
                               epochs=args.epochs, lr=args.lr, patience=args.patience,
                               seed=seed)
        per_seed["target_only"].append(
            evaluate(predict_returns(model, X[te]), true_te, price_te))

        # ---- transfer: pretrain on sources, then fine-tune -------------
        model = PriceGRU(dropout=args.dropout)
        model, pre = train_model(model, src_X, src_y, val_X, val_y,
                                 epochs=args.epochs, lr=args.lr,
                                 patience=args.patience, seed=seed)
        pretrain_epochs.append(pre["epoch"])
        # Lower LR for fine-tuning so the target's 365 weeks adapt the shared
        # dynamics rather than overwrite them.
        model, _ = train_model(model, X[tr], y[tr], X[va], y[va],
                               epochs=args.epochs, lr=args.lr * args.finetune_lr_scale,
                               patience=args.patience, seed=seed)
        per_seed["transfer"].append(
            evaluate(predict_returns(model, X[te]), true_te, price_te))

    for name, runs in per_seed.items():
        results[name] = {
            m: round(float(np.mean([r[m] for r in runs])), 4) for m in METRICS
        }
        results[name]["mae_std_across_seeds"] = round(
            float(np.std([r["mae"] for r in runs])), 4)
        results[name]["n"] = runs[0]["n"]

    results["_meta"] = {
        "train_weeks": int(tr.sum()),
        "val_weeks": int(va.sum()),
        "test_weeks": int(te.sum()),
        "pretrain_samples": int(len(src_X)),
        "mean_pretrain_best_epoch": float(np.mean(pretrain_epochs)),
    }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--finetune-lr-scale", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    df = load_panel()

    # Reuse any variety a previous run already finished and flush after each
    # one, so an interrupted run does not throw away completed targets.
    all_results = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            all_results = json.load(f).get("per_variety", {})
        if all_results:
            print(f"Resuming; already have: {', '.join(all_results)}")

    for target in VARIETIES:
        if target in all_results:
            print(f"\n=== target: {target} (cached) ===")
            continue
        print(f"\n=== target: {target} ===", flush=True)
        res = run_target(df, target, args.seeds, args)
        all_results[target] = res
        with open(OUT_PATH, "w") as f:
            json.dump({"config": vars(args), "per_variety": all_results}, f, indent=2)
        meta = res["_meta"]
        print(f"  train={meta['train_weeks']}w val={meta['val_weeks']}w "
              f"test={meta['test_weeks']}w  pretrain_samples={meta['pretrain_samples']}")
        for name in ("naive", "ar1", "target_only", "transfer"):
            r = res[name]
            print(f"  {name:12s} MAE={r['mae']:7.2f}  RMSE={r['rmse']:7.2f}  "
                  f"MAPE={r['mape']:5.2f}%  dir={r['direction_accuracy']:.3f}")

    # ---- aggregate across the five targets ----------------------------
    summary = {}
    for name in ("naive", "ar1", "target_only", "transfer"):
        summary[name] = {
            m: round(float(np.mean([all_results[v][name][m] for v in VARIETIES])), 4)
            for m in METRICS
        }

    base = summary["naive"]["mae"]
    for name in summary:
        summary[name]["mae_improvement_vs_naive_pct"] = round(
            (base - summary[name]["mae"]) / base * 100, 2)
    tgt = summary["target_only"]["mae"]
    summary["transfer"]["mae_improvement_vs_target_only_pct"] = round(
        (tgt - summary["transfer"]["mae"]) / tgt * 100, 2)

    # Per-variety win/loss, so a single lucky average cannot carry the claim.
    wins = sum(all_results[v]["transfer"]["mae"] < all_results[v]["target_only"]["mae"]
               for v in VARIETIES)
    summary["transfer"]["varieties_where_transfer_wins"] = f"{wins}/{len(VARIETIES)}"

    with open(OUT_PATH, "w") as f:
        json.dump({"config": vars(args), "per_variety": all_results,
                   "summary": summary}, f, indent=2)

    print("\n" + "=" * 68)
    print("MEAN ACROSS 5 VARIETIES (test period 2023-01 .. 2024-12)")
    print("=" * 68)
    print(f"{'model':13s} {'MAE':>8s} {'RMSE':>8s} {'MAPE':>7s} {'dir':>6s} {'vs naive':>9s}")
    for name in ("naive", "ar1", "target_only", "transfer"):
        s = summary[name]
        print(f"{name:13s} {s['mae']:8.2f} {s['rmse']:8.2f} {s['mape']:6.2f}% "
              f"{s['direction_accuracy']:6.3f} {s['mae_improvement_vs_naive_pct']:8.2f}%")
    print(f"\ntransfer vs target_only: "
          f"{summary['transfer']['mae_improvement_vs_target_only_pct']:+.2f}% MAE  "
          f"(wins on {summary['transfer']['varieties_where_transfer_wins']} varieties)")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
