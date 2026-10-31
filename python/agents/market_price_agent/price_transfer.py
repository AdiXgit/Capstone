"""Cross-variety transfer learning for weekly paddy price forecasting.

Why transfer is the right tool here, from the data itself:

  * each variety has only 522 weekly observations, and after holding out a
    validation and test period only ~365 remain to train on;
  * all five varieties share near-identical dynamics -- weekly volatility of
    4.6-4.9% and lag-1 autocorrelation of weekly *changes* of about -0.4
    (strong mean reversion);
  * yet their week-to-week shocks correlate only 0.14-0.31.

So the other varieties teach the shape of the process without revealing the
target's own movements. Pretraining on four varieties and fine-tuning on the
fifth is therefore a genuine 4x increase in training signal rather than a
disguised form of leakage.

The model predicts the next week's *log return*, not its price level. That is
what makes the weights transferable at all: price levels differ by hundreds of
rupees across varieties, whereas returns are scale-free and directly
comparable.
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../data/Karnataka_Market_Prices.csv")

VARIETIES = ["BPT-5204", "IR-64", "JGL-1798", "Jyothi", "MTU-1010"]
DEMAND_LEVELS = ["Low", "Medium", "High"]
SUPPLY_LEVELS = ["Deficit", "Normal", "Surplus"]

WINDOW = 26          # weeks of history fed to the model
RETURN_SCALE = 0.05  # ~1 std of weekly returns, so inputs land near unit scale

# Fixed calendar split. Dates not random indices: shuffling a time series would
# let the model train on the future it is being asked to predict.
TRAIN_END = "2021-12-31"
VAL_END = "2022-12-31"

N_FEATURES = 1 + 2 + len(DEMAND_LEVELS) + len(SUPPLY_LEVELS)


def load_panel(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values(["Variety", "Date"]).reset_index(drop=True)


def _one_hot(series, levels):
    codes = pd.Categorical(series, categories=levels).codes
    out = np.zeros((len(series), len(levels)), dtype=np.float32)
    valid = codes >= 0
    out[np.arange(len(series))[valid], codes[valid]] = 1.0
    return out


def build_windows(df, variety):
    """Return (X, y, dates, prices) of supervised next-week-return examples.

    X[i] has shape (WINDOW, N_FEATURES); y[i] is the log return realised in the
    week after the window closes. dates[i] is the date being predicted.
    """
    v = df[df["Variety"] == variety].sort_values("Date").reset_index(drop=True)
    price = v["Price_Rs_per_Quintal"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(price), prepend=np.nan)

    week = v["Date"].dt.isocalendar().week.to_numpy(dtype=np.float64)
    seasonal = np.stack([np.sin(2 * np.pi * week / 52.0),
                         np.cos(2 * np.pi * week / 52.0)], axis=1).astype(np.float32)
    demand = _one_hot(v["Market_Demand"], DEMAND_LEVELS)
    supply = _one_hot(v["Supply_Status"], SUPPLY_LEVELS)

    feats = np.concatenate(
        [(log_ret / RETURN_SCALE).astype(np.float32)[:, None], seasonal, demand, supply],
        axis=1,
    )

    X, y, dates, prices = [], [], [], []
    # start at 1 because log_ret[0] is NaN; need WINDOW rows then one target
    for t in range(1 + WINDOW, len(v)):
        X.append(feats[t - WINDOW:t])
        y.append(log_ret[t] / RETURN_SCALE)
        dates.append(v["Date"].iloc[t])
        prices.append(price[t - 1])  # last observed price, for reconstruction

    return (np.stack(X), np.array(y, dtype=np.float32),
            pd.to_datetime(dates), np.array(prices))


def split_by_date(dates, train_end=TRAIN_END, val_end=VAL_END):
    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    return (dates <= train_end,
            (dates > train_end) & (dates <= val_end),
            dates > val_end)


class PriceGRU(nn.Module):
    """Deliberately small: ~2.5k parameters for ~365 target-variety samples."""

    def __init__(self, n_features=N_FEATURES, hidden=24, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(self.drop(h[-1])).squeeze(-1)


def train_model(model, X, y, Xval, yval, epochs=120, lr=3e-3, batch_size=32,
                weight_decay=1e-4, patience=20, seed=0):
    """Fit with early stopping on validation MSE; returns the best state."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    Xv, yv = torch.from_numpy(Xval), torch.from_numpy(yval)

    best = {"loss": float("inf"), "state": None, "epoch": 0}
    since_improved = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(Xv), yv))

        if val_loss < best["loss"] - 1e-6:
            best = {"loss": val_loss,
                    "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "epoch": epoch}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best


@torch.no_grad()
def predict_returns(model, X):
    model.eval()
    return model(torch.from_numpy(X)).numpy() * RETURN_SCALE


def evaluate(pred_log_ret, true_log_ret, last_price):
    """Score in rupees, plus the direction accuracy a sell/hold call needs."""
    pred_price = last_price * np.exp(pred_log_ret)
    true_price = last_price * np.exp(true_log_ret)

    err = pred_price - true_price
    # Flat weeks are excluded from direction scoring: there is no direction to get right.
    moved = np.abs(true_log_ret) > 1e-9
    direction = (
        float((np.sign(pred_log_ret[moved]) == np.sign(true_log_ret[moved])).mean())
        if moved.any() else float("nan")
    )

    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mape": float((np.abs(err) / true_price).mean() * 100),
        "direction_accuracy": float(direction),
        "n": int(len(err)),
    }
