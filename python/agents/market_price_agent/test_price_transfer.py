import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from price_transfer import (
    N_FEATURES, RETURN_SCALE, VARIETIES, WINDOW, PriceGRU,
    build_windows, evaluate, load_panel, predict_returns, split_by_date,
)


@pytest.fixture(scope="module")
def panel():
    return load_panel()


def test_all_varieties_present(panel):
    assert sorted(panel["Variety"].unique()) == sorted(VARIETIES)


def test_window_shapes(panel):
    X, y, dates, prices = build_windows(panel, "BPT-5204")
    assert X.shape[1:] == (WINDOW, N_FEATURES)
    assert len(X) == len(y) == len(dates) == len(prices)


def test_no_nans_reach_the_model(panel):
    """log_ret[0] is NaN by construction; windows must start past it."""
    X, y, _, _ = build_windows(panel, "IR-64")
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()


def test_features_never_include_the_predicted_week(panel):
    """The window must close strictly before the week being forecast.

    This is the leakage guard the whole experiment rests on: `prices[i]` is the
    last price the model is allowed to see, so it must be the week *before*
    dates[i], and applying y[i] to it must reproduce the actual price at dates[i].
    """
    X, y, dates, prices = build_windows(panel, "Jyothi")
    raw = panel[panel["Variety"] == "Jyothi"].sort_values("Date").reset_index(drop=True)
    lookup = dict(zip(raw["Date"], raw["Price_Rs_per_Quintal"]))

    for i in (0, 25, len(X) // 2, len(X) - 1):
        predicted_week = dates[i]
        prev_week = predicted_week - pd.Timedelta(weeks=1)
        assert prices[i] == lookup[prev_week]
        reconstructed = prices[i] * np.exp(y[i] * RETURN_SCALE)
        assert abs(reconstructed - lookup[predicted_week]) < 0.5


def test_splits_are_temporal_and_disjoint(panel):
    _, _, dates, _ = build_windows(panel, "MTU-1010")
    tr, va, te = split_by_date(dates)
    assert (tr & va).sum() == 0 and (va & te).sum() == 0 and (tr & te).sum() == 0
    assert tr.sum() + va.sum() + te.sum() == len(dates)
    # every training date precedes every validation date, and so on
    assert dates[tr].max() < dates[va].min() < dates[te].min()
    assert va.sum() > 40 and te.sum() > 80


def test_evaluate_is_exact_on_perfect_forecast():
    true_ret = np.array([0.01, -0.02, 0.03])
    price = np.array([2500.0, 2400.0, 2600.0])
    m = evaluate(true_ret, true_ret, price)
    assert m["mae"] == pytest.approx(0.0)
    assert m["direction_accuracy"] == pytest.approx(1.0)


def test_evaluate_scores_inverted_forecast_at_zero_direction():
    true_ret = np.array([0.01, -0.02, 0.03])
    price = np.array([2500.0, 2400.0, 2600.0])
    assert evaluate(-true_ret, true_ret, price)["direction_accuracy"] == pytest.approx(0.0)


def test_flat_weeks_excluded_from_direction():
    true_ret = np.array([0.0, 0.02])
    price = np.array([2500.0, 2500.0])
    # Only the second week has a direction to get right.
    assert evaluate(np.array([0.5, 0.01]), true_ret, price)["direction_accuracy"] == 1.0


def test_model_is_small_enough_for_the_data(panel):
    """~365 target training weeks; a large model would simply memorise them."""
    n_params = sum(p.numel() for p in PriceGRU().parameters())
    assert n_params < 5000, f"PriceGRU has {n_params} params"


def test_prediction_shape_and_scale(panel):
    X, _, _, _ = build_windows(panel, "JGL-1798")
    out = predict_returns(PriceGRU(), X[:16])
    assert out.shape == (16,)
    # Untrained output must still be a plausible weekly return, not a price.
    assert np.abs(out).max() < 1.0
