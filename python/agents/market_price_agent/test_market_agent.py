from market_analysis import get_market_advisory, fit_sarima, load_market_data, VALID_VARIETIES

def test_all_varieties_return_valid_data():
    for variety in VALID_VARIETIES:
        result = get_market_advisory(variety)
        assert result["current_price"] > 0
        assert result["price_trend"] in ["rising", "falling", "stable"]
        assert result["market_demand"] in ["High", "Medium", "Low"]
        assert result["supply_status"] in ["Surplus", "Normal", "Deficit"]

def test_invalid_variety_raises():
    try:
        get_market_advisory("FakeVariety")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_best_month_is_valid_month_name():
    valid_months = ["January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November", "December"]
    for variety in VALID_VARIETIES:
        result = get_market_advisory(variety)
        assert result["best_month"] in valid_months


def test_sarima_status_is_valid():
    for variety in VALID_VARIETIES:
        result = get_market_advisory(variety)
        assert result["sarima_status"] in ["ok", "insufficient_history", "fit_failed"]


def test_sarima_produces_numeric_outputs_when_ok():
    for variety in VALID_VARIETIES:
        result = get_market_advisory(variety)
        if result["sarima_status"] == "ok":
            assert isinstance(result["sarima_seasonal_expected_price"], (int, float))
            assert isinstance(result["sarima_seasonal_deviation_pct"], (int, float))
            assert isinstance(result["sarima_next_month_forecast"], (int, float))
            assert result["sarima_seasonal_expected_price"] > 0
            assert result["sarima_next_month_forecast"] > 0


def test_sarima_seasonal_expected_price_is_realistic():
    for variety in VALID_VARIETIES:
        df = load_market_data(variety)
        actual_min = df["Price_Rs_per_Quintal"].min()
        actual_max = df["Price_Rs_per_Quintal"].max()

        result = get_market_advisory(variety)
        if result["sarima_status"] == "ok":
            expected = result["sarima_seasonal_expected_price"]
            lower_bound = actual_min * 0.8
            upper_bound = actual_max * 1.2
            assert lower_bound <= expected <= upper_bound, (
                f"{variety}: SARIMA expected price {expected} is outside "
                f"realistic range [{lower_bound:.0f}, {upper_bound:.0f}]"
            )


def test_sarima_deviation_pct_matches_manual_calculation():
    for variety in VALID_VARIETIES:
        result = get_market_advisory(variety)
        if result["sarima_status"] == "ok":
            current = result["current_price"]
            expected = result["sarima_seasonal_expected_price"]
            reported_dev = result["sarima_seasonal_deviation_pct"]

            manual_dev = round(((current - expected) / expected) * 100, 1)
            assert abs(manual_dev - reported_dev) < 0.2, (
                f"{variety}: reported deviation {reported_dev}% doesn't match "
                f"manual calculation {manual_dev}%"
            )


def test_sarima_forecast_direction_is_plausible():
    for variety in VALID_VARIETIES:
        result = get_market_advisory(variety)
        if result["sarima_status"] == "ok":
            current = result["current_price"]
            forecast = result["sarima_next_month_forecast"]
            pct_change = abs((forecast - current) / current) * 100
            assert pct_change < 50, (
                f"{variety}: 1-month forecast changed {pct_change:.1f}% — "
                f"likely unstable SARIMA fit"
            )


def test_advisory_mentions_seasonal_context_when_sarima_ok():
    for variety in VALID_VARIETIES:
        result = get_market_advisory(variety)
        if result["sarima_status"] == "ok":
            advisory_lower = result["advisory"].lower()
            assert "seasonal" in advisory_lower or "sarima" in advisory_lower


def test_fit_sarima_returns_none_gracefully_on_insufficient_data():
    import pandas as pd
    tiny_df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "Price_Rs_per_Quintal": [2000, 2010, 2005, 2020, 2015, 2030, 2025, 2040, 2035, 2050],
    })
    forecast, expected, status = fit_sarima(tiny_df)
    assert status == "insufficient_history"
    assert forecast is None
    assert expected is None