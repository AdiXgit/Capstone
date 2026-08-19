from market_analysis import get_market_advisory, VALID_VARIETIES

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