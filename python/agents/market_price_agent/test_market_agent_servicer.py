"""
test_market_agent.py's 10 tests all call get_market_advisory() directly --
none go through MarketAgentServicer.GetMarketAdvisory(), which is the only
place that (a) reads the variety off the real MarketRequest proto and
(b) forwards sarima_next_month_forecast into MarketResponse. Both of those
were broken (request.variety doesn't exist -- the field is request.crop --
and the SARIMA fields were computed but never copied into the response) and
10/10 green the whole time.
"""
import importlib.util
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../shared"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paddy_agents_pb2 as pb2

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("market_agent_module", os.path.join(_here, "market_agent.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MarketAgentServicer = _mod.MarketAgentServicer


class FakeContext:
    def __init__(self):
        self.code = None
        self.details = None
    def set_code(self, c): self.code = c
    def set_details(self, d): self.details = d


def test_crop_field_is_read_correctly():
    svc = MarketAgentServicer()
    r = svc.GetMarketAdvisory(pb2.MarketRequest(district="Mandya", crop="Jyothi"), FakeContext())
    assert r.variety == "Jyothi"


def test_sarima_forecast_reaches_the_response():
    svc = MarketAgentServicer()
    r = svc.GetMarketAdvisory(pb2.MarketRequest(district="Mandya", crop="BPT-5204"), FakeContext())
    assert r.sarima_status == "ok"
    assert r.sarima_next_month_forecast > 0
    assert r.sarima_next_month_forecast != r.current_price


def test_advisory_uses_sarima_direction_not_the_weak_slope_label():
    svc = MarketAgentServicer()
    r = svc.GetMarketAdvisory(pb2.MarketRequest(district="Mandya", crop="BPT-5204"), FakeContext())
    if r.sarima_status == "ok":
        assert "SARIMA forecasts" in r.advisory or "SARIMA does not forecast" in r.advisory
        assert "Short-term price trend is" not in r.advisory


def test_no_crop_given_defaults_sensibly():
    svc = MarketAgentServicer()
    ctx = FakeContext()
    r = svc.GetMarketAdvisory(pb2.MarketRequest(district="Mandya", crop=""), ctx)
    assert ctx.code is None
    assert r.variety == "BPT-5204"