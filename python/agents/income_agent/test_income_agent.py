"""
Run:  pytest test_income_agent.py -v
Part A: the math.  Part B: full chain with FAKE Soil/Market servers (no real agents needed).
"""
import os
import sys
from concurrent import futures

import grpc
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "../../shared"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paddy_agents_pb2 as pb2
import paddy_agents_pb2_grpc as pb2_grpc

import income_logic as logic


# ─────────────── PART A — math ───────────────

def test_fertilizer_cost_one_urea_bag():
    assert logic.fertilizer_cost_per_ha(45, 0, 0) == pytest.approx(266.50)


def test_quintals_conversion():
    assert logic.total_quintals(4000, 2) == 80          # 8000 kg = 80 quintals


def test_income_formula():
    gross, net = logic.income(80, 2500, 30000)
    assert gross == 200000 and net == 170000


def test_price_never_below_msp():
    assert logic.floor_at_msp(1500) == logic.MSP_RS_PER_QUINTAL
    assert logic.floor_at_msp(0) == logic.MSP_RS_PER_QUINTAL
    assert logic.floor_at_msp(3000) == 3000


def test_hold_scenario_subtracts_storage_cost():
    p = logic.price_scenarios(2600, 2800)
    assert p["sell_now"] == 2600
    assert p["hold_1_month"] == 2800 - logic.HOLDING_COST_RS_PER_QUINTAL_MONTH


def test_cost_of_production_from_msp():
    # MSP is 1.5 x A2+FL cost, so cost = 2441 / 1.5
    assert logic.COST_OF_PRODUCTION_RS_PER_QUINTAL == pytest.approx(1627.33, abs=0.01)


def test_default_cost_does_not_double_count_fertilizer():
    # 40 q/ha x 1627.33 = 65,093 total ; minus 5,000 fertilizer = 60,093 other
    assert logic.default_other_cost_per_ha(4000, 5000) == pytest.approx(60093.2, abs=1)


def test_invalid_area_rejected():
    with pytest.raises(ValueError):
        logic.total_quintals(4000, 0)


# ─────────────── PART B — full chain with fake agents ───────────────

class FakeSoil(pb2_grpc.SoilAgentServiceServicer):
    def GetSoilHealth(self, req, ctx):
        return pb2.SoilResponse(district=req.district, predicted_yield_kg_per_ha=4000,
                                urea_kg_per_ha=45, dap_kg_per_ha=0, potash_kg_per_ha=0)


class FakeMarket(pb2_grpc.MarketAgentServiceServicer):
    def GetMarketAdvisory(self, req, ctx):
        return pb2.MarketResponse(current_price=2600, best_month="August")


def _start(servicer, adder):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    adder(servicer, server)
    port = server.add_insecure_port("localhost:0")
    server.start()
    return server, f"localhost:{port}"


@pytest.fixture
def client(monkeypatch, request):
    down = getattr(request, "param", set())
    import income_agent
    servers = []
    for name, servicer, adder, attr in [
        ("soil", FakeSoil(), pb2_grpc.add_SoilAgentServiceServicer_to_server, "SOIL_ADDR"),
        ("market", FakeMarket(), pb2_grpc.add_MarketAgentServiceServicer_to_server, "MARKET_ADDR"),
    ]:
        if name in down:
            monkeypatch.setattr(income_agent, attr, "localhost:1")
        else:
            s, addr = _start(servicer, adder)
            servers.append(s)
            monkeypatch.setattr(income_agent, attr, addr)
    monkeypatch.setattr(income_agent, "TIMEOUT_S", 2)
    s, addr = _start(income_agent.IncomeAgentServicer(),
                     pb2_grpc.add_IncomeAgentServiceServicer_to_server)
    servers.append(s)
    yield pb2_grpc.IncomeAgentServiceStub(grpc.insecure_channel(addr))
    for s in servers:
        s.stop(None)


REQ = pb2.IncomeRequest(district="Mandya", season="Kharif", variety="BPT-5204", area_ha=2)


def test_full_chain_with_farmer_costs(client):
    req = pb2.IncomeRequest(district="Mandya", season="Kharif", variety="BPT-5204",
                            area_ha=2, other_cost_per_ha=40000)
    r = client.GetIncomeEstimate(req)
    # 4000 kg/ha x 2 ha = 80 q x Rs 2600 = 208,000
    # costs: fertilizer 266.50 x 2 = 533 ; other 40,000 x 2 = 80,000
    assert r.total_yield_quintals == 80
    assert r.fertilizer_cost_total == pytest.approx(533)
    assert r.other_cost_total == pytest.approx(80000)
    assert r.expected_net_income == pytest.approx(208000 - 533 - 80000)
    assert {s.name for s in r.scenarios} == {"msp", "sell_now", "hold_1_month"}
    assert r.best_month_to_sell == "August"
    assert len(r.warnings) == 0


def test_full_chain_uses_default_cost(client):
    r = client.GetIncomeEstimate(REQ)          # other_cost_per_ha not given
    # total cost = 1627.33 x 40 q/ha x 2 ha ; fertilizer is part of it
    total_cost = logic.COST_OF_PRODUCTION_RS_PER_QUINTAL * 40 * 2
    assert r.fertilizer_cost_total + r.other_cost_total == pytest.approx(total_cost, abs=1)
    assert any("Other costs" in w for w in r.warnings)


@pytest.mark.parametrize("client", [{"market"}], indirect=True)
def test_market_down_uses_msp(client):
    r = client.GetIncomeEstimate(REQ)
    sell_now = next(s for s in r.scenarios if s.name == "sell_now")
    assert sell_now.price_per_quintal == pytest.approx(logic.MSP_RS_PER_QUINTAL)
    assert any("Market" in w for w in r.warnings)


@pytest.mark.parametrize("client", [{"soil"}], indirect=True)
def test_soil_down_gives_clear_error(client):
    with pytest.raises(grpc.RpcError) as e:
        client.GetIncomeEstimate(REQ)
    assert e.value.code() == grpc.StatusCode.UNAVAILABLE