"""
Farm Income Agent (Aman) - Port 50057
Soil Agent (50052)   -> yield + fertilizer doses
Market Agent (50055) -> current price, forecast, best month
"""

import os
import sys
from concurrent import futures
from datetime import datetime

import grpc

sys.path.append(os.path.join(os.path.dirname(__file__), "../../shared"))
import paddy_agents_pb2 as pb2            # noqa: E402
import paddy_agents_pb2_grpc as pb2_grpc  # noqa: E402

import income_logic as logic               # noqa: E402

PORT = int(os.getenv("INCOME_AGENT_PORT", "50057"))
SOIL_ADDR = os.getenv("SOIL_AGENT_ADDR", "localhost:50052")
MARKET_ADDR = os.getenv("MARKET_AGENT_ADDR", "localhost:50055")
TIMEOUT_S = float(os.getenv("AGENT_TIMEOUT_S", "30"))   # SARIMA fit can take a few seconds


class IncomeAgentServicer(pb2_grpc.IncomeAgentServiceServicer):

    def __init__(self):
        self.soil = pb2_grpc.SoilAgentServiceStub(grpc.insecure_channel(SOIL_ADDR))
        self.market = pb2_grpc.MarketAgentServiceStub(grpc.insecure_channel(MARKET_ADDR))

    def _get_market(self, district, variety, warnings):
        try:
            m = self.market.GetMarketAdvisory(
                pb2.MarketRequest(district=district, crop=variety), timeout=TIMEOUT_S)
            forecast = getattr(m, "sarima_next_month_forecast", 0.0) or m.current_price
            return m.current_price, forecast, m.best_month
        except grpc.RpcError as e:
            warnings.append(f"Market Agent unreachable ({e.code().name}); used MSP as price.")
            return 0.0, 0.0, ""

    def GetIncomeEstimate(self, request, context):
        warnings = []
        area = request.area_ha
        if area <= 0:
            area = 1.0
            warnings.append("area_ha not given; assumed 1 hectare.")

        # 1. Soil Agent -> yield + fertilizer (required)
        try:
            soil = self.soil.GetSoilHealth(
                pb2.SoilRequest(district=request.district, season=request.season),
                timeout=TIMEOUT_S)
        except grpc.RpcError as e:
            context.abort(grpc.StatusCode.UNAVAILABLE,
                          f"Soil Agent unreachable ({e.code().name}); cannot estimate yield.")

        quintals = logic.total_quintals(soil.predicted_yield_kg_per_ha, area)
        fert_per_ha = logic.fertilizer_cost_per_ha(
            soil.urea_kg_per_ha, soil.dap_kg_per_ha, soil.potash_kg_per_ha)
        fert_total = round(fert_per_ha * area, 2)

        # Other costs: farmer's own figure, or the CACP-based default
        if request.other_cost_per_ha > 0:
            other_per_ha = request.other_cost_per_ha
        else:
            other_per_ha = logic.default_other_cost_per_ha(
                soil.predicted_yield_kg_per_ha, fert_per_ha)
            warnings.append(
                f"Other costs not given; used CACP-based estimate of "
                f"Rs {other_per_ha:,.0f}/ha (Rs {logic.COST_OF_PRODUCTION_RS_PER_QUINTAL:,.0f}/quintal).")
        other_total = round(other_per_ha * area, 2)
        costs = fert_total + other_total

        # 2. Market Agent -> prices (falls back to MSP)
        price_now, price_forecast, best_month = self._get_market(
            request.district, request.variety, warnings)

        # 3. Income under each price scenario
        prices = logic.price_scenarios(price_now, price_forecast)
        scenarios, net_by = [], {}
        for name, price in prices.items():
            gross, net = logic.income(quintals, price, costs)
            net_by[name] = net
            scenarios.append(pb2.PriceScenario(
                name=name, price_per_quintal=price, gross_income=gross, net_income=net))

        recommendation = "hold" if net_by["hold_1_month"] > net_by["sell_now"] else "sell now"

        return pb2.IncomeResponse(
            metadata=pb2.AgentMetadata(
                agent_id="income-agent", agent_name="Farm Income Agent",
                timestamp=datetime.now().isoformat(timespec="seconds"),
                confidence=0.9 if all("Other costs" in w for w in warnings) else 0.6),
            district=request.district,
            variety=request.variety,
            area_ha=area,
            yield_kg_per_ha=soil.predicted_yield_kg_per_ha,
            total_yield_quintals=quintals,
            fertilizer_cost_total=fert_total,
            other_cost_total=other_total,
            scenarios=scenarios,
            expected_net_income=net_by["sell_now"],
            recommendation=recommendation,
            best_month_to_sell=best_month,
            advisory=logic.build_advisory(net_by, best_month, warnings),
            warnings=warnings,
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_IncomeAgentServiceServicer_to_server(IncomeAgentServicer(), server)
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"Farm Income Agent gRPC server running on port {PORT}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()