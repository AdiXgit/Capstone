"""Market Price Agent (gRPC, :50055).

Serves price trend + sell/hold advisory computed from Karnataka_Market_Prices.csv.
Previously this logic lived only in the gateway; it is now a real registered agent
so the mesh is complete (System Status shows it ONLINE, served by gRPC).
"""
import os
import sys
import time
import logging
import threading
from datetime import datetime
from concurrent import futures

import numpy as np
import pandas as pd
import grpc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))

import market_pest_pb2 as pb
import market_pest_pb2_grpc as pb_grpc
import paddy_agents_pb2 as core_pb          # for orchestrator registration
import paddy_agents_pb2_grpc as core_grpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MARKET] %(levelname)s %(message)s")
log = logging.getLogger("MARKET")

DATA = os.path.join(os.path.dirname(__file__), "../../../data/Karnataka_Market_Prices.csv")
MSP_2025 = 2320.0


def _meta(conf=0.85, notes=""):
    return pb.AgentMeta(agent_id="market-agent-001", agent_name="Market Price Agent",
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        confidence=conf, notes=notes)


class MarketStore:
    def __init__(self, path):
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        self.df = df

    def advisory(self, variety):
        df = self.df
        varieties = sorted(df["Variety"].dropna().unique().tolist())
        chosen = variety if variety in varieties else varieties[0]
        d = df[df["Variety"] == chosen].sort_values("Date")
        recent = d.tail(30)
        current = float(recent["Price_Rs_per_Quintal"].iloc[-1])
        prev = float(recent["Price_Rs_per_Quintal"].iloc[0])
        slope = float(np.polyfit(range(len(recent)), recent["Price_Rs_per_Quintal"], 1)[0]) if len(recent) > 1 else 0.0
        direction = "RISING" if slope > 1.5 else "FALLING" if slope < -1.5 else "STABLE"
        sell = "SELL" if current >= MSP_2025 else "HOLD"
        if direction == "FALLING" and current >= MSP_2025:
            sell = "SELL"
        monthly = d.assign(Month=d["Date"].dt.month).groupby("Month")["Price_Rs_per_Quintal"].mean()
        best_month = datetime(2025, int(monthly.idxmax()) if len(monthly) else 1, 1).strftime("%B")
        reasoning = (f"{chosen} is trading at Rs{current:,.0f}/qtl, "
                     f"{'above' if current >= MSP_2025 else 'below'} the Rs{MSP_2025:,.0f} MSP. "
                     f"30-day trend is {direction.lower()} ({'+' if slope >= 0 else ''}{slope:.1f} Rs/day). "
                     f"Prices historically peak in {best_month}.")
        return {"variety": chosen, "current": current, "change": current - prev,
                "direction": direction, "sell": sell, "best_month": best_month,
                "reasoning": reasoning, "demand": str(recent["Market_Demand"].iloc[-1]),
                "supply": str(recent["Supply_Status"].iloc[-1])}


class MarketServicer(pb_grpc.MarketAgentServiceServicer):
    def __init__(self, store):
        self.store = store

    def GetMarketAdvisory(self, request, context):
        log.info("GetMarketAdvisory -> %s %s", request.district, request.variety)
        a = self.store.advisory(request.variety)
        return pb.MarketResponse(
            metadata=_meta(0.86), district=request.district, crop_variety=a["variety"],
            current_price_per_quintal=a["current"], msp_price_per_quintal=MSP_2025,
            price_change_30d=a["change"], trend_direction=a["direction"],
            sell_or_hold=a["sell"], best_selling_month=a["best_month"],
            reasoning=a["reasoning"], market_demand=a["demand"], supply_status=a["supply"])


def register(port):
    default_addr = "orchestrator:50051" if os.path.exists("/.dockerenv") else "localhost:50051"
    addr = os.environ.get("ORCHESTRATOR_ADDR", default_addr)
    host = os.environ.get("AGENT_ADVERTISE_HOST", "market-agent" if os.path.exists("/.dockerenv") else "localhost")
    for i in range(3):
        try:
            with grpc.insecure_channel(addr) as ch:
                core_grpc.OrchestratorServiceStub(ch).RegisterAgent(
                    core_pb.AgentRegistration(agent_id="market-agent-001", agent_name="Market Price Agent",
                                              agent_type="MARKET_PRICE", host=host, port=port), timeout=3)
                log.info("Registered with orchestrator at %s", addr)
                return
        except grpc.RpcError:
            if i < 2:
                time.sleep(2)
    log.info("No orchestrator at %s — serving standalone.", addr)


def serve():
    port = int(os.environ.get("MARKET_AGENT_PORT", "50055"))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb_grpc.add_MarketAgentServiceServicer_to_server(MarketServicer(MarketStore(DATA)), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("Market Price Agent running on :%d", port)
    threading.Thread(target=register, args=(port,), daemon=True).start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
