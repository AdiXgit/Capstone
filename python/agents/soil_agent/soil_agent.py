"""Soil Agent — NPK profiling, fertilizer recommendation, soil trend, and a
combined soil-health score.

Fixes applied:
  * Uses the correct generated base class `SoilAgentServiceServicer`
    (was `SoilAgentServicer`, which does not exist → the agent crashed on start).
  * Implements GetSoilHealth (called by the orchestrator's SOIL path).
  * Attaches AgentMetadata to responses so the orchestrator can log provenance.
  * Registers with the orchestrator on startup.
  * scipy is imported lazily inside GetSoilTrend so the agent still starts (and
    serves NPK / health) even if scipy is not installed.
"""
import os
import sys
import time
import logging
import threading
from concurrent import futures

import numpy as np
import pandas as pd
import grpc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
sys.path.insert(0, os.path.dirname(__file__))

import paddy_agents_pb2 as pb2
import paddy_agents_pb2_grpc as pb2_grpc

from npk_profiles import get_npk_for_district

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SOIL] %(levelname)s %(message)s")
log = logging.getLogger("SOIL")

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../data/Karnataka_Paddy_Main_Dataset.csv")


def _meta(conf=0.85, notes=""):
    return pb2.AgentMetadata(
        agent_id="soil-agent-001", agent_name="Soil Intelligence Agent",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        confidence=conf, notes=notes)


class _SoilStore:
    """Loads the main dataset once and computes soil-health inputs (pH, OC, score)."""

    def __init__(self, path):
        self.df = pd.read_csv(path)

    def health(self, district, season):
        d = self.df[self.df["District"] == district.strip().title()]
        if d.empty:
            d = self.df
        ds = d[d["Season"] == season] if season else d
        if ds.empty:
            ds = d

        n = float(d["Soil_Nitrogen_kg_per_ha"].mean())
        p = float(d["Soil_Phosphorus_kg_per_ha"].mean())
        k = float(d["Soil_Potassium_kg_per_ha"].mean())
        ph = float(d["Soil_pH"].mean())
        oc = float(d["Soil_Organic_Carbon_Percent"].mean())

        # 0-10 agronomic health score for paddy (same formula as the gateway).
        ph_score = max(0.0, 1 - abs(ph - 6.5) / 2.0)
        oc_score = min(1.0, oc / 0.75)
        n_score = min(1.0, n / 280.0)
        p_score = min(1.0, p / 25.0)
        k_score = min(1.0, k / 200.0)
        score = min(10.0, round(ph_score * 2.5 + oc_score * 2.5 + n_score * 2 + p_score * 1.5 + k_score * 1.5, 2))

        quality = str(d["Soil_Quality"].mode()[0]) if "Soil_Quality" in d else "medium"
        kvk = {"high": (100, 50, 30), "medium": (130, 65, 40), "low": (160, 80, 50)}
        urea, dap, potash = kvk.get(quality.lower(), kvk["medium"])
        if season == "Rabi":
            urea, dap, potash = round(urea * 0.85), round(dap * 0.90), round(potash * 0.90)
        rec = (f"Soil health {score}/10 ({quality}). Apply ~{urea} kg/acre Urea, {dap} kg/acre DAP, "
               f"{potash} kg/acre Potash. " +
               ("Organic carbon low — add green manure. " if oc < 0.5 else "") +
               ("pH acidic — apply lime. " if ph < 6.0 else "pH alkaline — check irrigation water. " if ph > 7.5 else ""))
        return {"score": score, "n": n, "p": p, "k": k, "ph": ph, "oc": oc, "recommendation": rec.strip()}


class SoilAgentServicer(pb2_grpc.SoilAgentServiceServicer):
    def __init__(self, store: _SoilStore):
        self.store = store

    def GetNPKProfile(self, request, context):
        try:
            npk = get_npk_for_district(request.district)
            return pb2.NPKResponse(metadata=_meta(0.9), district=request.district,
                                   nitrogen=npk["N"], phosphorus=npk["P"], potassium=npk["K"])
        except ValueError as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return pb2.NPKResponse()

    def GetSoilHealth(self, request, context):
        log.info("GetSoilHealth -> %s %s", request.district, request.season)
        h = self.store.health(request.district, request.season or "Kharif")
        return pb2.SoilHealthResponse(
            metadata=_meta(0.88), district=request.district,
            soil_health_score=h["score"], recommendation=h["recommendation"],
            nitrogen=h["n"], phosphorus=h["p"], potassium=h["k"],
            ph=h["ph"], organic_carbon=h["oc"])

    def GetFertilizerRec(self, request, context):
        try:
            from fertilizer_model import predict_fertilizer
            r = predict_fertilizer(
                district=request.district, nitrogen=request.nitrogen,
                phosphorus=request.phosphorus, potassium=request.potassium,
                ph=request.ph, organic_carbon=request.organic_carbon, season=request.season)
            return pb2.FertilizerResponse(
                metadata=_meta(0.85),
                urea_kg_per_acre=r["urea_kg_per_acre"], dap_kg_per_acre=r["dap_kg_per_acre"],
                potash_kg_per_acre=r["potash_kg_per_acre"], timing_advice=r["timing_advice"],
                season=r["season"])
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.FertilizerResponse()

    def GetSoilTrend(self, request, context):
        try:
            from soil_trend import get_soil_trend  # lazy: pulls in scipy
            t = get_soil_trend(request.district)
            return pb2.SoilTrendResponse(
                metadata=_meta(0.85), district=t["district"],
                ph_slope=t["ph_slope"], ph_last_value=t["ph_last_value"],
                oc_slope=t["oc_slope"], oc_last_value=t["oc_last_value"],
                interpretation=t["interpretation"])
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.SoilTrendResponse()


def register_with_orchestrator(port, max_retries=3):
    default_addr = "orchestrator:50051" if os.path.exists("/.dockerenv") else "localhost:50051"
    addr = os.environ.get("ORCHESTRATOR_ADDR", default_addr)
    host = os.environ.get("AGENT_ADVERTISE_HOST", "soil-agent" if os.path.exists("/.dockerenv") else "localhost")
    for i in range(max_retries):
        try:
            with grpc.insecure_channel(addr) as ch:
                pb2_grpc.OrchestratorServiceStub(ch).RegisterAgent(
                    pb2.AgentRegistration(agent_id="soil-agent-001",
                                          agent_name="Soil Intelligence Agent",
                                          agent_type="SOIL", host=host, port=port), timeout=3)
                log.info("Registered with orchestrator at %s", addr)
                return
        except grpc.RpcError:
            if i < max_retries - 1:
                time.sleep(2)
    log.info("No orchestrator at %s — serving standalone.", addr)


def serve():
    port = int(os.environ.get("SOIL_AGENT_PORT", "50052"))
    store = _SoilStore(DATA_PATH)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_SoilAgentServiceServicer_to_server(SoilAgentServicer(store), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("Soil Agent running on :%d", port)
    threading.Thread(target=register_with_orchestrator, args=(port,), daemon=True).start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
