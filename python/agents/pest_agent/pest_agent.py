"""Pest Risk Agent (gRPC, :50056).

KVK rule engine: combines historical district/season incidence with the current
stage and weather conditions into a risk level, likely pests/diseases and a
preventive action. Weather values are passed in the request (supplied by the
caller/orchestrator) so this agent stays independent.
"""
import os
import sys
import time
import logging
import threading
from concurrent import futures

import pandas as pd
import grpc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../crop_health_agent"))

import market_pest_pb2 as pb
import market_pest_pb2_grpc as pb_grpc
import paddy_agents_pb2 as core_pb
import paddy_agents_pb2_grpc as core_grpc
from treatment_map import PEST_BY_STAGE, DISEASE_BY_STAGE, PEST_DISEASE_TREATMENT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PEST] %(levelname)s %(message)s")
log = logging.getLogger("PEST")

DATA = os.path.join(os.path.dirname(__file__), "../../../data/Karnataka_Paddy_Main_Dataset.csv")


def _meta(conf=0.83, notes=""):
    return pb.AgentMeta(agent_id="pest-agent-001", agent_name="Pest Risk Agent",
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        confidence=conf, notes=notes)


class PestStore:
    def __init__(self, path):
        self.df = pd.read_csv(path, usecols=["District", "Season", "Pest_Incidence", "Disease_Incidence"])

    def history(self, district, season):
        sub = self.df[(self.df["District"] == district) & (self.df["Season"] == season)]
        if sub.empty:
            sub = self.df[self.df["Season"] == season]
        if sub.empty:
            return {"pest": "Medium", "disease": "Medium"}
        return {"pest": sub["Pest_Incidence"].mode()[0], "disease": sub["Disease_Incidence"].mode()[0]}


class PestServicer(pb_grpc.PestAgentServiceServicer):
    def __init__(self, store):
        self.store = store

    def GetPestRisk(self, request, context):
        stage = request.growth_stage or "Vegetative"
        temp = request.temperature or 30.0
        humidity = request.humidity or 70.0
        rainfall = request.rainfall or 0.0
        log.info("GetPestRisk -> %s %s stage=%s", request.district, request.season, stage)

        hist = self.store.history(request.district, request.season or "Kharif")
        pests = list(PEST_BY_STAGE.get(stage, []))
        diseases = list(DISEASE_BY_STAGE.get(stage, []))

        score, rules = 0, []
        if str(hist.get("pest", "")).lower() == "high":
            score += 2; rules.append(f"{request.district} historically High pest incidence")
        elif str(hist.get("pest", "")).lower() == "medium":
            score += 1
        if humidity > 80:
            score += 2; rules.append(f"Humidity {humidity:.0f}% > 80%")
            for d in ["Blast", "Sheath Blight"]:
                if d not in diseases:
                    diseases.append(d)
        elif humidity > 70:
            score += 1; rules.append(f"Humidity {humidity:.0f}% > 70%")
        if temp > 32:
            score += 1; rules.append(f"Max temp {temp:.0f}C > 32C")
        if rainfall > 20:
            score += 1; rules.append(f"Rainfall {rainfall:.0f}mm favours spread")
        if stage in ("Flowering", "Grain_Filling"):
            score += 1; rules.append(f"{stage.replace('_',' ')} high-sensitivity window")

        level = "HIGH" if score >= 4 else "MEDIUM" if score >= 2 else "LOW"
        checkin = 2 if level == "HIGH" else 4 if level == "MEDIUM" else 7

        if stage == "Flowering":
            action = ("Do NOT spray insecticide during flowering (kills pollinators, causes sterility). "
                      "Scout daily; treat only after the flowering window closes.")
        elif level == "HIGH":
            parts = [f"{n}: {PEST_DISEASE_TREATMENT[n]}" for n in (pests[:1] + diseases[:1]) if n in PEST_DISEASE_TREATMENT]
            action = "Scout 10 hills/acre in a zigzag today. " + (" ".join(parts) if parts else "Confirm before spraying.")
        else:
            action = "Maintain weekly scouting. No prophylactic spray justified at this risk level."

        return pb.PestResponse(
            metadata=_meta(0.84), district=request.district, season=request.season or "Kharif",
            growth_stage=stage, risk_level=level, risk_score=score,
            likely_pests=pests[:3], likely_diseases=diseases[:3],
            preventive_action=action, next_checkin_days=checkin,
            rule_matched=" + ".join(rules) if rules else f"{request.season} + {stage} baseline")


def register(port):
    default_addr = "orchestrator:50051" if os.path.exists("/.dockerenv") else "localhost:50051"
    addr = os.environ.get("ORCHESTRATOR_ADDR", default_addr)
    host = os.environ.get("AGENT_ADVERTISE_HOST", "pest-agent" if os.path.exists("/.dockerenv") else "localhost")
    for i in range(3):
        try:
            with grpc.insecure_channel(addr) as ch:
                core_grpc.OrchestratorServiceStub(ch).RegisterAgent(
                    core_pb.AgentRegistration(agent_id="pest-agent-001", agent_name="Pest Risk Agent",
                                              agent_type="PEST_RISK", host=host, port=port), timeout=3)
                log.info("Registered with orchestrator at %s", addr)
                return
        except grpc.RpcError:
            if i < 2:
                time.sleep(2)
    log.info("No orchestrator at %s — serving standalone.", addr)


def serve():
    port = int(os.environ.get("PEST_AGENT_PORT", "50056"))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb_grpc.add_PestAgentServiceServicer_to_server(PestServicer(PestStore(DATA)), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("Pest Risk Agent running on :%d", port)
    threading.Thread(target=register, args=(port,), daemon=True).start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
