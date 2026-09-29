"""KrishiSense Orchestrator (Python).

This is the cross-agent brain of the mesh. Agents register themselves here on
startup; when a FarmerQuery arrives it fans out CONCURRENTLY to every relevant
agent over gRPC, collects their structured results, and synthesizes a single
human-readable advisory that combines what each agent said.

It implements the same OrchestratorService contract as the Go orchestrator in
go/orchestrator/main.go, but runs on the existing generated Python stubs so the
whole mesh can be brought up with `pip` alone (no protoc / Go toolchain needed).

Run:
    python python/orchestrator/orchestrator_server.py     # listens on :50051
"""
import os
import sys
import json
import time
import logging
import threading
from concurrent import futures

import grpc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../shared"))

import paddy_agents_pb2 as pb
import paddy_agents_pb2_grpc as pb_grpc
import market_pest_pb2 as mp
import market_pest_pb2_grpc as mp_grpc

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [ORCHESTRATOR] %(levelname)s %(message)s")
log = logging.getLogger("ORCHESTRATOR")

# Fallback addresses used when an agent has not registered yet. Ports match
# docker-compose.yml (soil 50052, weather 50053, crop-health 50054).
DEFAULT_AGENTS = {
    "SOIL":         ("localhost", 50052),
    "WEATHER":      ("localhost", 50053),
    "CROP_HEALTH":  ("localhost", 50054),
    "MARKET_PRICE": ("localhost", 50055),
    "PEST_RISK":    ("localhost", 50056),
}

# Which agents each query type consults.
ROUTING = {
    "WEATHER":     ["WEATHER"],
    "SOIL":        ["SOIL"],
    "CROP_HEALTH": ["CROP_HEALTH", "WEATHER"],
    "FERTILIZER":  ["SOIL", "WEATHER", "CROP_HEALTH"],
    "IRRIGATION":  ["WEATHER", "SOIL"],
    "MARKET":      ["MARKET_PRICE"],
    "PEST":        ["PEST_RISK", "WEATHER"],
    "FULL":        ["WEATHER", "SOIL", "CROP_HEALTH", "MARKET_PRICE", "PEST_RISK"],
}


class OrchestratorServicer(pb_grpc.OrchestratorServiceServicer):
    def __init__(self):
        self._agents = {}      # agent_id -> {name, type, host, port}
        self._lock = threading.RLock()
        self._query_count = 0

    # ── registration / status ────────────────────────────────────

    def RegisterAgent(self, request, context):
        with self._lock:
            self._agents[request.agent_id] = {
                "name": request.agent_name, "type": request.agent_type,
                "host": request.host, "port": request.port,
            }
        log.info("Registered %s (%s) at %s:%d",
                 request.agent_name, request.agent_type, request.host, request.port)
        return pb.RegistrationAck(success=True, message="Registered successfully")

    def GetSystemStatus(self, request, context):
        with self._lock:
            names = [a["name"] for a in self._agents.values()]
            count = self._query_count
        return pb.SystemStatusResponse(
            orchestrator_status="HEALTHY", active_agents=names, total_queries_handled=count)

    # ── routing ──────────────────────────────────────────────────

    def _addr_for(self, agent_type):
        with self._lock:
            for a in self._agents.values():
                if a["type"] == agent_type:
                    return a["host"], a["port"]
        host = os.environ.get(f"{agent_type}_AGENT_HOST")
        default_host, default_port = DEFAULT_AGENTS.get(agent_type, ("localhost", 50060))
        return (host or default_host), default_port

    def RouteQuery(self, request, context):
        with self._lock:
            self._query_count += 1
        targets = ROUTING.get(request.query_type, ["WEATHER", "SOIL", "CROP_HEALTH"])
        log.info("RouteQuery %s type=%s district=%s DAT=%d → %s",
                 request.query_id, request.query_type, request.district,
                 request.days_after_transplant, targets)

        results = [None] * len(targets)

        def work(i, atype):
            results[i] = self._call_agent(atype, request)

        threads = [threading.Thread(target=work, args=(i, a)) for i, a in enumerate(targets)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        agent_results = [r for r in results if r is not None]
        confs = [r.confidence for r in agent_results if r.status == "OK"]
        overall = sum(confs) / len(confs) if confs else 0.0

        recommendation = self._synthesize(agent_results, request)
        return pb.OrchestratorResponse(
            query_id=request.query_id,
            recommendation=recommendation,
            agent_results=agent_results,
            overall_confidence=overall,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    # ── per-agent gRPC calls ─────────────────────────────────────

    def _call_agent(self, agent_type, req):
        host, port = self._addr_for(agent_type)
        addr = f"{host}:{port}"
        try:
            with grpc.insecure_channel(addr) as ch:
                if agent_type == "WEATHER":
                    return self._call_weather(ch, req)
                if agent_type == "SOIL":
                    return self._call_soil(ch, req)
                if agent_type == "CROP_HEALTH":
                    return self._call_crop_health(ch, req)
                if agent_type == "MARKET_PRICE":
                    return self._call_market(ch, req)
                if agent_type == "PEST_RISK":
                    return self._call_pest(ch, req)
        except grpc.RpcError as e:
            log.warning("%s agent call failed (%s): %s", agent_type, addr, e.details())
            return pb.AgentResult(agent_name=f"{agent_type} Agent",
                                  result_json=json.dumps({"error": e.details(), "addr": addr}),
                                  confidence=0.0, status="ERROR")
        return pb.AgentResult(agent_name=f"{agent_type} Agent", status="UNKNOWN_AGENT")

    def _call_weather(self, ch, req):
        stub = pb_grpc.WeatherAgentServiceStub(ch)
        resp = stub.GetWeatherForecast(pb.WeatherRequest(
            district=req.district, forecast_days=7,
            days_after_transplant=req.days_after_transplant), timeout=10)
        payload = {
            "district": resp.district,
            "temperature_avg": round(resp.temperature_avg, 1),
            "temperature_max": round(resp.temperature_max, 1),
            "rainfall_mm": round(resp.rainfall_mm, 1),
            "humidity_percent": round(resp.humidity_percent, 1),
            "weather_condition": resp.weather_condition,
            "farming_advisory": resp.farming_advisory,
            "metadata": {"notes": resp.metadata.notes},
        }
        conf = resp.metadata.confidence or 0.88
        return pb.AgentResult(agent_name="Weather Agent", result_json=json.dumps(payload),
                              confidence=conf, status="OK")

    def _call_soil(self, ch, req):
        stub = pb_grpc.SoilAgentServiceStub(ch)
        resp = stub.GetSoilHealth(pb.SoilRequest(district=req.district, season=req.season), timeout=10)
        payload = {
            "district": resp.district,
            "soil_health_score": round(resp.soil_health_score, 2),
            "recommendation": resp.recommendation,
            "nitrogen": round(resp.nitrogen, 1),
            "phosphorus": round(resp.phosphorus, 1),
            "potassium": round(resp.potassium, 1),
            "ph": round(resp.ph, 2),
            "organic_carbon": round(resp.organic_carbon, 3),
        }
        conf = resp.metadata.confidence or 0.85
        return pb.AgentResult(agent_name="Soil Agent", result_json=json.dumps(payload),
                              confidence=conf, status="OK")

    def _call_crop_health(self, ch, req):
        stub = pb_grpc.CropHealthAgentServiceStub(ch)
        resp = stub.GetCropHealthStatus(pb.CropHealthRequest(
            district=req.district, season=req.season,
            year=time.gmtime().tm_year, days_after_sowing=req.days_after_transplant), timeout=10)
        payload = {
            "district": resp.district,
            "crop_status": resp.crop_status,
            "disease_risk": resp.disease_risk,
            "growth_stage": resp.growth_stage,
            "stress_level": resp.stress_level,
            "ndvi": round(resp.ndvi, 3),
            "ndvi_deviation": round(resp.ndvi_deviation, 3),
            "treatment": resp.treatment,
            "recommendation": resp.recommendation,
        }
        conf = resp.metadata.confidence or resp.confidence or 0.82
        return pb.AgentResult(agent_name="Crop Health Agent", result_json=json.dumps(payload),
                              confidence=conf, status="OK")

    def _call_market(self, ch, req):
        stub = mp_grpc.MarketAgentServiceStub(ch)
        resp = stub.GetMarketAdvisory(mp.MarketRequest(district=req.district, season=req.season), timeout=10)
        payload = {
            "district": resp.district, "crop_variety": resp.crop_variety,
            "current_price_per_quintal": round(resp.current_price_per_quintal, 0),
            "sell_or_hold": resp.sell_or_hold, "trend_direction": resp.trend_direction,
            "reasoning": resp.reasoning,
        }
        return pb.AgentResult(agent_name="Market Price Agent", result_json=json.dumps(payload),
                              confidence=resp.metadata.confidence or 0.86, status="OK")

    def _call_pest(self, ch, req):
        stub = mp_grpc.PestAgentServiceStub(ch)
        stage = "Flowering" if 50 <= req.days_after_transplant <= 70 else "Vegetative"
        resp = stub.GetPestRisk(mp.PestRequest(district=req.district, season=req.season,
                                               growth_stage=stage), timeout=10)
        payload = {
            "district": resp.district, "risk_level": resp.risk_level,
            "likely_pests": list(resp.likely_pests), "likely_diseases": list(resp.likely_diseases),
            "preventive_action": resp.preventive_action, "recommendation": resp.preventive_action,
        }
        return pb.AgentResult(agent_name="Pest Risk Agent", result_json=json.dumps(payload),
                              confidence=resp.metadata.confidence or 0.84, status="OK")

    # ── synthesis ────────────────────────────────────────────────

    def _synthesize(self, results, req):
        """Merge the agents' findings into one advisory. The value of the mesh is
        here: each agent contributes a slice, the orchestrator weaves them into a
        single, prioritized action list."""
        by_agent = {}
        for r in results:
            if r.status == "OK":
                try:
                    by_agent[r.agent_name] = json.loads(r.result_json)
                except json.JSONDecodeError:
                    pass

        lines = [f"Advisory for {req.district} · {req.season} · DAT {req.days_after_transplant}"]

        weather = by_agent.get("Weather Agent")
        crop = by_agent.get("Crop Health Agent")
        soil = by_agent.get("Soil Agent")
        market = by_agent.get("Market Price Agent")
        pest = by_agent.get("Pest Risk Agent")

        # Prioritised, cross-agent synthesis
        if weather:
            cond = weather.get("weather_condition", "")
            rain = weather.get("rainfall_mm", 0)
            adv = weather.get("farming_advisory", "")
            lines.append(f"• Weather: {cond}, {rain} mm rain. {adv}")
            note = weather.get("metadata", {}).get("notes", "")
            if note:
                lines.append(f"  ↳ {note}")

        if crop:
            stage = str(crop.get("growth_stage", "")).replace("_", " ")
            lines.append(
                f"• Crop health: {crop.get('crop_status')} — {crop.get('stress_level')} stress at "
                f"{stage} (NDVI {crop.get('ndvi')}, {crop.get('ndvi_deviation'):+} vs stage median). "
                f"Disease risk {crop.get('disease_risk')}.")
            if crop.get("treatment"):
                lines.append(f"  ↳ {crop['treatment']}")

        if soil:
            lines.append(
                f"• Soil: health {soil.get('soil_health_score')}/10 "
                f"(N {soil.get('nitrogen')}, P {soil.get('phosphorus')}, K {soil.get('potassium')}, "
                f"pH {soil.get('ph')}). {soil.get('recommendation', '')}")

        if pest:
            lines.append(
                f"• Pest risk: {pest.get('risk_level')} — likely {', '.join(pest.get('likely_pests', [])[:2]) or 'none'}. "
                f"{pest.get('preventive_action', '')}")

        if market:
            lines.append(
                f"• Market: {market.get('crop_variety')} at Rs{market.get('current_price_per_quintal')}/qtl — "
                f"{market.get('sell_or_hold')} ({market.get('trend_direction', '').lower()}).")

        # Cross-agent rule: heavy rain + flowering/grain-fill + disease risk → escalate
        if weather and crop:
            rain = weather.get("rainfall_mm", 0) or 0
            stage = crop.get("growth_stage", "")
            if rain > 20 and crop.get("disease_risk") in ("HIGH", "MEDIUM"):
                lines.append(
                    "⚠ Combined signal: recent rain plus elevated disease risk — hold off on foliar "
                    "sprays until the canopy dries, open drainage, and re-scout in 48 hours.")
            if stage == "Flowering":
                lines.append(
                    "⚠ Flowering window: do NOT spray insecticide (pollinator + sterility risk); "
                    "keep the field flooded at ~5 cm.")

        ok = [r for r in results if r.status == "OK"]
        bad = [r for r in results if r.status != "OK"]
        if bad:
            lines.append("Unavailable: " + ", ".join(f"{r.agent_name} ({r.status})" for r in bad))
        conf = (sum(r.confidence for r in ok) / len(ok) * 100) if ok else 0
        lines.append(f"Consulted {len(ok)}/{len(results)} agents · overall confidence {conf:.0f}%")
        return "\n".join(lines)


def serve():
    port = int(os.environ.get("ORCHESTRATOR_PORT", "50051"))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20),
                         options=[("grpc.max_receive_message_length", 16 * 1024 * 1024),
                                  ("grpc.max_send_message_length", 16 * 1024 * 1024)])
    pb_grpc.add_OrchestratorServiceServicer_to_server(OrchestratorServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("Orchestrator gRPC server listening on :%d", port)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
