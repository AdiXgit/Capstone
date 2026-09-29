"""Weather Agent (Python).

Serves the WeatherAgentService contract. Fetches a live 7-day forecast from
Open-Meteo (free, no key) and falls back to the historical dataset
(Karnataka_Weather_Daily.csv) when the API is unreachable. Advisory and alert
logic is stage-aware via days_after_transplant.

This replaces the not-yet-built Go weather agent so the mesh has a live,
runnable weather service the orchestrator can call. Run:
    python python/agents/weather_agent/weather_agent.py     # listens on :50053
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
import requests
import grpc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
# Reuse the shared district coordinates from the crop-health package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../crop_health_agent"))

import paddy_agents_pb2 as pb
import paddy_agents_pb2_grpc as pb_grpc
from stage_profiles import DISTRICT_COORDS, KARNATAKA_CENTROID

logging.basicConfig(level=logging.INFO, format="%(asctime)s [WEATHER] %(levelname)s %(message)s")
log = logging.getLogger("WEATHER")

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../data/Karnataka_Weather_Daily.csv")


def _meta(conf=0.85, notes=""):
    return pb.AgentMetadata(
        agent_id="weather-agent-001", agent_name="Weather Advisory Agent",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        confidence=conf, notes=notes)


def _stage_from_dat(dat):
    if dat <= 0:
        return "Vegetative"
    if dat < 25:
        return "Vegetative"
    if dat < 50:
        return "Vegetative"
    if dat < 70:
        return "Flowering"
    if dat < 95:
        return "Grain_Filling"
    return "Maturity"


def _advisory(temp_max, rain, humidity, dat):
    stage = _stage_from_dat(dat)
    if rain > 50:
        return "Heavy rain — open drainage channels and delay fertiliser by 48h.", 0.90
    if rain > 20:
        return "Field moisture adequate — skip the next irrigation and recheck in 48h.", 0.88
    if temp_max > 38:
        return "Heat stress risk — maintain 5 cm standing water and irrigate early morning.", 0.86
    if humidity > 88:
        return "High humidity favours blast/sheath blight — scout and prep a preventive spray.", 0.85
    if stage == "Flowering":
        return "Flowering: keep the field continuously flooded at 5 cm; avoid any water stress.", 0.84
    if rain < 2:
        return "Little rain expected — schedule irrigation within 2 days to hold 3–5 cm water.", 0.82
    return "Good spraying window — apply scheduled foliar treatment in the early morning.", 0.80


class _WeatherStore:
    def __init__(self, path):
        try:
            self.df = pd.read_csv(path)
            self.df["Date"] = pd.to_datetime(self.df["Date"])
        except Exception as e:
            log.warning("Weather dataset not loaded: %s", e)
            self.df = None

    def latest(self, district):
        if self.df is None:
            return None
        d = self.df[self.df["District"] == district]
        if d.empty:
            d = self.df
        if d.empty:
            return None
        r = d.sort_values("Date").iloc[-1]
        return {
            "temp_max": float(r["Temperature_Max_C"]), "temp_min": float(r["Temperature_Min_C"]),
            "temp_avg": float(r["Temperature_Avg_C"]), "humidity": float(r["Humidity_Percent"]),
            "wind": float(r["Wind_Speed_kmph"]), "sunshine": float(r["Sunshine_Hours"]),
            "rain": float(r["Rainfall_mm"]),
        }


def _open_meteo(district):
    lat, lon = DISTRICT_COORDS.get(district, KARNATAKA_CENTROID)
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "Asia/Kolkata", "forecast_days": 7,
    }, timeout=8)
    r.raise_for_status()
    j = r.json()
    daily, cur = j.get("daily", {}), j.get("current", {})
    forecast = []
    for i, date in enumerate(daily.get("time", [])):
        forecast.append({
            "date": date, "temp_max": daily["temperature_2m_max"][i],
            "temp_min": daily["temperature_2m_min"][i],
            "rainfall_prob": daily.get("precipitation_probability_max", [0] * 7)[i] or 0,
        })
    return {
        "temp_avg": cur.get("temperature_2m"), "humidity": cur.get("relative_humidity_2m"),
        "wind": cur.get("wind_speed_10m"), "rain": cur.get("precipitation", 0.0),
        "temp_max": forecast[0]["temp_max"] if forecast else cur.get("temperature_2m"),
        "temp_min": forecast[0]["temp_min"] if forecast else cur.get("temperature_2m"),
        "forecast": forecast,
    }


class WeatherAgentServicer(pb_grpc.WeatherAgentServiceServicer):
    def __init__(self, store):
        self.store = store

    def _gather(self, district):
        """Live Open-Meteo first, dataset fallback second. Returns (data, source)."""
        try:
            w = _open_meteo(district)
            hist = self.store.latest(district) or {}
            w.setdefault("sunshine", hist.get("sunshine", 7.0))
            return w, "LIVE_OPEN_METEO"
        except Exception as e:
            log.warning("Open-Meteo failed (%s) — using dataset", e)
            h = self.store.latest(district)
            if not h:
                return None, "NONE"
            h["forecast"] = []
            return h, "DATASET"

    def GetWeatherForecast(self, request, context):
        log.info("GetWeatherForecast -> %s DAT=%d", request.district, request.days_after_transplant)
        w, source = self._gather(request.district)
        if w is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("No weather data for district")
            return pb.WeatherResponse()

        rain = w.get("rain") or 0.0
        temp_max = w.get("temp_max") or w.get("temp_avg") or 0.0
        humidity = w.get("humidity") or 0.0
        advisory, conf = _advisory(temp_max, rain, humidity, request.days_after_transplant)
        condition = ("Heavy Rain" if rain > 20 else "Light Rain" if rain > 2
                     else "Hot" if temp_max > 35 else "Partly Cloudy")

        forecast = [pb.DailyForecast(
            date=f["date"], temp_max=float(f["temp_max"]), temp_min=float(f["temp_min"]),
            rainfall_prob=float(f.get("rainfall_prob", 0)), condition="") for f in w.get("forecast", [])]

        return pb.WeatherResponse(
            metadata=_meta(conf, notes=f"source={source}; DAT={request.days_after_transplant}"),
            district=request.district,
            temperature_max=float(temp_max), temperature_min=float(w.get("temp_min") or 0.0),
            temperature_avg=float(w.get("temp_avg") or 0.0), rainfall_mm=float(rain),
            humidity_percent=float(humidity), wind_speed_kmph=float(w.get("wind") or 0.0),
            sunshine_hours=float(w.get("sunshine") or 0.0),
            weather_condition=condition, farming_advisory=advisory, forecast=forecast)

    def GetWeatherAlert(self, request, context):
        w, source = self._gather(request.district)
        if w is None:
            return pb.WeatherAlertResponse(metadata=_meta(0.5), has_alert=False)
        rain = w.get("rain") or 0.0
        temp_max = w.get("temp_max") or 0.0
        humidity = w.get("humidity") or 0.0
        if rain > 50:
            a = ("FLOOD_RISK", "HIGH", f"{rain:.0f} mm rain — flooding risk.",
                 "Open drainage channels; delay fertiliser 48h.")
        elif temp_max > 38:
            a = ("HEAT_STRESS", "MEDIUM", f"Max temp {temp_max:.0f}°C — spikelet sterility risk.",
                 "Maintain 5 cm water; irrigate early morning.")
        elif humidity > 88:
            a = ("DISEASE_RISK", "MEDIUM", f"Humidity {humidity:.0f}% — blast/sheath blight risk.",
                 "Scout for lesions; prepare preventive fungicide.")
        else:
            return pb.WeatherAlertResponse(metadata=_meta(0.8, f"source={source}"), has_alert=False)
        return pb.WeatherAlertResponse(
            metadata=_meta(0.85, f"source={source}"), has_alert=True,
            alert_type=a[0], severity=a[1], message=a[2], action_required=a[3])


def register_with_orchestrator(port, max_retries=3):
    default_addr = "orchestrator:50051" if os.path.exists("/.dockerenv") else "localhost:50051"
    addr = os.environ.get("ORCHESTRATOR_ADDR", default_addr)
    host = os.environ.get("AGENT_ADVERTISE_HOST", "weather-agent" if os.path.exists("/.dockerenv") else "localhost")
    for i in range(max_retries):
        try:
            with grpc.insecure_channel(addr) as ch:
                pb_grpc.OrchestratorServiceStub(ch).RegisterAgent(
                    pb.AgentRegistration(agent_id="weather-agent-001",
                                         agent_name="Weather Advisory Agent",
                                         agent_type="WEATHER", host=host, port=port), timeout=3)
                log.info("Registered with orchestrator at %s", addr)
                return
        except grpc.RpcError:
            if i < max_retries - 1:
                time.sleep(2)
    log.info("No orchestrator at %s — serving standalone.", addr)


def serve():
    port = int(os.environ.get("WEATHER_AGENT_PORT", "50053"))
    store = _WeatherStore(DATA_PATH)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb_grpc.add_WeatherAgentServiceServicer_to_server(WeatherAgentServicer(store), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("Weather Agent running on :%d", port)
    threading.Thread(target=register_with_orchestrator, args=(port,), daemon=True).start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
