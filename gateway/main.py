"""KrishiSense REST gateway.

Browsers cannot speak gRPC, so this FastAPI service sits in front of the agent
mesh and exposes plain JSON to the React dashboard.

For each domain it tries the real gRPC agent first; if that agent is not
running it falls back to computing the same answer locally from the datasets in
data/. Every response carries a `_meta` block saying which path served it, so
the dashboard can be honest about where a number came from.
"""
import os
import sys
import time
import socket
import logging
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)
DATA_DIR = os.path.join(REPO, "data")
sys.path.insert(0, os.path.join(REPO, "python", "shared"))
sys.path.insert(0, os.path.join(REPO, "python", "agents", "crop_health_agent"))

import grpc
import paddy_agents_pb2 as pb
import paddy_agents_pb2_grpc as pb_grpc

from stage_profiles import (
    STAGE_ORDER, STAGE_PROFILES, DISTRICT_COORDS, KARNATAKA_CENTROID,
    das_from_season, das_to_stage, stage_ndvi_median,
)
from treatment_map import (
    get_treatment, crop_status_from_stress, STAGE_ADVISORIES,
    PEST_BY_STAGE, DISEASE_BY_STAGE, PEST_DISEASE_TREATMENT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [GATEWAY] %(levelname)s %(message)s")
log = logging.getLogger("gateway")

AGENTS = {
    "ORCHESTRATOR": int(os.environ.get("ORCHESTRATOR_PORT", 50051)),
    "SOIL":         int(os.environ.get("SOIL_AGENT_PORT", 50052)),
    "WEATHER":      int(os.environ.get("WEATHER_AGENT_PORT", 50053)),
    "CROP_HEALTH":  int(os.environ.get("CROP_HEALTH_AGENT_PORT", 50054)),
    "MARKET_PRICE": int(os.environ.get("MARKET_AGENT_PORT", 50055)),
    "PEST_RISK":    int(os.environ.get("PEST_AGENT_PORT", 50056)),
}
AGENT_HOST = os.environ.get("AGENT_HOST", "localhost")

app = FastAPI(title="KrishiSense Gateway", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────
# DATASETS
# ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def main_df():
    return pd.read_csv(os.path.join(DATA_DIR, "Karnataka_Paddy_Main_Dataset.csv"))


@lru_cache(maxsize=1)
def weather_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "Karnataka_Weather_Daily.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@lru_cache(maxsize=1)
def market_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "Karnataka_Market_Prices.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@lru_cache(maxsize=1)
def districts():
    return sorted(main_df()["District"].dropna().unique().tolist())


def meta(source, started, agent, notes=""):
    return {
        "source": source,          # GRPC_AGENT | GATEWAY_DATASET | LIVE_API
        "agent": agent,
        "latency_ms": round((time.time() - started) * 1000, 1),
        "notes": notes,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def channel(agent_key):
    return grpc.insecure_channel(f"{AGENT_HOST}:{AGENTS[agent_key]}")


def agent_online(agent_key, timeout=0.35):
    """Liveness probe.

    A plain TCP connect rather than grpc.channel_ready_future — the latter is
    unreliable when called from uvicorn's worker threads and reports a false
    OFFLINE for agents that are demonstrably up.
    """
    started = time.time()
    try:
        with socket.create_connection((AGENT_HOST, AGENTS[agent_key]), timeout=timeout):
            return True, round((time.time() - started) * 1000, 1)
    except OSError:
        return False, round((time.time() - started) * 1000, 1)


# ────────────────────────────────────────────────────────────────
# META
# ────────────────────────────────────────────────────────────────

@app.get("/api/meta")
def get_meta():
    return {
        "districts": districts(),
        "seasons": ["Kharif", "Rabi"],
        "growth_stages": STAGE_ORDER,
        "stage_baselines": {
            s: {
                "q25": STAGE_PROFILES[s]["ndvi"][0],
                "median": STAGE_PROFILES[s]["ndvi"][1],
                "q75": STAGE_PROFILES[s]["ndvi"][2],
                "das": list(STAGE_PROFILES[s]["das"]),
            }
            for s in STAGE_ORDER
        },
        "stage_advisories": STAGE_ADVISORIES,
    }


# ────────────────────────────────────────────────────────────────
# SOIL
# ────────────────────────────────────────────────────────────────

def _soil_payload(district, season):
    df = main_df()
    d = df[df["District"] == district]
    if d.empty:
        d = df
    ds = d[d["Season"] == season]
    if ds.empty:
        ds = d

    n = float(d["Soil_Nitrogen_kg_per_ha"].mean())
    p = float(d["Soil_Phosphorus_kg_per_ha"].mean())
    k = float(d["Soil_Potassium_kg_per_ha"].mean())
    ph = float(d["Soil_pH"].mean())
    oc = float(d["Soil_Organic_Carbon_Percent"].mean())

    yearly = d.groupby("Numeric_Year")[["Soil_pH", "Soil_Organic_Carbon_Percent"]].mean().reset_index()

    def slope(x, y):
        if len(x) < 2:
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    ph_slope = slope(yearly["Numeric_Year"], yearly["Soil_pH"])
    oc_slope = slope(yearly["Numeric_Year"], yearly["Soil_Organic_Carbon_Percent"])

    # 0-10 health score from agronomic optima for paddy
    ph_score = max(0.0, 1 - abs(ph - 6.5) / 2.0)
    oc_score = min(1.0, oc / 0.75)
    n_score = min(1.0, n / 280.0)
    p_score = min(1.0, p / 25.0)
    k_score = min(1.0, k / 200.0)
    score = round((ph_score * 2.5 + oc_score * 2.5 + n_score * 2 + p_score * 1.5 + k_score * 1.5), 2)

    quality = str(d["Soil_Quality"].mode()[0]) if "Soil_Quality" in d else "medium"
    kvk = {"high": (100, 50, 30), "medium": (130, 65, 40), "low": (160, 80, 50)}
    urea, dap, potash = kvk.get(quality.lower(), kvk["medium"])
    if season == "Rabi":
        urea, dap, potash = round(urea * 0.85), round(dap * 0.90), round(potash * 0.90)
        timing = ("Apply 40% Urea basal at sowing, 30% at tillering, 30% at panicle initiation. "
                  "Full DAP and Potash as basal.")
    else:
        timing = ("Apply 50% Urea basal at transplanting, remaining 50% at active tillering (25–30 DAT). "
                  "Full DAP and Potash as basal.")

    interp = []
    interp.append("Organic carbon declining — apply 2–3 t/ha green manure (Dhaincha) before transplanting."
                  if oc_slope < -0.001 else "Organic carbon stable — maintain current residue practice.")
    interp.append("pH drifting acidic — apply 2 t/ha lime before the season."
                  if ph_slope < -0.01 else ("pH drifting alkaline — check irrigation water quality."
                                             if ph_slope > 0.01 else "pH within the optimal 6.0–7.0 band for paddy."))

    trend_series = [
        {"year": int(r["Numeric_Year"]), "ph": round(float(r["Soil_pH"]), 3),
         "organic_carbon": round(float(r["Soil_Organic_Carbon_Percent"]), 3)}
        for _, r in yearly.iterrows()
    ]

    district_comparison = (
        df.groupby("District")[["Soil_Nitrogen_kg_per_ha", "Soil_Phosphorus_kg_per_ha", "Soil_Potassium_kg_per_ha"]]
        .mean().round(1).reset_index()
        .rename(columns={"Soil_Nitrogen_kg_per_ha": "nitrogen",
                          "Soil_Phosphorus_kg_per_ha": "phosphorus",
                          "Soil_Potassium_kg_per_ha": "potassium",
                          "District": "district"})
        .to_dict("records")
    )

    return {
        "district": district,
        "season": season,
        "nitrogen": round(n, 1),
        "phosphorus": round(p, 1),
        "potassium": round(k, 1),
        "ph": round(ph, 2),
        "organic_carbon": round(oc, 3),
        "soil_health_score": min(10.0, score),
        "soil_quality": quality,
        "urea_kg_per_acre": urea,
        "dap_kg_per_acre": dap,
        "potash_kg_per_acre": potash,
        "timing_advice": timing,
        "ph_slope": round(ph_slope, 4),
        "ph_last_value": round(float(yearly["Soil_pH"].iloc[-1]), 2) if len(yearly) else round(ph, 2),
        "oc_slope": round(oc_slope, 4),
        "oc_last_value": round(float(yearly["Soil_Organic_Carbon_Percent"].iloc[-1]), 3) if len(yearly) else round(oc, 3),
        "interpretation": " ".join(interp),
        "trend_series": trend_series,
        "district_comparison": district_comparison,
        "avg_yield_kg_per_ha": round(float(ds["Yield_Kg_per_Ha"].mean()), 1),
    }


def _soil_impl(district: str, season: str = "Kharif"):
    started = time.time()
    online, _ = agent_online("SOIL")
    if online:
        try:
            with channel("SOIL") as ch:
                stub = pb_grpc.SoilAgentServiceStub(ch)
                resp = stub.GetNPKProfile(pb.DistrictRequest(district=district), timeout=3)
                payload = _soil_payload(district, season)
                payload.update({
                    "nitrogen": round(resp.nitrogen, 1),
                    "phosphorus": round(resp.phosphorus, 1),
                    "potassium": round(resp.potassium, 1),
                })
                return {**payload, "_meta": meta("GRPC_AGENT", started, "SOIL")}
        except Exception as e:
            log.warning("Soil agent call failed, using dataset: %s", e)
    return {**_soil_payload(district, season),
            "_meta": meta("GATEWAY_DATASET", started, "SOIL",
                           "Soil agent offline — computed from Karnataka_Paddy_Main_Dataset.csv")}


# ────────────────────────────────────────────────────────────────
# WEATHER
# ────────────────────────────────────────────────────────────────

def _open_meteo_forecast(district):
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
            "date": date,
            "temp_max": daily["temperature_2m_max"][i],
            "temp_min": daily["temperature_2m_min"][i],
            "rainfall_mm": daily["precipitation_sum"][i],
            "rainfall_prob": daily.get("precipitation_probability_max", [0] * 7)[i] or 0,
        })
    return {
        "temperature_avg": cur.get("temperature_2m"),
        "humidity_percent": cur.get("relative_humidity_2m"),
        "wind_speed_kmph": cur.get("wind_speed_10m"),
        "rainfall_mm": cur.get("precipitation", 0.0),
        "forecast": forecast,
    }


def _weather_dataset(district):
    df = weather_df()
    d = df[df["District"] == district]
    if d.empty:
        d = df
    recent = d.sort_values("Date").tail(7)
    last = recent.iloc[-1]
    forecast = [{
        "date": r["Date"].strftime("%Y-%m-%d"),
        "temp_max": float(r["Temperature_Max_C"]),
        "temp_min": float(r["Temperature_Min_C"]),
        "rainfall_mm": float(r["Rainfall_mm"]),
        "rainfall_prob": min(100, round(float(r["Rainfall_mm"]) * 6)),
    } for _, r in recent.iterrows()]
    return {
        "temperature_avg": float(last["Temperature_Avg_C"]),
        "temperature_max": float(last["Temperature_Max_C"]),
        "temperature_min": float(last["Temperature_Min_C"]),
        "humidity_percent": float(last["Humidity_Percent"]),
        "wind_speed_kmph": float(last["Wind_Speed_kmph"]),
        "sunshine_hours": float(last["Sunshine_Hours"]),
        "rainfall_mm": float(last["Rainfall_mm"]),
        "forecast": forecast,
    }


def _weather_alert(w, stage):
    rain = w.get("rainfall_mm") or 0
    temp = w.get("temperature_max") or w.get("temperature_avg") or 0
    hum = w.get("humidity_percent") or 0
    if rain > 50:
        return {"has_alert": True, "alert_type": "FLOOD_RISK", "severity": "HIGH",
                "message": f"{rain:.0f} mm rainfall recorded — flooding risk for {stage.replace('_',' ')} stage paddy.",
                "action_required": "Open drainage channels immediately and delay all fertiliser application by 48 hours."}
    if rain > 20:
        return {"has_alert": True, "alert_type": "SKIP_IRRIGATION", "severity": "LOW",
                "message": f"{rain:.0f} mm rainfall recorded — field moisture already adequate.",
                "action_required": "Skip the next scheduled irrigation and re-check soil moisture in 48 hours."}
    if temp > 38:
        return {"has_alert": True, "alert_type": "HEAT_STRESS", "severity": "MEDIUM",
                "message": f"Maximum temperature {temp:.1f}°C — spikelet sterility risk at flowering.",
                "action_required": "Maintain 5 cm standing water and irrigate in the early morning."}
    if hum > 88:
        return {"has_alert": True, "alert_type": "DISEASE_RISK", "severity": "MEDIUM",
                "message": f"Relative humidity {hum:.0f}% — conditions favour blast and sheath blight.",
                "action_required": "Scout for lesions and prepare a preventive fungicide spray."}
    if stage == "Flowering":
        return {"has_alert": True, "alert_type": "FLOWERING_FLOOD_REQUIRED", "severity": "LOW",
                "message": "Flowering stage — the crop is most sensitive to water stress right now.",
                "action_required": "Keep the field continuously flooded at 5 cm until grain filling begins."}
    if rain < 2:
        return {"has_alert": True, "alert_type": "IRRIGATION_NEEDED", "severity": "MEDIUM",
                "message": "No meaningful rainfall recorded and none forecast in the next 48 hours.",
                "action_required": "Schedule irrigation within 2 days to hold 3–5 cm standing water."}
    return {"has_alert": False, "alert_type": "", "severity": "", "message": "", "action_required": ""}


def _weather_impl(district: str, season: str = "Kharif", stage: str = "Vegetative"):
    started = time.time()
    online, _ = agent_online("WEATHER")
    if online:
        try:
            with channel("WEATHER") as ch:
                stub = pb_grpc.WeatherAgentServiceStub(ch)
                resp = stub.GetWeatherForecast(pb.WeatherRequest(district=district, forecast_days=7), timeout=5)
                payload = {
                    "district": district,
                    "temperature_avg": resp.temperature_avg,
                    "temperature_max": resp.temperature_max,
                    "temperature_min": resp.temperature_min,
                    "rainfall_mm": resp.rainfall_mm,
                    "humidity_percent": resp.humidity_percent,
                    "wind_speed_kmph": resp.wind_speed_kmph,
                    "sunshine_hours": resp.sunshine_hours,
                    "weather_condition": resp.weather_condition,
                    "farming_advisory": resp.farming_advisory,
                    "forecast": [{"date": f.date, "temp_max": f.temp_max, "temp_min": f.temp_min,
                                   "rainfall_prob": f.rainfall_prob, "rainfall_mm": 0.0} for f in resp.forecast],
                }
                payload["alert"] = _weather_alert(payload, stage)
                return {**payload, "_meta": meta("GRPC_AGENT", started, "WEATHER")}
        except Exception as e:
            log.warning("Weather agent call failed: %s", e)

    source, notes = "LIVE_API", "Weather agent offline — called Open-Meteo directly."
    try:
        w = _open_meteo_forecast(district)
        hist = _weather_dataset(district)
        w.setdefault("sunshine_hours", hist["sunshine_hours"])
        w["temperature_max"] = w["forecast"][0]["temp_max"] if w["forecast"] else hist["temperature_max"]
        w["temperature_min"] = w["forecast"][0]["temp_min"] if w["forecast"] else hist["temperature_min"]
    except Exception as e:
        log.warning("Open-Meteo failed, using dataset: %s", e)
        w = _weather_dataset(district)
        source, notes = "GATEWAY_DATASET", "Weather agent and Open-Meteo unavailable — using Karnataka_Weather_Daily.csv."

    rain = w.get("rainfall_mm") or 0
    temp = w.get("temperature_avg") or 0
    condition = ("Heavy Rain" if rain > 20 else "Light Rain" if rain > 2
                 else "Hot" if temp > 35 else "Partly Cloudy")
    advisory = ("Hold off on spraying — rainfall will wash off any foliar application."
                if rain > 10 else
                "Good spraying window. Apply any scheduled foliar treatment in the early morning.")

    df = weather_df()
    d = df[df["District"] == district]
    hist_series = []
    if not d.empty:
        yearly = d.assign(Year=d["Date"].dt.year).groupby("Year").agg(
            rainfall=("Rainfall_mm", "sum"), temp=("Temperature_Avg_C", "mean"),
            humidity=("Humidity_Percent", "mean")).round(1).reset_index()
        hist_series = [{"year": int(r["Year"]), "rainfall": float(r["rainfall"]),
                         "temperature": float(r["temp"]), "humidity": float(r["humidity"])}
                        for _, r in yearly.iterrows()]

    payload = {
        "district": district,
        "temperature_avg": round(w.get("temperature_avg") or 0, 1),
        "temperature_max": round(w.get("temperature_max") or 0, 1),
        "temperature_min": round(w.get("temperature_min") or 0, 1),
        "rainfall_mm": round(rain, 1),
        "humidity_percent": round(w.get("humidity_percent") or 0, 1),
        "wind_speed_kmph": round(w.get("wind_speed_kmph") or 0, 1),
        "sunshine_hours": round(w.get("sunshine_hours") or 0, 1),
        "weather_condition": condition,
        "farming_advisory": advisory,
        "forecast": w.get("forecast", []),
        "historical": hist_series,
    }
    payload["alert"] = _weather_alert(payload, stage)
    return {**payload, "_meta": meta(source, started, "WEATHER", notes)}


# ────────────────────────────────────────────────────────────────
# CROP HEALTH
# ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def local_store():
    from data_store import CropHealthDataStore
    return CropHealthDataStore(
        os.path.join(DATA_DIR, "Karnataka_Paddy_AI_ML_Dataset.xlsx"),
        os.path.join(DATA_DIR, "Karnataka_Paddy_Main_Dataset.csv"),
    )


def _stage_curve():
    return [{"stage": s.replace("_", " "), "stage_key": s,
             "q25": STAGE_PROFILES[s]["ndvi"][0],
             "median": STAGE_PROFILES[s]["ndvi"][1],
             "q75": STAGE_PROFILES[s]["ndvi"][2]} for s in STAGE_ORDER]


def _crop_health_impl(district: str, season: str = "Kharif",
                       stage: str = "", ndvi=None, das: int = 0):
    started = time.time()
    online, _ = agent_online("CROP_HEALTH")

    # Live agent path — only when the caller has not overridden stage/NDVI by hand.
    if online and ndvi is None and not stage:
        try:
            with channel("CROP_HEALTH") as ch:
                stub = pb_grpc.CropHealthAgentServiceStub(ch)
                r = stub.GetCropHealthStatus(pb.CropHealthRequest(
                    district=district, season=season, year=datetime.now().year,
                    days_after_sowing=das), timeout=5)
                return {
                    "district": r.district, "crop_status": r.crop_status,
                    "disease_risk": r.disease_risk, "recommendation": r.recommendation,
                    "confidence": round(r.confidence, 3), "growth_stage": r.growth_stage,
                    "days_after_sowing": r.days_after_sowing, "ndvi": round(r.ndvi, 3),
                    "ndvi_deviation": round(r.ndvi_deviation, 3), "stress_level": r.stress_level,
                    "treatment": r.treatment, "ndvi_source": r.ndvi_source,
                    "stage_curve": _stage_curve(),
                    "stage_baseline": stage_ndvi_median(r.growth_stage),
                    "_meta": meta("GRPC_AGENT", started, "CROP_HEALTH"),
                }
        except Exception as e:
            log.warning("Crop health agent call failed: %s", e)

    store = local_store()
    resolved_das = das or das_from_season(season)
    rec = store.get_closest_growth_record(district, season, resolved_das)
    resolved_stage = stage or (rec["growth_stage"] if rec else das_to_stage(resolved_das))
    prof = STAGE_PROFILES.get(resolved_stage, {})

    observed = store.get_stage_profile(district, season, resolved_stage) if stage else None

    if rec and not stage:
        lai, height, tillers = rec["leaf_area_index"], rec["plant_height_cm"], rec["tiller_count"]
        resolved_das = rec["days_after_sowing"]
        dataset_ndvi = rec["ndvi"]
    elif observed:
        # Stage picked by hand — use what this district actually recorded at that stage.
        lai, height, tillers = observed["lai_mean"], observed["height"], observed["tillers"]
        resolved_das = int(observed["das_mean"])
        dataset_ndvi = observed["ndvi_mean"]
    else:
        lai = float(np.mean(prof.get("lai", (2.0, 4.0))))
        height = float(np.mean(prof.get("height", (50, 90))))
        tillers = int(np.mean(prof.get("tillers", (10, 16))))
        dataset_ndvi = stage_ndvi_median(resolved_stage)

    used_ndvi = float(ndvi) if ndvi is not None else float(dataset_ndvi)
    ndvi_source = "MANUAL" if ndvi is not None else "DATASET"

    stress = store.predict_stress(used_ndvi, lai, resolved_das, height, tillers, resolved_stage)
    treatment = get_treatment(stress, resolved_stage)
    pest = store.get_pest_risk(district, season)

    return {
        "district": district,
        "crop_status": crop_status_from_stress(stress),
        "disease_risk": str(pest.get("disease", "MEDIUM")).upper(),
        "recommendation": treatment,
        "confidence": 0.9 if ndvi is None else 0.82,
        "growth_stage": resolved_stage,
        "days_after_sowing": int(resolved_das),
        "ndvi": round(used_ndvi, 3),
        "ndvi_deviation": round(used_ndvi - stage_ndvi_median(resolved_stage), 3),
        "stress_level": stress,
        "treatment": treatment,
        "ndvi_source": ndvi_source,
        "stage_curve": _stage_curve(),
        "stage_baseline": stage_ndvi_median(resolved_stage),
        "_meta": meta("GATEWAY_DATASET", started, "CROP_HEALTH",
                       "Scored in-process with the same Gradient Boosting model the agent serves."),
    }


@app.get("/api/crop-health/matrix")
def get_treatment_matrix():
    return {
        "stages": [{
            "stage": s,
            "label": s.replace("_", " "),
            "pests": PEST_BY_STAGE.get(s, []),
            "diseases": DISEASE_BY_STAGE.get(s, []),
            "advisory": STAGE_ADVISORIES.get(s, ""),
            "treatments": {lvl: get_treatment(lvl, s) for lvl in ["Healthy", "Mild", "Moderate", "Severe"]},
        } for s in STAGE_ORDER]
    }


# ────────────────────────────────────────────────────────────────
# MARKET PRICE
# ────────────────────────────────────────────────────────────────

MSP_2025 = 2320.0


def _market_impl(district: str, season: str = "Kharif", variety: str = ""):
    started = time.time()
    online, _ = agent_online("MARKET_PRICE")
    if online:
        notes = "Market agent reachable but GetMarketAdvisory is not implemented yet."
    else:
        notes = "Market agent not built yet — computed from Karnataka_Market_Prices.csv."

    df = market_df()
    varieties = sorted(df["Variety"].dropna().unique().tolist())
    chosen = variety if variety in varieties else varieties[0]
    d = df[df["Variety"] == chosen].sort_values("Date")

    recent = d.tail(30)
    current = float(recent["Price_Rs_per_Quintal"].iloc[-1])
    prev = float(recent["Price_Rs_per_Quintal"].iloc[0])
    slope = float(np.polyfit(range(len(recent)), recent["Price_Rs_per_Quintal"], 1)[0]) if len(recent) > 1 else 0.0

    direction = "RISING" if slope > 1.5 else "FALLING" if slope < -1.5 else "STABLE"
    sell = "HOLD" if direction == "RISING" and current < current * 1.05 else ("SELL" if current >= MSP_2025 else "HOLD")
    if direction == "FALLING" and current >= MSP_2025:
        sell = "SELL"

    monthly = d.assign(Month=d["Date"].dt.month).groupby("Month")["Price_Rs_per_Quintal"].mean()
    best_month = int(monthly.idxmax()) if len(monthly) else 1
    month_name = datetime(2025, best_month, 1).strftime("%B")

    reasoning = (
        f"{chosen} is trading at ₹{current:,.0f}/qtl, {'above' if current >= MSP_2025 else 'below'} the "
        f"₹{MSP_2025:,.0f} MSP. The 30-day trend is {direction.lower()} "
        f"({'+' if slope >= 0 else ''}{slope:.1f} ₹/day). Historically prices peak in {month_name}."
    )

    trend = [{"date": r["Date"].strftime("%Y-%m-%d"), "price": float(r["Price_Rs_per_Quintal"]),
              "demand": r["Market_Demand"], "supply": r["Supply_Status"]}
             for _, r in recent.iterrows()]

    variety_table = []
    for v in varieties:
        vd = df[df["Variety"] == v].sort_values("Date")
        if vd.empty:
            continue
        variety_table.append({
            "variety": v,
            "price": float(vd["Price_Rs_per_Quintal"].iloc[-1]),
            "demand": str(vd["Market_Demand"].iloc[-1]),
            "supply": str(vd["Supply_Status"].iloc[-1]),
        })

    return {
        "district": district,
        "crop_variety": chosen,
        "varieties": varieties,
        "current_price_per_quintal": round(current, 0),
        "msp_price_per_quintal": MSP_2025,
        "price_change_30d": round(current - prev, 0),
        "trend_direction": direction,
        "sell_or_hold": sell,
        "best_selling_month": month_name,
        "reasoning": reasoning,
        "price_trend_30d": trend,
        "variety_table": variety_table,
        "market_demand": str(recent["Market_Demand"].iloc[-1]),
        "supply_status": str(recent["Supply_Status"].iloc[-1]),
        "_meta": meta("GATEWAY_DATASET", started, "MARKET_PRICE", notes),
    }


# ────────────────────────────────────────────────────────────────
# PEST RISK
# ────────────────────────────────────────────────────────────────

def _pest_impl(district: str, season: str = "Kharif", stage: str = "Vegetative"):
    started = time.time()
    online, _ = agent_online("PEST_RISK")
    notes = ("Pest agent reachable but GetPestRisk is not implemented yet." if online
             else "Pest agent not built yet — KVK rules evaluated in the gateway.")

    try:
        w = _weather_impl(district, season, stage)
        temp = w.get("temperature_max", 30.0)
        humidity = w.get("humidity_percent", 70.0)
        rainfall = w.get("rainfall_mm", 0.0)
    except Exception:
        temp, humidity, rainfall = 30.0, 70.0, 0.0

    hist = local_store().get_pest_risk(district, season)
    pests = list(PEST_BY_STAGE.get(stage, []))
    diseases = list(DISEASE_BY_STAGE.get(stage, []))

    score, rules = 0, []
    if str(hist.get("pest", "")).lower() == "high":
        score += 2
        rules.append(f"{district} has a historically High pest incidence in {season}")
    elif str(hist.get("pest", "")).lower() == "medium":
        score += 1
    if humidity > 80:
        score += 2
        rules.append(f"Humidity {humidity:.0f}% > 80%")
        for d in ["Blast", "Sheath Blight"]:
            if d not in diseases:
                diseases.append(d)
    elif humidity > 70:
        score += 1
        rules.append(f"Humidity {humidity:.0f}% > 70%")
    if temp > 32:
        score += 1
        rules.append(f"Max temp {temp:.0f}°C > 32°C")
    if rainfall > 20:
        score += 1
        rules.append(f"Rainfall {rainfall:.0f} mm favours splash-borne spread")
    if stage in ("Flowering", "Grain_Filling"):
        score += 1
        rules.append(f"{stage.replace('_',' ')} is a high-value, high-sensitivity window")

    level = "HIGH" if score >= 4 else "MEDIUM" if score >= 2 else "LOW"
    checkin = 2 if level == "HIGH" else 4 if level == "MEDIUM" else 7

    actions = []
    for name in (pests[:2] + diseases[:2]):
        if name in PEST_DISEASE_TREATMENT:
            actions.append({"name": name, "treatment": PEST_DISEASE_TREATMENT[name]})
    if stage == "Flowering":
        preventive = ("Do NOT spray insecticide during flowering — it kills pollinators and causes sterility. "
                      "Scout daily and treat only after the flowering window closes.")
    elif level == "HIGH":
        preventive = ("Scout 10 hills per acre in a zigzag pattern today. Confirm the pest before spraying, "
                      "and consult your local KVK for the current resistance advisory.")
    else:
        preventive = "Maintain weekly scouting. No prophylactic spray is justified at this risk level."

    return {
        "district": district,
        "season": season,
        "growth_stage": stage,
        "risk_level": level,
        "risk_score": score,
        "likely_pests": pests[:3],
        "likely_diseases": diseases[:3],
        "pest_actions": actions,
        "preventive_action": preventive,
        "next_checkin_days": checkin,
        "rule_matched": " + ".join(rules) if rules else f"{season} + {stage.replace('_',' ')} baseline",
        "historical_pest_incidence": hist.get("pest"),
        "historical_disease_incidence": hist.get("disease"),
        "avg_sprays_per_season": round(float(hist.get("sprays", 4.0)), 1),
        "conditions": {"temperature_max": temp, "humidity_percent": humidity, "rainfall_mm": rainfall},
        "_meta": meta("GATEWAY_DATASET", started, "PEST_RISK", notes),
    }



# ── Route wrappers (kept separate so internal fan-in can call the impls) ──

@app.get("/api/soil")
def get_soil(district: str = Query(...), season: str = Query("Kharif")):
    return _soil_impl(district, season)


@app.get("/api/weather")
def get_weather(district: str = Query(...), season: str = Query("Kharif"),
                 stage: str = Query("Vegetative")):
    return _weather_impl(district, season, stage)


@app.get("/api/crop-health")
def get_crop_health(district: str = Query(...), season: str = Query("Kharif"),
                     stage: str = Query(""), ndvi: float = Query(None), das: int = Query(0)):
    return _crop_health_impl(district, season, stage, ndvi, das)


@app.get("/api/market")
def get_market(district: str = Query(...), season: str = Query("Kharif"), variety: str = Query("")):
    return _market_impl(district, season, variety)


@app.get("/api/pest-risk")
def get_pest_risk(district: str = Query(...), season: str = Query("Kharif"),
                   stage: str = Query("Vegetative")):
    return _pest_impl(district, season, stage)


# ────────────────────────────────────────────────────────────────
# SYSTEM STATUS + OVERVIEW
# ────────────────────────────────────────────────────────────────

AGENT_INFO = {
    "SOIL":         ("Soil Agent", "Python", "NPK profiling · KVK fertiliser · trend regression"),
    "WEATHER":      ("Weather Agent", "Go", "Open-Meteo live · 7-day forecast · stage-aware alerts"),
    "CROP_HEALTH":  ("Crop Health Agent", "Python", "NDVI deviation · Gradient Boosting · KVK treatment"),
    "MARKET_PRICE": ("Market Price Agent", "Python", "Price trend · sell/hold advisory"),
    "PEST_RISK":    ("Pest Risk Agent", "Go", "KVK rule engine · stage × weather risk"),
}


@app.get("/api/system/status")
def system_status():
    agents = []
    for key, (name, lang, desc) in AGENT_INFO.items():
        online, ms = agent_online(key)
        agents.append({
            "agent_type": key, "agent_name": name, "language": lang, "description": desc,
            "port": AGENTS[key], "status": "ONLINE" if online else "OFFLINE",
            "last_response_ms": ms,
            "served_by": "gRPC agent" if online else "gateway fallback",
        })
    orch_online, orch_ms = agent_online("ORCHESTRATOR")
    return {
        "orchestrator_status": "ONLINE" if orch_online else "OFFLINE",
        "orchestrator_port": AGENTS["ORCHESTRATOR"],
        "orchestrator_latency_ms": orch_ms,
        "active_agents": agents,
        "online_count": sum(1 for a in agents if a["status"] == "ONLINE"),
        "total_agents": len(agents),
        "gateway_note": ("Endpoints answer from the live gRPC agent when it is running, otherwise the "
                          "gateway computes the same result from the datasets in data/."),
    }


@app.get("/api/overview")
def overview(district: str = Query(...), season: str = Query("Kharif"), stage: str = Query("")):
    started = time.time()
    soil = _soil_impl(district, season)
    crop = _crop_health_impl(district, season, stage)
    resolved_stage = stage or crop["growth_stage"]
    weather = _weather_impl(district, season, resolved_stage)
    market = _market_impl(district, season)
    pest = _pest_impl(district, season, resolved_stage)

    advisory = []
    if weather["alert"]["has_alert"]:
        advisory.append({
            "agent": "Weather agent", "severity": weather["alert"]["severity"],
            "title": weather["alert"]["alert_type"].replace("_", " "),
            "body": weather["alert"]["message"], "action": weather["alert"]["action_required"],
        })
    advisory.append({
        "agent": "Crop health agent", "severity": "INFO",
        "title": f"{crop['stress_level']} — {crop['growth_stage'].replace('_',' ')}",
        "body": (f"NDVI {crop['ndvi']:.3f} against a stage median of {crop['stage_baseline']:.3f} "
                  f"({crop['ndvi_deviation']:+.3f})."),
        "action": crop["treatment"],
    })
    advisory.append({
        "agent": "Soil agent", "severity": "INFO",
        "title": f"Soil health {soil['soil_health_score']:.1f}/10",
        "body": soil["interpretation"],
        "action": (f"Apply {soil['urea_kg_per_acre']} kg/acre Urea, {soil['dap_kg_per_acre']} kg/acre DAP, "
                    f"{soil['potash_kg_per_acre']} kg/acre Potash. {soil['timing_advice']}"),
    })
    advisory.append({
        "agent": "Pest risk agent", "severity": pest["risk_level"],
        "title": f"{pest['risk_level']} pest risk",
        "body": pest["rule_matched"], "action": pest["preventive_action"],
    })
    advisory.append({
        "agent": "Market price agent", "severity": "INFO",
        "title": f"{market['sell_or_hold']} — ₹{market['current_price_per_quintal']:,.0f}/qtl",
        "body": market["reasoning"], "action": f"Best historical selling month: {market['best_selling_month']}.",
    })

    status = system_status()
    return {
        "district": district, "season": season, "growth_stage": resolved_stage,
        "cards": {
            "soil_health_score": soil["soil_health_score"],
            "soil_sub": f"pH {soil['ph']} · OC {soil['organic_carbon']}%",
            "temperature": weather["temperature_avg"],
            "temperature_min": weather["temperature_min"],
            "weather_sub": f"{weather['weather_condition']} · {weather['rainfall_mm']:.0f} mm · {weather['humidity_percent']:.0f}% RH",
            "stress_level": crop["stress_level"],
            "crop_sub": f"NDVI {crop['ndvi']:.3f} ({crop['ndvi_deviation']:+.3f} vs median)",
            "market_price": market["current_price_per_quintal"],
            "market_sub": f"{market['sell_or_hold']} · {market['trend_direction'].lower()} · MSP ₹{market['msp_price_per_quintal']:,.0f}",
            "pest_risk": pest["risk_level"],
            "pest_sub": f"Re-scout in {pest['next_checkin_days']} days · {(pest['likely_pests'] or ['No major pest'])[0]}",
        },
        "advisory": advisory,
        "coverage": [
            {"label": "Districts covered", "value": f"{len(districts())} Karnataka paddy districts"},
            {"label": "Soil + yield records", "value": f"{len(main_df()):,} rows (2009–2024)"},
            {"label": "NDVI growth-stage records", "value": "1,400 stage-tagged field records"},
            {"label": "Daily weather rows", "value": f"{len(weather_df()):,} district-days"},
            {"label": "Market price rows", "value": f"{len(market_df()):,} variety-days"},
        ],
        "agents": status["active_agents"],
        "online_count": status["online_count"],
        "_meta": meta("GATEWAY_DATASET", started, "ORCHESTRATOR", "Fan-in across all five agents."),
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "krishisense-gateway"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("GATEWAY_PORT", 8000)))
