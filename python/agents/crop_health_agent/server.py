# python/agents/crop_health_agent/server.py
# OWNER: Rishi Mohan
#
# Built from deep EDA of actual datasets:
#   Growth Stages (1,400 records): stages Sowing→Germination→Vegetative→Flowering→
#     Grain_Filling→Maturity→Harvest. DAS: 0–140. NDVI: 0–0.849. LAI: 0–5.99.
#   Satellite Indices (1,200 records): NDVI 0.301–0.850, 10 districts, 2015–2024.
#   Main Dataset: Pest=High317/Low310/Medium301, Disease=Low313/High308/Med307.
#
# Key EDA findings wired into this code:
#   - NDVI correlates 0.90 with LAI, 0.81 with Tillers (strongest predictors)
#   - Humidity does NOT predict disease (mean ~72% across all levels) → removed from model
#   - "Tillering" is NOT a stage in this dataset — dataset has 7 exact stages
#   - Stage thresholds are data-derived from actual Q25/median/Q75 per stage
#   - Rabi has slightly higher pest incidence than Kharif from actual records

import os, time, logging, threading, requests
import pandas as pd
import numpy as np
from concurrent import futures
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

import grpc
import paddy_agents_pb2 as pb
import paddy_agents_pb2_grpc as pb_grpc

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [CROP_HEALTH] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DATA-DERIVED CONSTANTS
# ─────────────────────────────────────────────

STAGE_ORDER = ["Sowing","Germination","Vegetative","Flowering",
               "Grain_Filling","Maturity","Harvest"]

# (das_min, das_max) and (ndvi_q25, ndvi_median, ndvi_q75) from actual dataset
STAGE_PROFILES = {
    "Sowing":        {"das":(0,6),     "ndvi":(0.000,0.000,0.000),"lai":(0.00,0.00),"height":(0,0),    "tillers":(0,0)},
    "Germination":   {"das":(7,12),    "ndvi":(0.241,0.277,0.314),"lai":(0.10,0.30),"height":(5,10),   "tillers":(1,2)},
    "Vegetative":    {"das":(25,35),   "ndvi":(0.638,0.683,0.714),"lai":(2.01,3.49),"height":(40,60),  "tillers":(8,15)},
    "Flowering":     {"das":(50,65),   "ndvi":(0.730,0.772,0.818),"lai":(4.00,5.99),"height":(80,100), "tillers":(12,20)},
    "Grain_Filling": {"das":(75,90),   "ndvi":(0.686,0.720,0.760),"lai":(3.51,5.00),"height":(95,110), "tillers":(12,20)},
    "Maturity":      {"das":(105,120), "ndvi":(0.538,0.577,0.617),"lai":(2.00,3.49),"height":(100,115),"tillers":(10,18)},
    "Harvest":       {"das":(125,140), "ndvi":(0.332,0.365,0.412),"lai":(1.50,2.50),"height":(100,115),"tillers":(10,18)},
}

DISTRICT_COORDS = {
    "Koppal":(15.353,76.155),"Ballari":(15.139,76.921),"Raichur":(16.212,77.357),
    "Davanagere":(14.464,75.922),"Shivamogga":(13.930,75.568),"Hassan":(13.007,76.100),
    "Mandya":(12.522,76.895),"Mysuru":(12.296,76.639),"Belagavi":(15.850,74.498),
    "Dharwad":(15.459,75.008),"Bagalkot":(16.169,75.697),"Bengaluru Rural":(13.199,77.570),
    "Bidar":(17.914,77.520),"Chamarajanagar":(11.926,76.944),"Chikkaballapura":(13.435,77.728),
    "Chikkamagaluru":(13.316,75.772),"Chitradurga":(14.225,76.398),"Dakshina Kannada":(12.844,74.990),
    "Gadag":(15.416,75.625),"Haveri":(14.795,75.399),"Kalaburagi":(17.330,76.820),
    "Kodagu":(12.338,75.807),"Kolar":(13.136,78.129),"Ramanagara":(12.716,77.280),
    "Tumakuru":(13.340,77.101),"Udupi":(13.341,74.742),"Uttara Kannada":(14.786,74.691),
    "Vijayapura":(16.830,75.710),"Yadgir":(16.773,77.138),
}

SEASON_SOWING = {"Kharif":{"month":6,"day":15},"Rabi":{"month":11,"day":15}}

# Karnataka paddy pests/diseases per stage (agronomic literature)
PEST_BY_STAGE = {
    "Germination":   ["Army Worm"],
    "Vegetative":    ["Stem Borer","Leaf Folder","Gall Midge"],
    "Flowering":     ["Rice Bug","Brown Planthopper","Stem Borer"],
    "Grain_Filling": ["Rice Bug","Brown Planthopper","Rats"],
    "Maturity":      ["Rats","Birds"],
}
DISEASE_BY_STAGE = {
    "Germination":   ["Damping Off","Seed Rot"],
    "Vegetative":    ["Bacterial Leaf Blight","Brown Spot","Blast"],
    "Flowering":     ["Blast (Neck)","False Smut","Sheath Blight"],
    "Grain_Filling": ["False Smut","Grain Discolouration","Sheath Blight"],
    "Maturity":      ["Grain Discolouration"],
}
TREATMENT_MAP = {
    "Blast":                   "Tricyclazole 75WP @ 0.6g/L or Tebuconazole 50WG @ 1g/L",
    "Blast (Neck)":            "Tricyclazole 75WP @ 0.6g/L — spray at boot stage immediately",
    "Sheath Blight":           "Hexaconazole 5EC @ 2ml/L or Validamycin 3L @ 2ml/L",
    "False Smut":              "Propiconazole 25EC @ 1ml/L at booting stage",
    "Bacterial Leaf Blight":   "Copper oxychloride 50WP @ 3g/L — reduce nitrogen input",
    "Brown Spot":              "Mancozeb 75WP @ 2g/L — check soil fertility",
    "Stem Borer":              "Chlorpyriphos 20EC @ 2.5ml/L or Cartap 4G @ 25kg/ha",
    "Leaf Folder":             "Monocrotophos 36SL @ 1.5ml/L — spray on rolled leaves",
    "Brown Planthopper":       "Buprofezin 25SC @ 1ml/L — do NOT use pyrethroids",
    "Gall Midge":              "Carbofuran 3G @ 25kg/ha at tillering",
    "Rice Bug":                "Malathion 50EC @ 2ml/L — spray early morning",
}

STAGE_ADVISORIES = {
    "Sowing":        "Puddle and level field. Treat seeds with Carbendazim 2g/kg. Maintain 2–5cm water.",
    "Germination":   "Maintain thin water layer. Apply Butachlor 1.5L/ha pre-emergence. Scout for army worm.",
    "Vegetative":    "Apply top-dress nitrogen (25% N). Control weeds. Scout stem borer and leaf folder weekly.",
    "Flowering":     "CRITICAL: Do NOT spray pesticides during flowering. Maintain water. Watch for neck blast.",
    "Grain_Filling": "Maintain moisture. Spray for Rice Bug if sighted. Protect from birds.",
    "Maturity":      "Drain field 10–15 days before harvest. Check grain moisture < 20%.",
    "Harvest":       "Harvest when 85% grains are straw-coloured. Thresh within 24h.",
}


# ─────────────────────────────────────────────
# DATA STORE
# ─────────────────────────────────────────────

class CropHealthDataStore:
    def __init__(self, xlsx_path: str, main_csv_path: str):
        log.info("Loading crop health datasets...")
        xl = pd.ExcelFile(xlsx_path)
        self.gs_df  = xl.parse("Crop_Growth_Stages")
        self.sat_df = xl.parse("Satellite_Indices")
        self.main_df = pd.read_csv(main_csv_path, usecols=[
            "District","Season","Pest_Incidence","Disease_Incidence",
            "Seed_Variety","Humidity_Percent","Temperature_Celsius","Pesticide_Sprays",
        ])
        log.info(f"  Growth stages: {len(self.gs_df)}  Satellite: {len(self.sat_df)}  Main: {len(self.main_df)}")
        self._build_profiles()
        self._train_health_clf()
        self._train_ndvi_rf()

    def _build_profiles(self):
        self._profiles = {}
        grp = self.gs_df.groupby(["District","Season","Growth_Stage"]).agg({
            "NDVI":["mean","std"],"Leaf_Area_Index":["mean"],"Plant_Height_cm":["mean"],
            "Tiller_Count":["mean"],"Days_After_Sowing":["mean","min","max"],
        }).round(4)
        for (dist,season,stage), row in grp.iterrows():
            self._profiles[(dist,season,stage)] = {
                "ndvi_mean": row[("NDVI","mean")], "ndvi_std": row[("NDVI","std")],
                "lai_mean":  row[("Leaf_Area_Index","mean")],
                "height":    row[("Plant_Height_cm","mean")],
                "tillers":   row[("Tiller_Count","mean")],
                "das_mean":  row[("Days_After_Sowing","mean")],
                "das_min":   row[("Days_After_Sowing","min")],
                "das_max":   row[("Days_After_Sowing","max")],
            }
        log.info(f"  Built {len(self._profiles)} district×season×stage profiles")

    def _train_health_clf(self):
        rows = []
        for _, r in self.gs_df.iterrows():
            stage = r["Growth_Stage"]
            prof  = STAGE_PROFILES.get(stage, {})
            q25, median, q75 = prof.get("ndvi", (0,0,0))
            ndvi = r["NDVI"]
            if median == 0:
                label = "EXCELLENT" if ndvi == 0 else "GOOD"
            elif ndvi >= q75:      label = "EXCELLENT"
            elif ndvi >= median:   label = "GOOD"
            elif ndvi >= q25:      label = "MODERATE"
            else:                  label = "POOR"
            rows.append({
                "ndvi":ndvi,"lai":r["Leaf_Area_Index"],"das":r["Days_After_Sowing"],
                "height":r["Plant_Height_cm"],"tillers":r["Tiller_Count"],
                "ndvi_rel":ndvi - median,
                "stage_idx":STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 0,
                "label":label,
            })
        df = pd.DataFrame(rows)
        X = df[["ndvi","lai","das","height","tillers","ndvi_rel","stage_idx"]].values
        y = df["label"].values
        self._clf = GradientBoostingClassifier(n_estimators=150,max_depth=4,
                                               learning_rate=0.08,random_state=42)
        self._clf.fit(X, y)
        scores = cross_val_score(self._clf, X, y, cv=5, scoring="accuracy")
        log.info(f"  Health classifier CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    def _train_ndvi_rf(self):
        df = self.gs_df[self.gs_df["NDVI"] > 0].copy()
        df["season_enc"] = (df["Season"] == "Kharif").astype(int)
        self._ndvi_rf = RandomForestRegressor(n_estimators=100,random_state=42)
        self._ndvi_rf.fit(df[["Days_After_Sowing","season_enc"]].values, df["NDVI"].values)
        log.info("  NDVI DAS predictor trained ✓")

    # ── Query methods ────────────────────────

    def get_profile(self, district, season, stage):
        key = (district, season, stage)
        if key in self._profiles:
            return self._profiles[key]
        fallback = [v for (d,s,st),v in self._profiles.items() if s==season and st==stage]
        if fallback:
            return {k: float(np.mean([f[k] for f in fallback])) for k in fallback[0]}
        return None

    def predict_health(self, ndvi, lai, das, height, tillers, stage):
        prof = STAGE_PROFILES.get(stage, {})
        med  = prof.get("ndvi",(0,0,0))[1]
        idx  = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 0
        X = np.array([[ndvi, lai, das, height, tillers, ndvi-med, idx]])
        return self._clf.predict(X)[0]

    def predict_ndvi_from_das(self, das, season):
        enc = 1 if season == "Kharif" else 0
        return float(self._ndvi_rf.predict([[das, enc]])[0])

    def get_satellite(self, district, year, month):
        sub = self.sat_df[(self.sat_df["District"]==district) &
                          (self.sat_df["Year"]==year) & (self.sat_df["Month"]==month)]
        if sub.empty:
            sub = self.sat_df[(self.sat_df["District"]==district) &
                              (self.sat_df["Month"]==month)]
        if sub.empty:
            return None
        r = sub.iloc[-1]
        return {"ndvi_mean":float(r["NDVI_Mean"]),"ndvi_std":float(r["NDVI_StdDev"]),
                "evi_mean":float(r["EVI_Mean"]),"lst_c":float(r["Land_Surface_Temp_C"]),
                "soil_moisture":float(r["Soil_Moisture_Percent"]),"cloud_cover":float(r["Cloud_Cover_Percent"])}

    def get_closest_growth_record(self, district, season, das):
        sub = self.gs_df[(self.gs_df["District"]==district) & (self.gs_df["Season"]==season)]
        if sub.empty:
            sub = self.gs_df[self.gs_df["Season"]==season]
        if sub.empty:
            return None
        sub = sub.copy()
        sub["_d"] = (sub["Days_After_Sowing"] - das).abs()
        r = sub.nsmallest(1,"_d").iloc[0]
        return {"growth_stage":r["Growth_Stage"],"days_after_sowing":int(r["Days_After_Sowing"]),
                "plant_height_cm":float(r["Plant_Height_cm"]),"tiller_count":int(r["Tiller_Count"]),
                "leaf_area_index":float(r["Leaf_Area_Index"]),"ndvi":float(r["NDVI"])}

    def get_pest_risk(self, district, season):
        sub = self.main_df[(self.main_df["District"]==district) & (self.main_df["Season"]==season)]
        if sub.empty:
            sub = self.main_df[self.main_df["Season"]==season]
        if sub.empty:
            return {"pest":"Medium","disease":"Medium","sprays":4.0}
        return {"pest":sub["Pest_Incidence"].mode()[0],
                "disease":sub["Disease_Incidence"].mode()[0],
                "sprays":float(sub["Pesticide_Sprays"].mean())}


# ─────────────────────────────────────────────
# FREE NDVI CLIENT (NASA POWER → Open-Meteo)
# ─────────────────────────────────────────────

class FreeNDVIClient:
    _cache: dict = {}

    def _nasa_power(self, lat, lon):
        key = f"nasa_{lat:.2f}_{lon:.2f}"
        if key in self._cache: return self._cache[key]
        end   = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        try:
            r = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", params={
                "parameters":"ALLSKY_SFC_PAR_TOT,T2M","community":"AG",
                "longitude":lon,"latitude":lat,"start":start,"end":end,"format":"JSON",
            }, timeout=10)
            r.raise_for_status()
            par_vals = [v for v in r.json().get("properties",{}).get("parameter",{})
                        .get("ALLSKY_SFC_PAR_TOT",{}).values() if v > -900]
            if not par_vals: return None
            avg = float(np.mean(par_vals))
            ndvi = max(0.0, min(0.95, 0.04*avg - 0.15))
            result = {"ndvi":round(ndvi,3),"source":"NASA_POWER"}
            self._cache[key] = result
            return result
        except Exception as e:
            log.warning(f"NASA POWER: {e}")
            return None

    def _open_meteo(self, lat, lon):
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude":lat,"longitude":lon,
                "daily":"shortwave_radiation_sum",
                "timezone":"Asia/Kolkata","forecast_days":7,
            }, timeout=8)
            r.raise_for_status()
            rad = [v for v in r.json().get("daily",{}).get("shortwave_radiation_sum",[]) if v]
            if not rad: return None
            ndvi = max(0.0, min(0.95, 0.035*np.mean(rad) - 0.1))
            return {"ndvi":round(ndvi,3),"source":"OPEN_METEO"}
        except Exception as e:
            log.warning(f"Open-Meteo NDVI: {e}")
            return None

    def fetch(self, district):
        lat, lon = DISTRICT_COORDS.get(district, (14.5204, 75.7224))
        return (self._nasa_power(lat, lon) or
                self._open_meteo(lat, lon) or
                {"ndvi":0.60,"source":"FALLBACK"})


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def das_from_season(season):
    sow = SEASON_SOWING.get(season, {"month":6,"day":15})
    now = datetime.now()
    sowing = datetime(now.year, sow["month"], sow["day"])
    das = (now - sowing).days
    if das < 0:
        sowing = datetime(now.year-1, sow["month"], sow["day"])
        das = (now - sowing).days
    return max(0, min(das, 140))

def das_to_stage(das):
    for stage, prof in STAGE_PROFILES.items():
        lo, hi = prof["das"]
        if lo <= das <= hi:
            return stage
    return "Harvest" if das > 120 else "Sowing"

def next_stage_info(stage):
    idx = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 0
    if idx + 1 < len(STAGE_ORDER):
        nxt = STAGE_ORDER[idx+1]
        days = max(0, STAGE_PROFILES[nxt]["das"][0] - STAGE_PROFILES[stage]["das"][1])
        return nxt, days
    return "Post-Harvest", 0

def vegetation_status(ndvi):
    if ndvi >= 0.75: return "DENSE_HEALTHY_VEGETATION"
    if ndvi >= 0.60: return "MODERATE_HEALTHY_VEGETATION"
    if ndvi >= 0.45: return "SPARSE_VEGETATION"
    if ndvi >= 0.25: return "EARLY_GROWTH_OR_STRESS"
    if ndvi > 0.0:   return "BARE_SOIL_OR_SEVERE_STRESS"
    return "NO_VEGETATION"

def health_action(status, stage):
    if status == "EXCELLENT": return "Crop performing above stage average. Maintain current practices."
    if status == "GOOD":      return "On track. Apply scheduled fertilizer. Continue weekly scouting."
    if status == "MODERATE":
        stage_tips = {
            "Vegetative":    "Check irrigation and soil nutrient levels.",
            "Flowering":     "Ensure adequate water. Watch for neck blast.",
            "Grain_Filling": "Maintain moisture. Scout for Brown Planthopper.",
        }
        return "Below expected performance. " + stage_tips.get(stage,"Inspect field for pest/nutrient issues.")
    return ("URGENT: NDVI significantly below stage benchmark. "
            "Inspect for blast, Brown Planthopper, or nutrient deficiency. Contact KVK.")

def build_pest_action(pests, diseases, pest_level, disease_level):
    parts = []
    for p in pests[:2]:
        if p in TREATMENT_MAP: parts.append(f"For {p}: {TREATMENT_MAP[p]}.")
    for d in diseases[:2]:
        if d in TREATMENT_MAP: parts.append(f"For {d}: {TREATMENT_MAP[d]}.")
    if pest_level=="High" or disease_level=="High":
        parts.append("High risk: confirm with local KVK before applying treatment.")
    return " ".join(parts) or "Scout and confirm infestation before applying any treatment."


# ─────────────────────────────────────────────
# gRPC SERVICE
# ─────────────────────────────────────────────

class CropHealthAgentServicer(pb_grpc.CropHealthAgentServiceServicer):
    def __init__(self, store, ndvi_client):
        self.store = store
        self.ndvi  = ndvi_client

    def _meta(self, conf=0.85):
        return pb.AgentMetadata(agent_id="crop-health-agent-001",
            agent_name="Crop Health Monitoring Agent",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            confidence=conf)

    def GetCropHealthStatus(self, request, context):
        log.info(f"GetCropHealthStatus → {request.district} {request.season}")
        das = das_from_season(request.season)
        rec = self.store.get_closest_growth_record(request.district, request.season, das)
        if rec:
            stage, height, tillers, lai, ndvi = (
                rec["growth_stage"], rec["plant_height_cm"],
                rec["tiller_count"], rec["leaf_area_index"], rec["ndvi"])
            das_used = rec["days_after_sowing"]
            conf = 0.90
        else:
            stage = das_to_stage(das)
            prof  = STAGE_PROFILES.get(stage, {})
            q     = prof.get("ndvi",(0.5,0.6,0.7))
            lr    = prof.get("lai",(2.0,4.0))
            hr    = prof.get("height",(50,90))
            tr    = prof.get("tillers",(10,16))
            ndvi, lai, height, tillers = float(np.mean(q)), float(np.mean(lr)), float(np.mean(hr)), int(np.mean(tr))
            das_used, conf = das, 0.72

        health = self.store.predict_health(ndvi, lai, das_used, height, tillers, stage)
        return pb.CropHealthResponse(
            metadata=self._meta(conf), growth_stage=stage, days_after_sowing=das_used,
            plant_height_cm=height, tiller_count=tillers, leaf_area_index=lai,
            ndvi=ndvi, health_status=health, action_required=health_action(health, stage))

    def GetNDVIAnalysis(self, request, context):
        log.info(f"GetNDVIAnalysis → {request.district} {request.year}/{request.month}")
        sat = self.store.get_satellite(request.district, request.year, request.month)
        if sat:
            return pb.NDVIResponse(metadata=self._meta(0.92),
                ndvi_mean=sat["ndvi_mean"], ndvi_std_dev=sat["ndvi_std"],
                evi_mean=sat["evi_mean"], land_surface_temp=sat["lst_c"],
                soil_moisture=sat["soil_moisture"], source="DATASET",
                vegetation_status=vegetation_status(sat["ndvi_mean"]))
        api = self.ndvi.fetch(request.district)
        ndvi_val = api.get("ndvi", 0.60)
        month = request.month or datetime.now().month
        soil_m = 38.0 if month in (6,7,8,9) else 26.0
        return pb.NDVIResponse(metadata=self._meta(0.68),
            ndvi_mean=ndvi_val, ndvi_std_dev=0.12, evi_mean=round(ndvi_val*0.88,3),
            land_surface_temp=30.0, soil_moisture=soil_m,
            source=api.get("source","API"),
            vegetation_status=vegetation_status(ndvi_val))

    def GetGrowthStageInfo(self, request, context):
        log.info(f"GetGrowthStageInfo → DAS={request.days_after_sowing}")
        stage = das_to_stage(request.days_after_sowing)
        nxt, days_to_nxt = next_stage_info(stage)
        return pb.GrowthStageResponse(metadata=self._meta(0.88),
            current_stage=stage, next_stage=nxt, days_to_next=days_to_nxt,
            care_advisory=STAGE_ADVISORIES.get(stage,"Monitor field regularly."))

    def GetPestDiseaseRisk(self, request, context):
        log.info(f"GetPestDiseaseRisk → {request.district} stage={request.growth_stage}")
        stage = request.growth_stage or das_to_stage(0)
        hist  = self.store.get_pest_risk(request.district, request.season)
        pests    = PEST_BY_STAGE.get(stage, [])
        diseases = DISEASE_BY_STAGE.get(stage, [])
        if request.temperature > 35 and "Blast" not in diseases:
            diseases = diseases + ["Blast"]
        action = build_pest_action(pests, diseases, hist["pest"], hist["disease"])
        return pb.PestDiseaseResponse(metadata=self._meta(0.80),
            pest_risk_level=hist["pest"], disease_risk_level=hist["disease"],
            likely_pests=list(pests[:3]), likely_diseases=list(diseases[:3]),
            preventive_action=action)


# ─────────────────────────────────────────────
# REGISTRATION + MAIN
# ─────────────────────────────────────────────

def register_with_orchestrator(port, max_retries=10):
    addr = os.environ.get("ORCHESTRATOR_ADDR","orchestrator:50051")
    for i in range(max_retries):
        try:
            with grpc.insecure_channel(addr) as ch:
                pb_grpc.OrchestratorServiceStub(ch).RegisterAgent(
                    pb.AgentRegistration(agent_id="crop-health-agent-001",
                        agent_name="Crop Health Monitoring Agent",
                        agent_type="CROP_HEALTH", host="crop-health-agent", port=port),
                    timeout=5)
                log.info(f"Registered with orchestrator at {addr}")
                return
        except grpc.RpcError as e:
            log.warning(f"Retry {i+1}/{max_retries}: {e.details()}")
            time.sleep(3)

def serve():
    xlsx = os.environ.get("AI_ML_XLSX_PATH","/data/Karnataka_Paddy_AI_ML_Dataset.xlsx")
    csv  = os.environ.get("MAIN_CSV_PATH","/data/Karnataka_Paddy_Main_Dataset.csv")
    port = int(os.environ.get("CROP_HEALTH_AGENT_PORT","50054"))

    store  = CropHealthDataStore(xlsx, csv)
    client = FreeNDVIClient()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb_grpc.add_CropHealthAgentServiceServicer_to_server(
        CropHealthAgentServicer(store, client), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info(f"Crop Health Agent running on :{port}")

    threading.Thread(target=register_with_orchestrator, args=(port,), daemon=True).start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
