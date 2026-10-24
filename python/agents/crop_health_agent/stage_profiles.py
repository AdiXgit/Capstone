from datetime import datetime

# 7 growth stages present in Karnataka_Paddy_AI_ML_Dataset.xlsx / Crop_Growth_Stages
STAGE_ORDER = ["Sowing", "Germination", "Vegetative", "Flowering",
               "Grain_Filling", "Maturity", "Harvest"]

# (das_min, das_max), (ndvi_q25, ndvi_median, ndvi_q75) — data-derived per-stage baselines.
# NDVI deviation features are computed against the median here.
STAGE_PROFILES = {
    "Sowing":        {"das": (0, 6),     "ndvi": (0.000, 0.000, 0.000), "lai": (0.00, 0.00), "height": (0, 0),     "tillers": (0, 0)},
    "Germination":   {"das": (7, 12),    "ndvi": (0.241, 0.277, 0.314), "lai": (0.10, 0.30), "height": (5, 10),    "tillers": (1, 2)},
    "Vegetative":    {"das": (25, 35),   "ndvi": (0.638, 0.683, 0.714), "lai": (2.01, 3.49), "height": (40, 60),   "tillers": (8, 15)},
    "Flowering":     {"das": (50, 65),   "ndvi": (0.730, 0.772, 0.818), "lai": (4.00, 5.99), "height": (80, 100),  "tillers": (12, 20)},
    "Grain_Filling": {"das": (75, 90),   "ndvi": (0.686, 0.720, 0.760), "lai": (3.51, 5.00), "height": (95, 110),  "tillers": (12, 20)},
    "Maturity":      {"das": (105, 120), "ndvi": (0.538, 0.577, 0.617), "lai": (2.00, 3.49), "height": (100, 115), "tillers": (10, 18)},
    "Harvest":       {"das": (125, 140), "ndvi": (0.332, 0.365, 0.412), "lai": (1.50, 2.50), "height": (100, 115), "tillers": (10, 18)},
}

DISTRICT_COORDS = {
    "Koppal": (15.353, 76.155), "Ballari": (15.139, 76.921), "Raichur": (16.212, 77.357),
    "Davanagere": (14.464, 75.922), "Shivamogga": (13.930, 75.568), "Hassan": (13.007, 76.100),
    "Mandya": (12.522, 76.895), "Mysuru": (12.296, 76.639), "Belagavi": (15.850, 74.498),
    "Dharwad": (15.459, 75.008), "Bagalkot": (16.169, 75.697), "Bengaluru Rural": (13.199, 77.570),
    "Bidar": (17.914, 77.520), "Chamarajanagar": (11.926, 76.944), "Chikkaballapura": (13.435, 77.728),
    "Chikkamagaluru": (13.316, 75.772), "Chitradurga": (14.225, 76.398), "Dakshina Kannada": (12.844, 74.990),
    "Gadag": (15.416, 75.625), "Haveri": (14.795, 75.399), "Kalaburagi": (17.330, 76.820),
    "Kodagu": (12.338, 75.807), "Kolar": (13.136, 78.129), "Ramanagara": (12.716, 77.280),
    "Tumakuru": (13.340, 77.101), "Udupi": (13.341, 74.742), "Uttara Kannada": (14.786, 74.691),
    "Vijayapura": (16.830, 75.710), "Yadgir": (16.773, 77.138),
}
KARNATAKA_CENTROID = (14.5204, 75.7224)

SEASON_SOWING = {"Kharif": {"month": 6, "day": 15}, "Rabi": {"month": 11, "day": 15}}


def das_from_season(season):
    sow = SEASON_SOWING.get(season, {"month": 6, "day": 15})
    now = datetime.now()
    sowing = datetime(now.year, sow["month"], sow["day"])
    das = (now - sowing).days
    if das < 0:
        sowing = datetime(now.year - 1, sow["month"], sow["day"])
        das = (now - sowing).days
    return max(0, min(das, 140))


def das_to_stage(das):
    for stage, prof in STAGE_PROFILES.items():
        lo, hi = prof["das"]
        if lo <= das <= hi:
            return stage
    return "Harvest" if das > 120 else "Sowing"


def stage_index(stage):
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 0


def stage_ndvi_median(stage):
    return STAGE_PROFILES.get(stage, {}).get("ndvi", (0.5, 0.6, 0.7))[1]
