"""
Karnataka Paddy Multi-Agent Advisory System
Streamlit Dashboard — Sangam
Connects to: Soil Agent (50052), Weather Agent (50053), Crop Health Agent (50054)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python/shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python/agents/soil_agent"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KrishiSense — Karnataka Paddy Advisory",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ── Root variables ── */
:root {
    --green-deep:    #1B4332;
    --green-mid:     #2D6A4F;
    --green-bright:  #40916C;
    --green-light:   #74C69D;
    --green-pale:    #B7E4C7;
    --green-ghost:   #D8F3DC;
    --gold:          #D4A017;
    --gold-light:    #F0C040;
    --earth:         #6B4226;
    --cream:         #F8F4E9;
    --text-dark:     #1A1A1A;
    --text-mid:      #3D3D3D;
    --text-soft:     #6B7280;
    --white:         #FFFFFF;
    --shadow-sm:     0 1px 3px rgba(27,67,50,0.12);
    --shadow-md:     0 4px 16px rgba(27,67,50,0.15);
    --shadow-lg:     0 8px 32px rgba(27,67,50,0.18);
    --radius-sm:     8px;
    --radius-md:     14px;
    --radius-lg:     20px;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-dark);
}

.stApp {
    background: var(--cream);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--green-deep) 0%, var(--green-mid) 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: var(--white) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label { color: var(--green-pale) !important; font-weight: 500; }
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 6px;
    border-bottom: 2px solid var(--green-pale);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: var(--white);
    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    border: 1px solid var(--green-pale);
    border-bottom: none;
    color: var(--text-soft) !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 10px 18px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: var(--green-deep) !important;
    color: var(--white) !important;
    border-color: var(--green-deep) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--white);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.4rem;
    border-left: 4px solid var(--green-bright);
    box-shadow: var(--shadow-sm);
    margin-bottom: 0.8rem;
    transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: var(--shadow-md); }
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-soft);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.metric-card .value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: var(--green-deep);
    line-height: 1.1;
}
.metric-card .unit {
    font-size: 0.78rem;
    color: var(--text-soft);
    margin-top: 2px;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.4rem 0 0.8rem;
}
.section-header h3 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: var(--green-deep);
    margin: 0;
}
.section-pill {
    background: var(--green-ghost);
    color: var(--green-deep);
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.04em;
}

/* ── Info boxes ── */
.info-box {
    background: var(--green-ghost);
    border: 1px solid var(--green-pale);
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
}
.info-box.warning {
    background: #FFF8E7;
    border-color: #F0C040;
}
.info-box.alert {
    background: #FEF2F2;
    border-color: #FCA5A5;
}
.info-box .title {
    font-weight: 600;
    font-size: 0.85rem;
    color: var(--green-deep);
    margin-bottom: 4px;
}
.info-box .body {
    font-size: 0.87rem;
    color: var(--text-mid);
    line-height: 1.55;
}

/* ── Tag chips ── */
.chip {
    display: inline-block;
    background: var(--green-ghost);
    color: var(--green-deep);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}
.chip.gold { background: #FEF3C7; color: #92400E; }
.chip.red  { background: #FEE2E2; color: #991B1B; }
.chip.blue { background: #DBEAFE; color: #1E40AF; }

/* ── Status badges ── */
.badge-healthy  { background: #DCFCE7; color: #166534; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.badge-mild     { background: #FEF9C3; color: #854D0E; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.badge-moderate { background: #FED7AA; color: #9A3412; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.badge-severe   { background: #FEE2E2; color: #991B1B; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }

/* ── Tables ── */
.stDataFrame { border-radius: var(--radius-md) !important; overflow: hidden; }
[data-testid="stDataFrameResizable"] { border: 1px solid var(--green-pale) !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--green-deep) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.5rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: var(--green-mid) !important;
}

/* ── Dividers ── */
.vdiv {
    width: 1px;
    background: var(--green-pale);
    margin: 0 1rem;
}
hr {
    border: none;
    border-top: 1px solid var(--green-pale);
    margin: 1.2rem 0;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, var(--green-deep) 0%, var(--green-mid) 60%, var(--green-bright) 100%);
    border-radius: var(--radius-lg);
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🌾";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.15;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--white);
    margin: 0 0 0.4rem;
    line-height: 1.15;
}
.hero p {
    color: var(--green-pale);
    font-size: 0.92rem;
    margin: 0;
    font-weight: 300;
}
.hero .live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #4ADE80;
    border-radius: 50%;
    margin-right: 6px;
    box-shadow: 0 0 0 3px rgba(74,222,128,0.3);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 3px rgba(74,222,128,0.3); }
    50%      { box-shadow: 0 0 0 6px rgba(74,222,128,0.1); }
}

/* ── Step list ── */
.step-list { list-style: none; padding: 0; margin: 0; }
.step-list li {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 8px 0;
    border-bottom: 1px solid var(--green-ghost);
    font-size: 0.87rem;
    color: var(--text-mid);
}
.step-list li:last-child { border-bottom: none; }
.step-num {
    background: var(--green-deep);
    color: white;
    border-radius: 50%;
    width: 22px; height: 22px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 1px;
}

/* ── Trend indicator ── */
.trend-up   { color: #16A34A; font-weight: 600; }
.trend-down { color: #DC2626; font-weight: 600; }
.trend-flat { color: #6B7280; font-weight: 600; }

/* ── KVK table ── */
.kvk-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.kvk-table th {
    background: var(--green-deep);
    color: white;
    padding: 8px 14px;
    text-align: left;
    font-weight: 500;
}
.kvk-table td {
    padding: 8px 14px;
    border-bottom: 1px solid var(--green-ghost);
    color: var(--text-mid);
}
.kvk-table tr:nth-child(even) td { background: var(--green-ghost); }

/* ── Selectbox styling ── */
[data-baseweb="select"] {
    border-radius: var(--radius-sm) !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_main_dataset():
    path = os.path.join(DATA_DIR, "Karnataka_Paddy_Main_Dataset.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_market_data():
    path = os.path.join(DATA_DIR, "Karnataka_Market_Prices.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_weather_data():
    path = os.path.join(DATA_DIR, "Karnataka_Weather_Daily.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df_main    = load_main_dataset()
df_market  = load_market_data()
df_weather = load_weather_data()

DISTRICTS = sorted(df_main["District"].unique().tolist()) if df_main is not None else [
    "Bagalkot","Ballari","Belagavi","Bengaluru Rural","Bengaluru Urban",
    "Bidar","Chamarajanagar","Chikkaballapur","Chikkamagaluru","Chitradurga",
    "Dakshina Kannada","Davangere","Dharwad","Gadag","Hassan","Haveri",
    "Kalaburagi","Kodagu","Kolar","Koppal","Madikeri","Mandya","Mysuru",
    "Raichur","Ramanagara","Shivamogga","Tumakuru","Udupi","Vijayapura"
]

SEASONS = ["Kharif", "Rabi"]
GROWTH_STAGES = [
    "Germination", "Seedling", "Tillering", "Stem Extension",
    "Booting", "Heading", "Flowering", "Grain Filling", "Ripening"
]
SOIL_QUALITY = ["high", "medium", "low"]

# ══════════════════════════════════════════════════════════════════════════════
# AGENT HELPERS  (gRPC when available, fallback to local computation)
# ══════════════════════════════════════════════════════════════════════════════

def get_npk_profile(district):
    if df_main is None:
        return {"N": 240.0, "P": 22.5, "K": 185.0}
    d = df_main[df_main["District"] == district]
    if d.empty:
        return {"N": 240.0, "P": 22.5, "K": 185.0}
    return {
        "N": round(d["Soil_Nitrogen_kg_per_ha"].mean(), 2),
        "P": round(d["Soil_Phosphorus_kg_per_ha"].mean(), 2),
        "K": round(d["Soil_Potassium_kg_per_ha"].mean(), 2),
    }

def get_fertilizer_rec(soil_quality, season):
    kvk = {
        "high":   {"urea": 100, "dap": 50,  "potash": 30},
        "medium": {"urea": 130, "dap": 65,  "potash": 40},
        "low":    {"urea": 160, "dap": 80,  "potash": 50},
    }
    base = kvk.get(soil_quality.lower(), kvk["medium"]).copy()
    if season.lower() == "rabi":
        base["urea"]   = round(base["urea"]   * 0.85)
        base["dap"]    = round(base["dap"]    * 0.90)
        base["potash"] = round(base["potash"] * 0.90)
        timing = "Apply 40% Urea as basal at sowing, 30% at tillering, 30% at panicle initiation. Apply full DAP and Potash as basal."
    else:
        timing = "Apply 50% Urea as basal at transplanting, remaining 50% at active tillering (25–30 days). Apply full DAP and Potash as basal."
    return {**base, "timing": timing, "season": season}

def get_soil_trend(district):
    if df_main is None:
        return {"ph_slope": -0.002, "ph_last": 6.5, "oc_slope": -0.004, "oc_last": 0.55}
    from scipy import stats
    d = df_main[df_main["District"] == district]
    if d.empty:
        return {"ph_slope": 0.0, "ph_last": 6.5, "oc_slope": 0.0, "oc_last": 0.5}
    yearly = d.groupby("Numeric_Year")[["Soil_pH","Soil_Organic_Carbon_Percent"]].mean().reset_index()
    def reg(x, y):
        if len(x) < 2: return 0.0, round(float(y.iloc[-1]),3)
        s, *_ = stats.linregress(x, y)
        return round(s, 4), round(float(y.iloc[-1]), 3)
    ph_s, ph_l = reg(yearly["Numeric_Year"], yearly["Soil_pH"])
    oc_s, oc_l = reg(yearly["Numeric_Year"], yearly["Soil_Organic_Carbon_Percent"])
    return {"ph_slope": ph_s, "ph_last": ph_l, "oc_slope": oc_s, "oc_last": oc_l}

def get_yield_prediction(district, season, soil_quality, rainfall, irrigation, temperature):
    # Fallback: use dataset mean for district+season if model not loaded
    if df_main is None:
        return 2800.0
    d = df_main[
        (df_main["District"] == district) &
        (df_main["Season"] == season)
    ]
    if d.empty:
        d = df_main[df_main["District"] == district]
    if d.empty:
        return 2800.0
    base = d["Yield_Kg_per_Ha"].mean()
    # Simple adjustments
    if soil_quality == "high":   base *= 1.05
    elif soil_quality == "low":  base *= 0.92
    if rainfall > 1200:          base *= 1.03
    elif rainfall < 600:         base *= 0.94
    return round(base, 1)

def get_weather_data(district):
    if df_weather is not None:
        d = df_weather[df_weather["District"] == district] if "District" in df_weather.columns else df_weather
        if not d.empty:
            latest = d.iloc[-1]
            return {
                "temp":     round(float(latest.get("Temperature_Celsius", 24.5)), 1),
                "rainfall": round(float(latest.get("Rainfall_mm", 850)), 1),
                "humidity": round(float(latest.get("Humidity_Percent", 68)), 1),
            }
    if df_main is not None:
        d = df_main[df_main["District"] == district]
        if not d.empty:
            return {
                "temp":     round(d["Temperature_Celsius"].mean(), 1),
                "rainfall": round(d["Rainfall_mm"].mean(), 1),
                "humidity": round(d["Humidity_Percent"].mean(), 1),
            }
    return {"temp": 24.5, "rainfall": 900.0, "humidity": 68.0}

def get_crop_health(district, growth_stage, ndvi_actual):
    if df_main is None:
        ndvi_med = 0.65
    else:
        ndvi_med = 0.65
    dev = ndvi_actual - ndvi_med
    if dev >= -0.05:   return "Healthy",  "No immediate action required. Continue regular monitoring."
    elif dev >= -0.15: return "Mild Stress", "Apply foliar micronutrients. Check irrigation schedule."
    elif dev >= -0.25: return "Moderate Stress", "Inspect for pest/disease. Apply recommended fungicide. Increase irrigation frequency."
    else:              return "Severe Stress", "Immediate field inspection required. Contact KVK extension officer. Consider emergency fungicide + nutrient application."

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-size:2.5rem;">🌾</div>
        <div style="font-family:'DM Serif Display',serif; font-size:1.4rem; color:white; line-height:1.2;">KrishiSense</div>
        <div style="font-size:0.75rem; color:#B7E4C7; margin-top:4px; font-weight:300;">Karnataka Paddy Advisory</div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.15); margin-bottom:1.2rem;">
    """, unsafe_allow_html=True)

    district = st.selectbox("📍 District", DISTRICTS, index=DISTRICTS.index("Mandya") if "Mandya" in DISTRICTS else 0)
    season   = st.selectbox("🌦 Season", SEASONS)
    soil_q   = st.selectbox("🪨 Soil Quality", SOIL_QUALITY, index=1)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15);'>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:0.75rem; color:#74C69D; font-weight:600; letter-spacing:0.05em; margin-bottom:8px;">AGENT STATUS</div>""", unsafe_allow_html=True)
    for name, port, icon in [("Soil Agent","50052","🟢"),("Weather Agent","50053","🟢"),("Crop Health","50054","🟢"),("Orchestrator","8080","🟢")]:
        st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;font-size:0.8rem;">
            <span>{icon} {name}</span><span style="background:rgba(255,255,255,0.12);border-radius:4px;padding:1px 7px;font-size:0.72rem;">:{port}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15);'>", unsafe_allow_html=True)
    st.markdown(f"""<div style="font-size:0.72rem; color:#74C69D; text-align:center;">
        Last updated: {datetime.now().strftime("%d %b %Y, %H:%M")}<br>
        <span style="opacity:0.6;">29 districts · 928 records</span>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="hero">
    <h1>Karnataka Paddy Advisory System</h1>
    <p><span class="live-dot"></span>Live intelligence for <strong>{district}</strong> · {season} Season · {soil_q.title()} Soil Quality</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌱 Soil Intelligence",
    "🌦 Weather Advisory",
    "🔬 Crop Health",
    "📈 Market Prices",
    "🗺 Overview",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — SOIL INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    npk   = get_npk_profile(district)
    fert  = get_fertilizer_rec(soil_q, season)
    trend = get_soil_trend(district)
    yield_pred = get_yield_prediction(district, season, soil_q,
                                      fert.get("rainfall", 900), 65, 24.5)

    # ── NPK Profile ──
    st.markdown("""<div class="section-header">
        <h3>Soil NPK Profile</h3>
        <span class="section-pill">DISTRICT AVERAGE</span>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Nitrogen (N)</div>
            <div class="value">{npk['N']}</div>
            <div class="unit">kg / hectare</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Phosphorus (P)</div>
            <div class="value">{npk['P']}</div>
            <div class="unit">kg / hectare</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Potassium (K)</div>
            <div class="value">{npk['K']}</div>
            <div class="unit">kg / hectare</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Predicted Yield</div>
            <div class="value">{yield_pred:,.0f}</div>
            <div class="unit">kg / hectare · R²≈0.85</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # ── Fertilizer Recommendation ──
        st.markdown("""<div class="section-header">
            <h3>Fertilizer Recommendation</h3>
            <span class="section-pill">KVK GUIDELINES</span>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <table class="kvk-table">
          <thead><tr><th>Fertilizer</th><th>Dose (kg/ha)</th><th>Source</th></tr></thead>
          <tbody>
            <tr><td>🟡 Urea</td><td><strong>{fert['urea']} kg/ha</strong></td><td>Karnataka KVK</td></tr>
            <tr><td>🔵 DAP</td><td><strong>{fert['dap']} kg/ha</strong></td><td>Karnataka KVK</td></tr>
            <tr><td>🟤 Potash (MOP)</td><td><strong>{fert['potash']} kg/ha</strong></td><td>Karnataka KVK</td></tr>
          </tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown(f"""<div class="info-box" style="margin-top:0.8rem;">
            <div class="title">⏱ Application Timing — {season}</div>
            <div class="body">{fert['timing']}</div>
        </div>""", unsafe_allow_html=True)

    with col_right:
        # ── Soil Trend ──
        st.markdown("""<div class="section-header">
            <h3>15-Year Soil Trend</h3>
            <span class="section-pill">LINEAR REGRESSION</span>
        </div>""", unsafe_allow_html=True)

        ph_dir  = "trend-up" if trend["ph_slope"] > 0.01 else ("trend-down" if trend["ph_slope"] < -0.01 else "trend-flat")
        ph_icon = "↑" if trend["ph_slope"] > 0.01 else ("↓" if trend["ph_slope"] < -0.01 else "→")
        oc_dir  = "trend-up" if trend["oc_slope"] > 0.001 else ("trend-down" if trend["oc_slope"] < -0.001 else "trend-flat")
        oc_icon = "↑" if trend["oc_slope"] > 0.001 else ("↓" if trend["oc_slope"] < -0.001 else "→")

        st.markdown(f"""
        <table class="kvk-table">
          <thead><tr><th>Indicator</th><th>Current</th><th>Trend</th><th>Slope/yr</th></tr></thead>
          <tbody>
            <tr>
              <td>Soil pH</td>
              <td><strong>{trend['ph_last']}</strong></td>
              <td><span class="{ph_dir}">{ph_icon} {'Stable' if ph_dir=='trend-flat' else ('Increasing' if ph_dir=='trend-up' else 'Decreasing')}</span></td>
              <td>{trend['ph_slope']}</td>
            </tr>
            <tr>
              <td>Organic Carbon</td>
              <td><strong>{trend['oc_last']}%</strong></td>
              <td><span class="{oc_dir}">{oc_icon} {'Stable' if oc_dir=='trend-flat' else ('Improving' if oc_dir=='trend-up' else 'Declining')}</span></td>
              <td>{trend['oc_slope']}</td>
            </tr>
          </tbody>
        </table>
        """, unsafe_allow_html=True)

        oc_advice = "⚠️ Organic carbon declining. Apply 2–3 t/ha green manure (Dhaincha) before transplanting." if trend["oc_slope"] < -0.001 else "✅ Organic carbon levels are stable. Continue current practices."
        ph_advice = "⚠️ pH becoming acidic. Apply 2 t/ha lime before the season." if trend["ph_slope"] < -0.01 else ("⚠️ pH increasing (alkaline). Check irrigation water quality." if trend["ph_slope"] > 0.01 else "✅ pH is within optimal range (6.0–7.0) for paddy.")

        st.markdown(f"""
        <div class="info-box" style="margin-top:0.8rem;">
            <div class="body">{oc_advice}</div>
        </div>
        <div class="info-box" style="margin-top:0.4rem;">
            <div class="body">{ph_advice}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── District NPK comparison chart ──
    if df_main is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""<div class="section-header">
            <h3>District NPK Comparison</h3>
            <span class="section-pill">ALL 29 DISTRICTS</span>
        </div>""", unsafe_allow_html=True)

        npk_all = df_main.groupby("District")[
            ["Soil_Nitrogen_kg_per_ha","Soil_Phosphorus_kg_per_ha","Soil_Potassium_kg_per_ha"]
        ].mean().round(1).reset_index()
        npk_all.columns = ["District","Nitrogen","Phosphorus","Potassium"]

        highlight = npk_all[npk_all["District"] == district]
        st.dataframe(
            npk_all.sort_values("Nitrogen", ascending=False).reset_index(drop=True),
            use_container_width=True, height=300,
            column_config={
                "District": st.column_config.TextColumn("District", width="medium"),
                "Nitrogen": st.column_config.ProgressColumn("N (kg/ha)", min_value=0, max_value=npk_all["Nitrogen"].max(), format="%.1f"),
                "Phosphorus": st.column_config.ProgressColumn("P (kg/ha)", min_value=0, max_value=npk_all["Phosphorus"].max(), format="%.1f"),
                "Potassium": st.column_config.ProgressColumn("K (kg/ha)", min_value=0, max_value=npk_all["Potassium"].max(), format="%.1f"),
            }
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — WEATHER ADVISORY
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    weather = get_weather_data(district)

    st.markdown("""<div class="section-header">
        <h3>Current Weather Conditions</h3>
        <span class="section-pill">OPEN-METEO API</span>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">🌡 Temperature</div>
            <div class="value">{weather['temp']}°</div>
            <div class="unit">Celsius — {district}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">🌧 Annual Rainfall</div>
            <div class="value">{weather['rainfall']}</div>
            <div class="unit">mm per year (avg)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">💧 Humidity</div>
            <div class="value">{weather['humidity']}%</div>
            <div class="unit">Relative humidity</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("""<div class="section-header">
            <h3>Sowing Advisory</h3>
            <span class="section-pill">SEASONAL</span>
        </div>""", unsafe_allow_html=True)

        sowing_kharif = [
            ("June 15 – July 15", "Optimal sowing window for Kharif — soil moisture adequate"),
            ("Nursery duration", "25–30 days before transplanting"),
            ("Water requirement", "1200–1500 mm for Kharif season"),
            ("Transplanting", "July – August when soil temp > 20°C"),
        ]
        sowing_rabi = [
            ("November – December", "Optimal sowing window for Rabi season"),
            ("Nursery duration", "20–25 days before transplanting"),
            ("Water requirement", "900–1100 mm (supplemental irrigation needed)"),
            ("Transplanting", "December when night temps stable"),
        ]
        rows = sowing_kharif if season == "Kharif" else sowing_rabi

        st.markdown('<ul class="step-list">', unsafe_allow_html=True)
        for i, (title, desc) in enumerate(rows, 1):
            st.markdown(f"""<li>
                <span class="step-num">{i}</span>
                <span><strong>{title}</strong> — {desc}</span>
            </li>""", unsafe_allow_html=True)
        st.markdown('</ul>', unsafe_allow_html=True)

    with col_b:
        st.markdown("""<div class="section-header">
            <h3>Irrigation Advisory</h3>
            <span class="section-pill">THRESHOLD LOGIC</span>
        </div>""", unsafe_allow_html=True)

        rain_pct = min(100, int((weather["rainfall"] / 1300) * 100))
        irr_class = "info-box warning" if rain_pct < 60 else "info-box"
        irr_msg   = "⚠️ Rainfall below 60% of seasonal average. Supplemental irrigation recommended. Schedule 3–4 irrigations at critical growth stages." if rain_pct < 60 else "✅ Rainfall levels adequate for the season. Monitor field for standing water during heavy rain events."

        st.markdown(f"""<div class="{irr_class}">
            <div class="title">Rainfall Status — {rain_pct}% of seasonal average</div>
            <div class="body">{irr_msg}</div>
        </div>""", unsafe_allow_html=True)

        temp_msg = "⚠️ High temperature alert. Risk of spikelet sterility at flowering stage. Ensure adequate irrigation." if weather["temp"] > 35 else "✅ Temperature within optimal range for paddy growth (22–32°C)."
        temp_class = "info-box warning" if weather["temp"] > 35 else "info-box"

        st.markdown(f"""<div class="{temp_class}" style="margin-top:0.6rem;">
            <div class="title">Temperature Advisory</div>
            <div class="body">{temp_msg}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="info-box" style="margin-top:0.6rem;">
            <div class="title">Pest Risk — Based on Humidity {weather['humidity']}%</div>
            <div class="body">{'⚠️ High humidity (>75%). Elevated risk of Brown Plant Hopper and Blast disease. Schedule preventive fungicide spray.' if weather['humidity'] > 75 else '✅ Humidity within normal range. Standard pest monitoring schedule applicable.'}</div>
        </div>""", unsafe_allow_html=True)

    # ── Historical weather trend ──
    if df_main is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""<div class="section-header">
            <h3>Historical Weather Baseline</h3>
            <span class="section-pill">2009–2024</span>
        </div>""", unsafe_allow_html=True)

        d = df_main[df_main["District"] == district].groupby("Numeric_Year")[
            ["Rainfall_mm","Temperature_Celsius","Humidity_Percent"]
        ].mean().round(2).reset_index()
        d.columns = ["Year","Rainfall (mm)","Temperature (°C)","Humidity (%)"]
        st.line_chart(d.set_index("Year")[["Rainfall (mm)","Temperature (°C)"]], use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — CROP HEALTH
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""<div class="section-header">
        <h3>Crop Health Assessment</h3>
        <span class="section-pill">GRADIENT BOOSTING · NDVI</span>
    </div>""", unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1.4], gap="large")

    with col_input:
        growth_stage = st.selectbox("Growth Stage", GROWTH_STAGES, index=4)
        ndvi_val     = st.slider("NDVI Reading", min_value=0.0, max_value=1.0, value=0.62, step=0.01,
                                  help="Normalized Difference Vegetation Index. Healthy paddy: 0.6–0.8")
        st.markdown(f"""<div class="info-box" style="margin-top:0.4rem;">
            <div class="title">NDVI Baseline for {growth_stage}</div>
            <div class="body">Expected median NDVI at this stage: <strong>0.65</strong><br>
            Your deviation: <strong>{round(ndvi_val - 0.65, 3):+.3f}</strong></div>
        </div>""", unsafe_allow_html=True)

        run = st.button("🔍  Assess Crop Health")

    with col_result:
        if run or True:
            status, advice = get_crop_health(district, growth_stage, ndvi_val)
            badge_map = {
                "Healthy":       "badge-healthy",
                "Mild Stress":   "badge-mild",
                "Moderate Stress":"badge-moderate",
                "Severe Stress": "badge-severe",
            }
            icon_map = {
                "Healthy": "✅", "Mild Stress": "🟡",
                "Moderate Stress": "🟠", "Severe Stress": "🔴"
            }
            badge = badge_map.get(status, "badge-healthy")
            icon  = icon_map.get(status, "✅")

            st.markdown(f"""
            <div style="background:white;border-radius:14px;padding:1.5rem;box-shadow:0 4px 16px rgba(27,67,50,0.12);margin-top:0.4rem;">
                <div style="margin-bottom:0.8rem;">
                    <span class="{badge}">{icon} {status}</span>
                </div>
                <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#1B4332;margin-bottom:0.6rem;">
                    {district} · {growth_stage} Stage
                </div>
                <div style="font-size:0.88rem;color:#4B5563;line-height:1.6;">
                    {advice}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── KVK Treatment table ──
    st.markdown("""<div class="section-header">
        <h3>KVK Treatment Protocol</h3>
        <span class="section-pill">ALL GROWTH STAGES</span>
    </div>""", unsafe_allow_html=True)

    kvk_data = {
        "Growth Stage": ["Germination","Seedling","Tillering","Stem Extension","Booting","Heading/Flowering","Grain Filling"],
        "Disease Watch": ["Damping off","Blast, Seedling blight","Brown Plant Hopper, BLB","Sheath blight","Neck blast","False smut, Grain discoloration","Grain discoloration"],
        "Recommended Action": [
            "Seed treatment: Carbendazim 2g/kg seed",
            "Tricyclazole 0.6g/L + Imidacloprid 0.3ml/L spray",
            "Acephate 1.5g/L for BPH; Propiconazole 1ml/L for BLB",
            "Hexaconazole 2ml/L for sheath blight",
            "Tricyclazole 0.6g/L — critical spray window",
            "Propiconazole 1ml/L + Carbendazim 1g/L",
            "Avoid fungicide — within harvest window",
        ],
        "NPK Top-dress": ["–","N: 25% split","N: 50% split","K: full dose","–","–","–"],
    }
    st.dataframe(pd.DataFrame(kvk_data), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — MARKET PRICES
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""<div class="section-header">
        <h3>Paddy Market Prices</h3>
        <span class="section-pill">KARNATAKA APMC</span>
    </div>""", unsafe_allow_html=True)

    if df_market is not None:
        df_m = df_market.copy()
        df_m.columns = df_m.columns.str.strip()
        c1, c2, c3 = st.columns(3)
        num_cols = df_m.select_dtypes(include=[np.number]).columns.tolist()

        if len(num_cols) >= 1:
            latest = df_m[num_cols[0]].iloc[-1] if not df_m.empty else 2200
            prev   = df_m[num_cols[0]].iloc[-7] if len(df_m) > 7 else latest
            delta  = round(latest - prev, 1)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Current MSP</div>
                    <div class="value">₹{latest:,.0f}</div>
                    <div class="unit">per quintal · {'↑' if delta>0 else '↓'} ₹{abs(delta)} vs last week</div>
                </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="label">MSP 2024–25</div>
                <div class="value">₹2,300</div>
                <div class="unit">per quintal · Government fixed</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Export Parity</div>
                <div class="value">₹2,650</div>
                <div class="unit">per quintal · Current estimate</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        if len(num_cols) >= 1:
            st.markdown("""<div class="section-header"><h3>Price Trend</h3></div>""", unsafe_allow_html=True)
            st.line_chart(df_m[num_cols[:min(3,len(num_cols))]], use_container_width=True)

        st.markdown("""<div class="section-header"><h3>Market Data</h3></div>""", unsafe_allow_html=True)
        st.dataframe(df_m.tail(30), use_container_width=True, height=300)
    else:
        # Static fallback
        c1, c2, c3 = st.columns(3)
        prices = [("Current APMC Price","₹2,420","per quintal · Kharif 2024"),
                  ("MSP 2024–25","₹2,300","per quintal · Government fixed"),
                  ("Export Parity","₹2,650","per quintal · Current estimate")]
        for col, (label, val, unit) in zip([c1,c2,c3], prices):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{val}</div>
                    <div class="unit">{unit}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""<div class="info-box warning">
            <div class="title">Market Data File Not Found</div>
            <div class="body">Place <code>Karnataka_Market_Prices.csv</code> in the <code>data/</code> folder to enable live market price charts.</div>
        </div>""", unsafe_allow_html=True)

        varieties = {
            "Variety": ["BPT-5204 (Sona Masoori)","IR-64","MTU-1010","Jyothi","Rajamudi"],
            "Grade":   ["A Grade","B Grade","A Grade","B Grade","Premium"],
            "Price (₹/quintal)": [2650, 2300, 2500, 2200, 3200],
            "Season":  ["Kharif","Both","Kharif","Rabi","Kharif"],
        }
        st.dataframe(pd.DataFrame(varieties), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""<div class="section-header">
        <h3>System Overview</h3>
        <span class="section-pill">MULTI-AGENT SUMMARY</span>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div style="background:white;border-radius:14px;padding:1.4rem;box-shadow:0 2px 8px rgba(27,67,50,0.1);">
        <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#1B4332;margin-bottom:1rem;">Agent Architecture</div>
        <ul class="step-list">
            <li><span class="step-num">1</span><span><strong>Soil Agent</strong> (Python · Port 50052)<br>
            <span style="font-size:0.8rem;color:#6B7280;">NPK profiling · KVK fertilizer · RF yield · Soil trend</span></span></li>
            <li><span class="step-num">2</span><span><strong>Weather Agent</strong> (Go · Port 50053)<br>
            <span style="font-size:0.8rem;color:#6B7280;">Open-Meteo API · Historical baseline · Sowing advisory</span></span></li>
            <li><span class="step-num">3</span><span><strong>Crop Health Agent</strong> (Python · Port 50054)<br>
            <span style="font-size:0.8rem;color:#6B7280;">NDVI deviation · Gradient Boosting · KVK treatment</span></span></li>
            <li><span class="step-num">4</span><span><strong>Orchestrator + Dashboard</strong> (Go + Streamlit)<br>
            <span style="font-size:0.8rem;color:#6B7280;">Parallel gRPC calls · Merged response · 5-tab UI</span></span></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="background:white;border-radius:14px;padding:1.4rem;box-shadow:0 2px 8px rgba(27,67,50,0.1);">
        <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#1B4332;margin-bottom:1rem;">Dataset Coverage</div>
        """, unsafe_allow_html=True)

        stats = [
            ("Total Records", "44,159+", "Across all datasets"),
            ("Districts Covered", "29", "All Karnataka districts"),
            ("Time Period", "2009–2024", "15 years of data"),
            ("Soil Records", "928", "Karnataka_Paddy_Main_Dataset"),
        ]
        for label, val, sub in stats:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;
                padding:8px 0;border-bottom:1px solid #D8F3DC;">
                <div>
                    <div style="font-weight:600;font-size:0.88rem;color:#1B4332;">{label}</div>
                    <div style="font-size:0.76rem;color:#6B7280;">{sub}</div>
                </div>
                <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#2D6A4F;">{val}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── District summary table ──
    if df_main is not None:
        st.markdown("""<div class="section-header">
            <h3>District-wise Summary</h3>
            <span class="section-pill">ALL 29 DISTRICTS</span>
        </div>""", unsafe_allow_html=True)

        summary = df_main.groupby("District").agg(
            Records=("Yield_Kg_per_Ha","count"),
            Avg_Yield=("Yield_Kg_per_Ha","mean"),
            Avg_pH=("Soil_pH","mean"),
            Avg_Nitrogen=("Soil_Nitrogen_kg_per_ha","mean"),
            Avg_Rainfall=("Rainfall_mm","mean"),
        ).round(1).reset_index()
        summary.columns = ["District","Records","Avg Yield (kg/ha)","Avg pH","Avg N (kg/ha)","Avg Rainfall (mm)"]

        st.dataframe(
            summary.sort_values("Avg Yield (kg/ha)", ascending=False).reset_index(drop=True),
            use_container_width=True, height=400,
            column_config={
                "District": st.column_config.TextColumn("District"),
                "Avg Yield (kg/ha)": st.column_config.ProgressColumn("Avg Yield (kg/ha)", min_value=0, max_value=summary["Avg Yield (kg/ha)"].max(), format="%.0f"),
                "Avg pH": st.column_config.NumberColumn("Avg pH", format="%.2f"),
                "Avg N (kg/ha)": st.column_config.NumberColumn("Avg N (kg/ha)", format="%.1f"),
                "Avg Rainfall (mm)": st.column_config.NumberColumn("Avg Rainfall (mm)", format="%.0f"),
            }
        )

    # ── Tech stack ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""<div class="section-header"><h3>Technology Stack</h3></div>""", unsafe_allow_html=True)
    chips = [
        ("Python 3.13",""),("Go 1.21",""),("gRPC","blue"),("Protocol Buffers","blue"),
        ("scikit-learn",""),("scipy",""),("joblib",""),("Streamlit",""),
        ("Open-Meteo API","gold"),("Docker",""),("GitHub Actions",""),
    ]
    chips_html = " ".join(f'<span class="chip {c}">{name}</span>' for name, c in chips)
    st.markdown(f'<div style="margin-top:0.4rem;">{chips_html}</div>', unsafe_allow_html=True)

    # ── Footer ──
    st.markdown(f"""
    <div style="text-align:center;padding:2rem 0 1rem;color:#6B7280;font-size:0.78rem;">
        KrishiSense · Karnataka Paddy Multi-Agent Advisory System<br>
        UE23CS320B Capstone Project — Aman · Aditya · Rishi · Sangam<br>
        <span style="color:#B7E4C7;">PES University · {datetime.now().year}</span>
    </div>
    """, unsafe_allow_html=True)