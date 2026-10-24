# streamlit_dashboard/app.py
# Phase 1 Dashboard — Karnataka Paddy Multi-Agent System
# Connects to all 3 gRPC agents and displays real-time data

import os
import json
import time
import grpc
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

import paddy_agents_pb2 as pb
import paddy_agents_pb2_grpc as pb_grpc

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Karnataka Paddy Multi-Agent System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f5f7f0; }
    .stMetric { background: white; border-radius: 10px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .agent-card { background: white; border-radius: 12px; padding: 15px; margin: 8px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .status-ok  { color: #2ecc71; font-weight: bold; }
    .status-err { color: #e74c3c; font-weight: bold; }
    h1 { color: #2c3e50; }
    .phase-badge {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white; padding: 4px 12px; border-radius: 20px;
        font-size: 0.85em; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# gRPC CLIENT CONNECTIONS
# ─────────────────────────────────────────────

AGENT_ADDRS = {
    "weather":     os.environ.get("WEATHER_AGENT_ADDR",     "localhost:50052"),
    "soil":        os.environ.get("SOIL_AGENT_ADDR",        "localhost:50053"),
    "crop_health": os.environ.get("CROP_HEALTH_AGENT_ADDR", "localhost:50054"),
    "orchestrator":os.environ.get("ORCHESTRATOR_ADDR",      "localhost:50051"),
}

@st.cache_resource
def get_channel(addr: str):
    return grpc.insecure_channel(addr)

def weather_client():
    return pb_grpc.WeatherAgentServiceStub(get_channel(AGENT_ADDRS["weather"]))

def soil_client():
    return pb_grpc.SoilAgentServiceStub(get_channel(AGENT_ADDRS["soil"]))

def crop_client():
    return pb_grpc.CropHealthAgentServiceStub(get_channel(AGENT_ADDRS["crop_health"]))

def orch_client():
    return pb_grpc.OrchestratorServiceStub(get_channel(AGENT_ADDRS["orchestrator"]))

# ─────────────────────────────────────────────
# DATA LOADING (local CSVs for dashboard charts)
# ─────────────────────────────────────────────

@st.cache_data
def load_weather_df():
    return pd.read_csv("/data/Karnataka_Weather_Daily.csv", parse_dates=["Date"])

@st.cache_data
def load_main_df():
    return pd.read_csv("/data/Karnataka_Paddy_Main_Dataset.csv")

@st.cache_data
def load_market_df():
    return pd.read_csv("/data/Karnataka_Market_Prices.csv", parse_dates=["Date"])

@st.cache_data
def load_satellite_df():
    import openpyxl
    from openpyxl import load_workbook
    df = pd.read_excel("/data/Karnataka_Paddy_AI_ML_Dataset.xlsx",
                       sheet_name="Satellite_Indices")
    return df

# ─────────────────────────────────────────────
# HELPER: safe gRPC call
# ─────────────────────────────────────────────

def safe_grpc(fn, *args, **kwargs):
    try:
        return fn(*args, timeout=8, **kwargs), None
    except grpc.RpcError as e:
        return None, f"gRPC error: {e.details()}"
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Karnataka_state_symbol.png/240px-Karnataka_state_symbol.png",
             width=80)
    st.markdown("## 🌾 Paddy MAS")
    st.markdown('<span class="phase-badge">Phase 1 — 40% Complete</span>', unsafe_allow_html=True)
    st.markdown("---")

    DISTRICTS = [
        "Koppal", "Ballari", "Raichur", "Davanagere",
        "Shivamogga", "Hassan", "Mandya", "Mysuru",
        "Belagavi", "Dharwad",
    ]
    selected_district = st.selectbox("📍 District", DISTRICTS)
    selected_season   = st.radio("🌦 Season", ["Kharif", "Rabi"])
    selected_variety  = st.selectbox("🌱 Seed Variety",
        ["BPT-5204", "MTU-1010", "JGL-1798", "Rasi", "IR-64"])
    das_input = st.slider("📅 Days After Sowing", 0, 140, 45)
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)

    st.markdown("---")
    st.markdown("**Agent Status**")

    # Check orchestrator
    orch_resp, err = safe_grpc(orch_client().GetSystemStatus,
                               pb.StatusRequest(requester="dashboard"))
    if orch_resp:
        st.success(f"🟢 Orchestrator: HEALTHY")
        for a in orch_resp.active_agents:
            st.markdown(f"  └ 🟢 {a}")
        st.markdown(f"  Queries handled: **{orch_resp.total_queries_handled}**")
    else:
        st.warning("🟡 Orchestrator: Offline (demo mode)")

    st.markdown("---")
    st.markdown("**Owners**")
    st.markdown("🔵 **Aditya** — Weather Agent")
    st.markdown("🟢 **Aman** — Soil Agent")
    st.markdown("🟠 **Rishi Mohan** — Crop Health")

if auto_refresh:
    time.sleep(30)
    st.rerun()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
# 🌾 Karnataka Paddy Multi-Agent System
### Phase 1 Dashboard — Weather · Soil · Crop Health
""")

col_h1, col_h2, col_h3, col_h4 = st.columns(4)
with col_h1:
    st.metric("Total Records", "44,159", "↑ 15 years")
with col_h2:
    st.metric("Districts Tracked", "10", "Weather stations")
with col_h3:
    st.metric("Active Agents", "3", "Phase 1")
with col_h4:
    st.metric("Data Sources", "5 datasets", "CSV + XLSX")

st.markdown("---")

# ─────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌤 Weather Agent (Aditya)",
    "🌱 Soil Agent (Aman)",
    "🛰 Crop Health (Rishi Mohan)",
    "📊 Data Analytics",
    "🔀 Orchestrator",
])

# ══════════════════════════════════════════════
# TAB 1: WEATHER (Aditya)
# ══════════════════════════════════════════════

with tab1:
    st.subheader(f"🌤 Weather Intelligence — {selected_district}")
    st.caption("Owner: **Aditya** | gRPC port 50052 | Data: Karnataka_Weather_Daily.csv (36,530 records)")

    col_w1, col_w2 = st.columns([1, 2])

    with col_w1:
        # Live gRPC call
        w_resp, w_err = safe_grpc(weather_client().GetWeatherForecast,
            pb.WeatherRequest(district=selected_district,
                              date=datetime.now().strftime("%Y-%m-%d"),
                              forecast_days=7))

        if w_resp:
            st.success("✅ Agent responding")
            st.metric("🌡 Max Temp", f"{w_resp.temperature_max:.1f}°C")
            st.metric("🌡 Min Temp", f"{w_resp.temperature_min:.1f}°C")
            st.metric("🌧 Rainfall",  f"{w_resp.rainfall_mm:.1f} mm")
            st.metric("💧 Humidity",  f"{w_resp.humidity_percent:.0f}%")
            st.metric("💨 Wind",      f"{w_resp.wind_speed_kmph:.1f} km/h")
            st.metric("☀ Sunshine",   f"{w_resp.sunshine_hours:.1f} hrs")
            st.info(f"**Condition:** {w_resp.weather_condition}")
            st.warning(f"**Advisory:** {w_resp.farming_advisory}")

            # Alert
            a_resp, _ = safe_grpc(weather_client().GetWeatherAlert,
                pb.WeatherRequest(district=selected_district,
                                  date=datetime.now().strftime("%Y-%m-%d")))
            if a_resp and a_resp.has_alert:
                st.error(f"🚨 **{a_resp.alert_type}** ({a_resp.severity}): {a_resp.message}")
                st.warning(f"Action: {a_resp.action_required}")
        else:
            st.warning(f"Agent offline — using local data ({w_err})")

    with col_w2:
        # Historical weather chart from CSV
        try:
            wdf = load_weather_df()
            dist_w = wdf[wdf["District"] == selected_district].copy()
            dist_w["Month"] = dist_w["Date"].dt.to_period("M").dt.to_timestamp()
            monthly = dist_w.groupby("Month").agg({
                "Temperature_Avg_C": "mean",
                "Rainfall_mm": "sum",
            }).reset_index().tail(36)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly["Month"], y=monthly["Rainfall_mm"],
                name="Monthly Rainfall (mm)", marker_color="#3498db", yaxis="y2",
            ))
            fig.add_trace(go.Scatter(
                x=monthly["Month"], y=monthly["Temperature_Avg_C"],
                name="Avg Temp (°C)", line=dict(color="#e74c3c", width=2),
            ))
            fig.update_layout(
                title=f"Temperature & Rainfall — {selected_district} (3 years)",
                yaxis=dict(title="Temperature (°C)", side="left"),
                yaxis2=dict(title="Rainfall (mm)", overlaying="y", side="right"),
                legend=dict(x=0, y=1.1, orientation="h"),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            # 7-day forecast chart from gRPC
            if w_resp and w_resp.forecast:
                fc = w_resp.forecast
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=[f.date for f in fc],
                    y=[f.temp_max for f in fc],
                    name="Max Temp", line=dict(color="#e74c3c"),
                ))
                fig_fc.add_trace(go.Scatter(
                    x=[f.date for f in fc],
                    y=[f.temp_min for f in fc],
                    name="Min Temp", fill="tonexty", line=dict(color="#3498db"),
                ))
                fig_fc.update_layout(
                    title="7-Day Temperature Forecast", height=250,
                    yaxis_title="Temperature (°C)",
                )
                st.plotly_chart(fig_fc, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

# ══════════════════════════════════════════════
# TAB 2: SOIL (Aman)
# ══════════════════════════════════════════════

with tab2:
    st.subheader(f"🌱 Soil Health Analysis — {selected_district}")
    st.caption("Owner: **Aman** | gRPC port 50053 | Data: Karnataka_Paddy_Main_Dataset.csv (928 records)")

    col_s1, col_s2 = st.columns([1, 2])

    with col_s1:
        s_resp, s_err = safe_grpc(soil_client().GetSoilHealth,
            pb.SoilRequest(district=selected_district, season=selected_season))

        if s_resp:
            st.success("✅ Agent responding")
            health_color = {
                "EXCELLENT": "success", "GOOD": "info",
                "MODERATE": "warning", "POOR": "error"
            }.get(s_resp.health_status, "info")
            getattr(st, health_color)(f"Soil Status: **{s_resp.health_status}**")
            st.metric("🧪 Soil pH",       f"{s_resp.soil_ph:.2f}")
            st.metric("🌿 Organic Carbon", f"{s_resp.organic_carbon:.2f}%")
            st.metric("🔵 Nitrogen",       f"{s_resp.nitrogen_kg_ha:.1f} kg/ha")
            st.metric("🟡 Phosphorus",     f"{s_resp.phosphorus_kg_ha:.1f} kg/ha")
            st.metric("🟠 Potassium",      f"{s_resp.potassium_kg_ha:.1f} kg/ha")
            st.info(f"**Recommendation:** {s_resp.recommendation}")
        else:
            st.warning(f"Agent offline — loading from dataset ({s_err})")

        # Fertilizer advice
        f_resp, _ = safe_grpc(soil_client().GetFertilizerAdvice,
            pb.SoilRequest(district=selected_district, season=selected_season))
        if f_resp:
            st.subheader("💊 Fertilizer Recommendation")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.metric("Urea",   f"{f_resp.urea_kg_ha:.0f} kg/ha")
                st.metric("DAP",    f"{f_resp.dap_kg_ha:.0f} kg/ha")
            with col_f2:
                st.metric("Potash", f"{f_resp.potash_kg_ha:.0f} kg/ha")
                st.metric("Organic",f"{f_resp.organic_kg_ha:.0f} kg/ha")
            st.caption(f"⏰ {f_resp.timing_advice}")

    with col_s2:
        try:
            mdf = load_main_df()

            # Soil health radar chart
            dist_soil = mdf[mdf["District"] == selected_district]
            if dist_soil.empty:
                dist_soil = mdf

            avg_vals = dist_soil[[
                "Soil_pH", "Soil_Organic_Carbon_Percent",
                "Soil_Nitrogen_kg_per_ha", "Soil_Phosphorus_kg_per_ha",
                "Soil_Potassium_kg_per_ha",
            ]].mean()

            # Normalize to 0–10 scale for radar
            norm = {
                "pH (×1.4)":        avg_vals["Soil_pH"] * 1.4,
                "Org Carbon (×10)": avg_vals["Soil_Organic_Carbon_Percent"] * 10,
                "Nitrogen (÷30)":   avg_vals["Soil_Nitrogen_kg_per_ha"] / 30,
                "Phosphorus (÷3)":  avg_vals["Soil_Phosphorus_kg_per_ha"] / 3,
                "Potassium (÷20)":  avg_vals["Soil_Potassium_kg_per_ha"] / 20,
            }
            cats   = list(norm.keys())
            values = list(norm.values())

            fig_radar = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=cats + [cats[0]],
                fill="toself",
                fillcolor="rgba(46, 204, 113, 0.3)",
                line=dict(color="#27ae60"),
                name=selected_district,
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(range=[0, 12])),
                title=f"Soil Nutrient Profile — {selected_district}",
                height=380,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Soil quality by district
            sq_dist = mdf.groupby(["District", "Soil_Quality"])["Yield_Kg_per_Ha"].mean().reset_index()
            fig_sq = px.box(
                mdf[mdf["District"].isin(DISTRICTS[:5])],
                x="District", y="Yield_Kg_per_Ha", color="Soil_Quality",
                title="Yield Distribution by Soil Quality (Top 5 Districts)",
                height=300,
            )
            st.plotly_chart(fig_sq, use_container_width=True)

        except Exception as e:
            st.error(f"Chart error: {e}")

# ══════════════════════════════════════════════
# TAB 3: CROP HEALTH (Rishi Mohan)
# ══════════════════════════════════════════════

with tab3:
    st.subheader(f"🛰 Crop Health & NDVI — {selected_district}")
    st.caption("Owner: **Rishi Mohan** | gRPC port 50054 | Data: AI_ML_Dataset (Satellite_Indices + Crop_Growth_Stages)")

    col_c1, col_c2 = st.columns([1, 2])

    with col_c1:
        # Crop health status
        ch_resp, ch_err = safe_grpc(crop_client().GetCropHealthStatus,
            pb.CropHealthRequest(
                district=selected_district,
                season=selected_season,
                year=datetime.now().year,
                variety=selected_variety,
            ))

        if ch_resp:
            st.success("✅ Agent responding")
            health_emoji = {
                "EXCELLENT": "🟢", "GOOD": "🔵",
                "MODERATE": "🟡", "POOR": "🔴"
            }.get(ch_resp.health_status, "⚪")
            st.metric("🌱 Growth Stage", ch_resp.growth_stage)
            st.metric("📅 Days After Sowing", ch_resp.days_after_sowing)
            st.metric("📏 Plant Height", f"{ch_resp.plant_height_cm:.1f} cm")
            st.metric("🌿 Tiller Count", ch_resp.tiller_count)
            st.metric("📐 LAI", f"{ch_resp.leaf_area_index:.2f}")
            st.metric("🛰 NDVI", f"{ch_resp.ndvi:.3f}")
            st.info(f"{health_emoji} Health: **{ch_resp.health_status}**")
            st.warning(f"Action: {ch_resp.action_required}")
        else:
            st.warning(f"Agent offline ({ch_err})")

        # NDVI from API
        ndvi_resp, _ = safe_grpc(crop_client().GetNDVIAnalysis,
            pb.NDVIRequest(
                district=selected_district,
                year=datetime.now().year,
                month=datetime.now().month,
            ))
        if ndvi_resp:
            st.subheader("🛰 Satellite NDVI")
            st.metric("NDVI Mean",   f"{ndvi_resp.ndvi_mean:.3f}")
            st.metric("EVI Mean",    f"{ndvi_resp.evi_mean:.3f}")
            st.metric("Soil Moisture", f"{ndvi_resp.soil_moisture:.1f}%")
            st.caption(f"Source: **{ndvi_resp.source}** | {ndvi_resp.vegetation_status}")

        # Pest/disease risk
        pd_resp, _ = safe_grpc(crop_client().GetPestDiseaseRisk,
            pb.PestDiseaseRequest(
                district=selected_district,
                growth_stage=ch_resp.growth_stage if ch_resp else "Vegetative",
                season=selected_season,
                humidity=75.0, temperature=28.0,
            ))
        if pd_resp:
            st.subheader("🐛 Pest & Disease Risk")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                color = "🔴" if pd_resp.pest_risk_level == "HIGH" else "🟡"
                st.metric(f"{color} Pest Risk", pd_resp.pest_risk_level)
                for p in pd_resp.likely_pests:
                    st.markdown(f"  - {p}")
            with col_p2:
                color = "🔴" if pd_resp.disease_risk_level == "HIGH" else "🟡"
                st.metric(f"{color} Disease Risk", pd_resp.disease_risk_level)
                for d in pd_resp.likely_diseases:
                    st.markdown(f"  - {d}")
            st.info(pd_resp.preventive_action)

    with col_c2:
        try:
            sat_df = load_satellite_df()

            # NDVI over time for selected district
            dist_sat = sat_df[sat_df["District"] == selected_district].copy()
            if dist_sat.empty:
                dist_sat = sat_df.groupby(["Year", "Month"]).mean(numeric_only=True).reset_index()
                dist_sat["District"] = "State Average"

            dist_sat["Period"] = pd.to_datetime(
                dist_sat["Year"].astype(str) + "-" + dist_sat["Month"].astype(str).str.zfill(2) + "-01"
            )

            fig_ndvi = go.Figure()
            fig_ndvi.add_trace(go.Scatter(
                x=dist_sat["Period"], y=dist_sat["NDVI_Mean"],
                name="NDVI Mean", line=dict(color="#27ae60", width=2),
                fill="tozeroy", fillcolor="rgba(39, 174, 96, 0.15)",
            ))
            fig_ndvi.add_trace(go.Scatter(
                x=dist_sat["Period"], y=dist_sat["EVI_Mean"],
                name="EVI Mean", line=dict(color="#2980b9", width=1.5, dash="dot"),
            ))
            # NDVI threshold lines
            fig_ndvi.add_hline(y=0.7, line_dash="dash", line_color="green",
                               annotation_text="Healthy (>0.7)")
            fig_ndvi.add_hline(y=0.4, line_dash="dash", line_color="orange",
                               annotation_text="Stress (<0.4)")
            fig_ndvi.update_layout(
                title=f"NDVI & EVI Trend — {selected_district}",
                yaxis_title="Vegetation Index", height=300,
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_ndvi, use_container_width=True)

            # Growth stage timeline
            gs_df = pd.read_excel(
                "/data/Karnataka_Paddy_AI_ML_Dataset.xlsx",
                sheet_name="Crop_Growth_Stages"
            )
            dist_gs = gs_df[
                (gs_df["District"] == selected_district) &
                (gs_df["Season"] == selected_season)
            ]
            if dist_gs.empty:
                dist_gs = gs_df[gs_df["Season"] == selected_season].head(40)

            fig_gs = px.scatter(
                dist_gs, x="Days_After_Sowing", y="NDVI",
                color="Growth_Stage", size="Leaf_Area_Index",
                title=f"Growth Stage vs NDVI — {selected_season}",
                height=300,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_gs.add_vline(x=das_input, line_color="red",
                             annotation_text=f"Current DAS: {das_input}")
            st.plotly_chart(fig_gs, use_container_width=True)

            # Satellite soil moisture
            fig_sm = px.line(
                dist_sat, x="Period", y="Soil_Moisture_Percent",
                title="Satellite Soil Moisture %",
                height=250, color_discrete_sequence=["#8e44ad"],
            )
            fig_sm.add_hline(y=40, line_dash="dot", line_color="blue",
                             annotation_text="Optimal (40%)")
            st.plotly_chart(fig_sm, use_container_width=True)

        except Exception as e:
            st.error(f"Chart error: {e}")

# ══════════════════════════════════════════════
# TAB 4: DATA ANALYTICS
# ══════════════════════════════════════════════

with tab4:
    st.subheader("📊 Karnataka Paddy — Exploratory Analytics")

    try:
        mdf = load_main_df()
        wdf = load_weather_df()
        mkdf = load_market_df()

        col_a1, col_a2 = st.columns(2)

        with col_a1:
            # Yield by district
            yld_dist = mdf.groupby("District")["Yield_Kg_per_Ha"].mean().sort_values(ascending=True).reset_index()
            fig_yld = px.bar(
                yld_dist, x="Yield_Kg_per_Ha", y="District",
                orientation="h", title="Average Yield by District (2009–2024)",
                color="Yield_Kg_per_Ha",
                color_continuous_scale="Greens", height=450,
            )
            st.plotly_chart(fig_yld, use_container_width=True)

        with col_a2:
            # Yield trend over years
            yld_yr = mdf.groupby(["Numeric_Year", "Season"])["Yield_Kg_per_Ha"].mean().reset_index()
            fig_yr = px.line(
                yld_yr, x="Numeric_Year", y="Yield_Kg_per_Ha",
                color="Season", title="State Yield Trend by Season",
                markers=True, height=350,
            )
            st.plotly_chart(fig_yr, use_container_width=True)

            # Market price trend
            mkdf["Year"] = mkdf["Date"].dt.year
            mk_yr = mkdf.groupby(["Year", "Variety"])["Price_Rs_per_Quintal"].mean().reset_index()
            fig_mk = px.line(
                mk_yr, x="Year", y="Price_Rs_per_Quintal",
                color="Variety", title="Market Price by Variety (Rs/Quintal)",
                height=300,
            )
            st.plotly_chart(fig_mk, use_container_width=True)

        col_a3, col_a4 = st.columns(2)
        with col_a3:
            # Water management vs yield
            wm = mdf.groupby("Water_Management")["Yield_Kg_per_Ha"].agg(["mean", "std"]).reset_index()
            fig_wm = px.bar(
                wm, x="Water_Management", y="mean",
                error_y="std", title="Yield by Water Management",
                color="Water_Management",
                color_discrete_sequence=["#3498db", "#27ae60", "#9b59b6"],
                height=300,
            )
            st.plotly_chart(fig_wm, use_container_width=True)

        with col_a4:
            # Rainfall vs yield scatter
            fig_sc = px.scatter(
                mdf.sample(min(500, len(mdf))),
                x="Rainfall_mm", y="Yield_Kg_per_Ha",
                color="Season", size="Area_Hectares",
                title="Rainfall vs Yield (sample 500 records)",
                trendline="ols", height=300,
                opacity=0.6,
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    except Exception as e:
        st.error(f"Analytics error: {e}")

# ══════════════════════════════════════════════
# TAB 5: ORCHESTRATOR
# ══════════════════════════════════════════════

with tab5:
    st.subheader("🔀 Decision Support Orchestrator")
    st.caption("Go gRPC Orchestrator — concurrent fan-out to all agents")

    col_o1, col_o2 = st.columns([1, 1])

    with col_o1:
        st.markdown("### 🎯 Test Query Routing")
        query_type = st.selectbox("Query Type",
            ["FULL", "WEATHER", "SOIL", "CROP_HEALTH", "FERTILIZER"])
        farmer_id  = st.text_input("Farmer ID", "KA00001")

        if st.button("🚀 Route Query to Orchestrator", type="primary"):
            with st.spinner("Routing to agents concurrently..."):
                o_resp, o_err = safe_grpc(orch_client().RouteQuery,
                    pb.FarmerQuery(
                        query_id=f"Q{int(time.time())}",
                        farmer_id=farmer_id,
                        district=selected_district,
                        query_text="Full farm advisory",
                        query_type=query_type,
                        season=selected_season,
                        timestamp=datetime.now().isoformat(),
                    ))
                if o_resp:
                    st.success(f"✅ Response received | Confidence: {o_resp.overall_confidence:.0%}")
                    st.markdown(f"**Recommendation:** {o_resp.recommendation}")
                    for r in o_resp.agent_results:
                        with st.expander(f"Agent: {r.agent_name} | {r.status}"):
                            try:
                                st.json(json.loads(r.result_json))
                            except Exception:
                                st.code(r.result_json)
                else:
                    st.error(f"Orchestrator error: {o_err}")

    with col_o2:
        st.markdown("### 🏗 Architecture Overview")
        st.markdown("""
        ```
        FARMER QUERY
              │
        ┌─────▼───────────────────────────┐
        │   GO ORCHESTRATOR (port 50051)  │
        │   • goroutine fan-out           │
        │   • gRPC to all agents          │
        │   • conflict resolution         │
        └──┬──────────┬────────────┬──────┘
           │          │            │
        ┌──▼──┐   ┌───▼──┐  ┌────▼──────┐
        │ GO  │   │Python│  │  Python   │
        │Wthr │   │ Soil │  │CropHealth │
        │:50052│  │:50053│  │  :50054   │
        └─────┘   └──────┘  └───────────┘
           │          │            │
        36K CSV   928 CSV    1400 XLSX
                              + NASA API
                              + OpenMeteo
        ```
        """)

        st.markdown("### 📋 Git Workflow (Phase 1)")
        st.code("""
# ADITYA — Weather Agent
git checkout -b feature/aditya-weather-agent
# Work on: go/agents/weather/main.go
# Proto: WeatherAgentService
git add go/agents/weather/ proto/
git commit -m "feat(weather): gRPC server + 36K record data loader"
git push origin feature/aditya-weather-agent
# → Open PR to main

# AMAN — Soil Agent
git checkout -b feature/aman-soil-agent
# Work on: python/agents/soil_agent/server.py
# Proto: SoilAgentService
git add python/agents/soil_agent/ proto/
git commit -m "feat(soil): RF model + gRPC soil health + fertilizer advice"
git push origin feature/aman-soil-agent

# RISHI MOHAN — Crop Health Agent
git checkout -b feature/rishi-crop-health-agent
# Work on: python/agents/crop_health_agent/server.py
# Proto: CropHealthAgentService (NDVI + NASA POWER API)
git add python/agents/crop_health_agent/ proto/
git commit -m "feat(crop-health): NDVI via NASA POWER + growth stage tracking"
git push origin feature/rishi-crop-health-agent

# INTEGRATION (after PRs merged)
git checkout main && git pull
docker-compose up --build
        """, language="bash")

# Footer
st.markdown("---")
st.markdown(
    "**Karnataka Paddy Multi-Agent System** | Phase 1 | "
    "Built with Go + Python gRPC | Streamlit Dashboard | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
