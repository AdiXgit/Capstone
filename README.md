# 🌾 Karnataka Paddy Multi-Agent System — Phase 1

**40% Complete** | Go + Python | gRPC A2A | Streamlit Dashboard

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   FARMER INTERFACE LAYER                            │
│              Streamlit Dashboard (port 8501)                        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ gRPC
┌───────────────────────────▼─────────────────────────────────────────┐
│         GO DECISION SUPPORT ORCHESTRATOR (port 50051)               │
│   • goroutine-based concurrent fan-out                              │
│   • agent self-registration via RegisterAgent()                     │
│   • conflict resolution + synthesis                                 │
└────────┬────────────────────┬──────────────────────┬────────────────┘
         │ gRPC               │ gRPC                 │ gRPC
┌────────▼──────┐   ┌─────────▼────────┐   ┌────────▼──────────────┐
│ GO WEATHER    │   │ PYTHON SOIL      │   │ PYTHON CROP HEALTH    │
│ AGENT :50052  │   │ AGENT :50053     │   │ AGENT :50054          │
│ (Aditya)      │   │ (Aman)           │   │ (Rishi Mohan)         │
│               │   │                  │   │                       │
│ 36,530 CSV    │   │ RF Fertilizer    │   │ NDVI + NASA POWER     │
│ records       │   │ Model            │   │ Growth Stage Tracker  │
│ 10 districts  │   │ Soil Health      │   │ Pest/Disease Risk     │
│ 2015–2024     │   │ NPK profiles     │   │ 1,400 growth records  │
└───────────────┘   └──────────────────┘   └───────────────────────┘
```

---

## 🗂 Dataset Mapping

| Dataset | Records | Owner | Used By |
|---|---|---|---|
| `Karnataka_Weather_Daily.csv` | 36,530 | Aditya | Weather Agent |
| `Karnataka_Paddy_Main_Dataset.csv` | 928 | Aman | Soil Agent (soil cols) |
| `Karnataka_Paddy_Main_Dataset.csv` | 928 | Rishi Mohan | Crop Health (pest cols) |
| `Karnataka_Paddy_AI_ML_Dataset.xlsx` → `Satellite_Indices` | ~1,200 | Rishi Mohan | NDVI fallback |
| `Karnataka_Paddy_AI_ML_Dataset.xlsx` → `Crop_Growth_Stages` | 1,400 | Rishi Mohan | Growth stage |
| `Karnataka_Market_Prices.csv` | 2,610 | Phase 2 | Market Agent |
| `Karnataka_Paddy_Comprehensive_Dataset.xlsx` | 928 | Phase 2 | Yield Agent |

---

## 👥 Team Git Workflow

### Initial Setup (All team members)

```bash
# Clone the repo
git clone https://github.com/YOUR_ORG/paddy-multiagent.git
cd paddy-multiagent

# Copy datasets to data/ directory
mkdir -p data/
cp /path/to/datasets/*.csv data/
cp /path/to/datasets/*.xlsx data/

# Generate proto stubs (first time only)
chmod +x scripts/gen_proto.sh
./scripts/gen_proto.sh

# Install Python dependencies
pip install -r python/requirements.txt
```

---

### 🔵 ADITYA — Weather Agent (Go)

```bash
# 1. Create branch
git checkout main
git pull origin main
git checkout -b feature/aditya-weather-agent

# 2. Files to work on:
#    - go/agents/weather/main.go        ← Main implementation
#    - proto/paddy_agents.proto         ← WeatherAgentService section

# 3. Run locally for dev
WEATHER_CSV_PATH=./data/Karnataka_Weather_Daily.csv \
WEATHER_AGENT_PORT=50052 \
ORCHESTRATOR_ADDR=localhost:50051 \
  go run ./go/agents/weather/

# 4. Test with grpcurl
grpcurl -plaintext -d '{
  "district": "Koppal",
  "date": "2024-06-15",
  "forecast_days": 7
}' localhost:50052 paddyagents.WeatherAgentService/GetWeatherForecast

# 5. Commit and push
git add go/agents/weather/
git commit -m "feat(weather): Add 36K record data loader + 7-day forecast"
git commit -m "feat(weather): Add alert detection for flood/heat/humidity"
git commit -m "feat(weather): Add historical query + monthly average fallback"
git push origin feature/aditya-weather-agent

# 6. Open PR → main
# PR title: "[Weather Agent] Aditya - Phase 1 complete"
# Reviewers: Aman, Rishi Mohan
```

**Aditya's Key Responsibilities:**
- `GetWeatherForecast` — returns current + 7-day forecast per district
- `GetWeatherAlert` — flood/heat/disease risk alerts
- `GetHistoricalWeather` — date range queries for analytics
- Auto-register with orchestrator on startup

---

### 🟢 AMAN — Soil Agent (Python)

```bash
# 1. Create branch
git checkout main
git pull origin main
git checkout -b feature/aman-soil-agent

# 2. Files to work on:
#    - python/agents/soil_agent/server.py   ← Main implementation
#    - proto/paddy_agents.proto              ← SoilAgentService section

# 3. Copy proto stubs to your agent dir (after gen_proto.sh)
cp python/shared/paddy_agents_pb2*.py python/agents/soil_agent/

# 4. Run locally
MAIN_CSV_PATH=./data/Karnataka_Paddy_Main_Dataset.csv \
SOIL_AGENT_PORT=50053 \
ORCHESTRATOR_ADDR=localhost:50051 \
  python python/agents/soil_agent/server.py

# 5. Test with grpcurl
grpcurl -plaintext -d '{
  "district": "Mandya",
  "season": "Kharif",
  "soil_quality": "medium"
}' localhost:50053 paddyagents.SoilAgentService/GetSoilHealth

grpcurl -plaintext -d '{
  "district": "Ballari",
  "season": "Rabi"
}' localhost:50053 paddyagents.SoilAgentService/GetFertilizerAdvice

# 6. Commit
git add python/agents/soil_agent/
git commit -m "feat(soil): Load 928 district records, build NPK profiles per district"
git commit -m "feat(soil): Train Random Forest fertilizer recommendation model"
git commit -m "feat(soil): Add soil trend analysis (5-year pH + organic carbon)"
git push origin feature/aman-soil-agent

# PR title: "[Soil Agent] Aman - Phase 1 complete"
```

**Aman's Key Responsibilities:**
- `GetSoilHealth` — pH, NPK levels, health scoring per district
- `GetFertilizerAdvice` — RF model predicts optimal Urea/DAP/Potash
- `GetSoilTrend` — 5-year trend analysis, improvement/degradation signal
- Auto-register with orchestrator on startup

---

### 🟠 RISHI MOHAN — Crop Health Agent (Python)

```bash
# 1. Create branch
git checkout main
git pull origin main
git checkout -b feature/rishi-crop-health-agent

# 2. Files to work on:
#    - python/agents/crop_health_agent/server.py  ← Main implementation
#    - proto/paddy_agents.proto                    ← CropHealthAgentService section

# 3. Run locally
AI_ML_XLSX_PATH=./data/Karnataka_Paddy_AI_ML_Dataset.xlsx \
MAIN_CSV_PATH=./data/Karnataka_Paddy_Main_Dataset.csv \
CROP_HEALTH_AGENT_PORT=50054 \
ORCHESTRATOR_ADDR=localhost:50051 \
  python python/agents/crop_health_agent/server.py

# 4. Test with grpcurl
grpcurl -plaintext -d '{
  "district": "Koppal",
  "season": "Kharif",
  "year": 2024
}' localhost:50054 paddyagents.CropHealthAgentService/GetCropHealthStatus

# Test NDVI (will call NASA POWER API if dataset record not found)
grpcurl -plaintext -d '{
  "district": "Raichur",
  "year": 2024,
  "month": 8
}' localhost:50054 paddyagents.CropHealthAgentService/GetNDVIAnalysis

# 5. Commit
git add python/agents/crop_health_agent/
git commit -m "feat(crop-health): Load 1400 growth stage records + satellite NDVI"
git commit -m "feat(crop-health): Integrate NASA POWER API for free NDVI proxy"
git commit -m "feat(crop-health): Add Open-Meteo fallback + pest/disease risk matrix"
git push origin feature/rishi-crop-health-agent

# PR title: "[Crop Health Agent] Rishi Mohan - Phase 1 complete"
```

**Rishi Mohan's Key Responsibilities:**
- `GetCropHealthStatus` — growth stage, NDVI, LAI, health scoring
- `GetNDVIAnalysis` — Dataset → NASA POWER API → Open-Meteo (cascading)
- `GetGrowthStageInfo` — DAS-based growth tracking with care advisory
- `GetPestDiseaseRisk` — humidity × growth stage × historical risk matrix

---

## 🚀 Running the Full Stack

```bash
# 1. Prepare data directory
mkdir -p data/
cp /your/datasets/*.csv data/
cp /your/datasets/*.xlsx data/

# 2. Build and start all services
docker-compose up --build

# 3. Access dashboard
open http://localhost:8501

# 4. Check agent health
docker-compose ps
docker-compose logs -f weather-agent
docker-compose logs -f soil-agent
docker-compose logs -f crop-health-agent

# 5. Stop
docker-compose down
```

---

## 🔐 Free Satellite APIs (No Key Required)

### NASA POWER API
```
Endpoint: https://power.larc.nasa.gov/api/temporal/daily/point
Auth: None (completely free)
Data: PAR (photosynthetically active radiation) → NDVI proxy
Coverage: Global, 2015–present, 0.5° resolution
```

### Open-Meteo API
```
Endpoint: https://api.open-meteo.com/v1/forecast
Auth: None (free, 10K req/day)
Data: Shortwave radiation → vegetation health proxy
Coverage: Global, 11km resolution
```

### Sentinel Hub (Optional — 30-day free trial)
```bash
# Set in .env or docker-compose environment:
SENTINEL_HUB_CLIENT_ID=your_client_id
SENTINEL_HUB_CLIENT_SECRET=your_secret
# Sign up at: https://www.sentinel-hub.com/trial/
# Provides true NDVI from Sentinel-2 (10m resolution)
```

---

## 📊 Phase 1 Completion (40%)

| Component | Status | Owner |
|---|---|---|
| ✅ Proto definitions (all agents) | Complete | All |
| ✅ Go Orchestrator (concurrent fan-out) | Complete | — |
| ✅ Weather Agent (Go, 36K records) | Complete | Aditya |
| ✅ Soil Agent (Python, RF model) | Complete | Aman |
| ✅ Crop Health Agent (Python, NDVI API) | Complete | Rishi Mohan |
| ✅ Streamlit Dashboard (5 tabs) | Complete | — |
| ✅ Docker Compose | Complete | — |
| ⏳ Yield Prediction Agent (ML) | Phase 2 | — |
| ⏳ Market Intelligence Agent | Phase 2 | — |
| ⏳ Risk Assessment Agent | Phase 2 | — |
| ⏳ Kubernetes deployment | Phase 3 | — |
| ⏳ WhatsApp / SMS interface | Phase 3 | — |

---

## 🧪 Quick Test (without Docker)

```bash
# Terminal 1: Start orchestrator
go run ./go/orchestrator/

# Terminal 2: Start weather agent (Aditya)
WEATHER_CSV_PATH=./data/Karnataka_Weather_Daily.csv \
go run ./go/agents/weather/

# Terminal 3: Start soil agent (Aman)
MAIN_CSV_PATH=./data/Karnataka_Paddy_Main_Dataset.csv \
python python/agents/soil_agent/server.py

# Terminal 4: Start crop health agent (Rishi Mohan)
AI_ML_XLSX_PATH=./data/Karnataka_Paddy_AI_ML_Dataset.xlsx \
MAIN_CSV_PATH=./data/Karnataka_Paddy_Main_Dataset.csv \
python python/agents/crop_health_agent/server.py

# Terminal 5: Start dashboard
cd streamlit_dashboard && streamlit run app.py
```

---

## 📁 Project Structure

```
paddy-multiagent-phase1/
├── proto/
│   └── paddy_agents.proto          # All service definitions
├── go/
│   ├── orchestrator/
│   │   └── main.go                 # Go orchestrator (goroutines + gRPC)
│   └── agents/
│       └── weather/
│           └── main.go             # ADITYA: Weather agent
├── python/
│   ├── requirements.txt
│   ├── shared/                     # Generated proto stubs
│   └── agents/
│       ├── soil_agent/
│       │   └── server.py           # AMAN: Soil + fertilizer agent
│       └── crop_health_agent/
│           └── server.py           # RISHI MOHAN: Crop health + NDVI
├── streamlit_dashboard/
│   └── app.py                      # 5-tab Streamlit dashboard
├── docker/
│   ├── Dockerfile.orchestrator
│   ├── Dockerfile.weather
│   ├── Dockerfile.python_agent
│   └── Dockerfile.dashboard
├── docker-compose.yml
├── go.mod
├── scripts/
│   └── gen_proto.sh                # Proto code generation
└── README.md
```
