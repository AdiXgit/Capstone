# KrishiSense — Image Detection + Cross-Agent Orchestration

Two capabilities added on top of the existing paddy-advisory mesh.

## 1. YOLOv8 leaf disease detection (image path of the Crop Health agent)

The Crop Health agent previously scored stress only from tabular NDVI/LAI/DAS.
It now also classifies a **photo of a leaf** into a paddy disease and returns the
KVK treatment.

Files:
- `python/agents/crop_health_agent/disease_detection.py` — lazy-loading YOLOv8-cls detector
- `python/agents/crop_health_agent/disease_treatment.py` — class → disease → KVK treatment map
- `scripts/download_paddy_dataset.py` — download a public Kaggle rice-disease dataset
- `scripts/train_yolo_classifier.py` — fine-tune `yolov8n-cls.pt` → `models/paddy_disease_cls.pt`
- Gateway routes: `POST /api/crop-health/detect`, `GET /api/crop-health/detect/status`
- UI: `frontend/src/components/DiseaseDetect.tsx` (on the Crop Health page)

### Enable it (one-time)
```bash
pip install -r requirements.txt -r requirements-vision.txt

# 1) dataset (needs a Kaggle API token in ~/.kaggle/kaggle.json — see the script header)
python scripts/download_paddy_dataset.py

# 2) fine-tune (CPU ok with low epochs; GPU faster)
python scripts/train_yolo_classifier.py --epochs 20 --imgsz 224
```
`disease_detection.py` auto-loads `models/paddy_disease_cls.pt` once it exists.
Until then the endpoint works but uses base pretrained weights and flags results
as not paddy-specific (`is_paddy_condition: false`). `GET /detect/status` reports
which state you're in.

## 2. Cross-agent orchestration

`/api/overview` fans *in* from the gateway. The orchestrator instead routes ONE
query through gRPC and fans *out* to the agents concurrently, then synthesizes a
single advisory — including cross-agent rules (e.g. rain + disease-risk → hold
sprays; flowering → spray-lock).

Files:
- `python/orchestrator/orchestrator_server.py` — `OrchestratorService` (RegisterAgent / RouteQuery / GetSystemStatus)
- `python/agents/weather_agent/weather_agent.py` — new live weather agent (Open-Meteo + dataset fallback)
- `python/agents/soil_agent/soil_agent.py` — fixed (was crashing on a wrong servicer class name) + `GetSoilHealth`
- Gateway route: `GET /api/orchestrator`
- UI: `frontend/src/pages/Orchestrator.tsx`

> Note: `go/orchestrator/main.go` implements the same contract but has no `go.mod`
> / generated Go stubs yet, so the **Python orchestrator is the runnable one**.
> Its swapped SOIL/WEATHER port map was also fixed for when the Go build lands.

## Run the whole mesh locally
```bash
pip install -r requirements.txt
bash scripts/run_local.sh          # orchestrator + soil + weather + crop-health + gateway

# in another terminal
cd frontend && npm install && VITE_GATEWAY_URL=http://localhost:8000 npm run dev
```
Quick check:
```bash
curl "http://localhost:8000/api/orchestrator?district=Mandya&season=Kharif&stage=Flowering"
curl "http://localhost:8000/api/system/status"   # expect SOIL/WEATHER/CROP_HEALTH ONLINE
```

Stop the mesh:
```bash
pkill -f "orchestrator_server.py"; pkill -f "_agent.py"; pkill -f "gateway/main.py"
```

## Runtime note
The local Python is 3.9, which does not support `X | None` annotations at runtime
(PEP 604). New modules use `from __future__ import annotations` where needed.
