# KrishiSense Dashboard

React + TypeScript dashboard for the Karnataka paddy multi-agent advisory system.
Replaces the old Streamlit `app.py`.

## Architecture

```
Browser (React :5173)
        │  plain JSON over HTTP
        ▼
REST gateway (FastAPI :8000)          ← gateway/main.py
        │  gRPC
        ▼
Agent mesh: orchestrator :50051 · soil :50052 · weather :50053
            crop health :50054 · market :50055 · pest :50056
```

Browsers cannot speak gRPC, so the gateway is required — it is the only thing the
frontend talks to. For each domain the gateway calls the real agent when it is
running, and otherwise computes the same answer from the datasets in `data/`.
Every response carries a `_meta.source` field (`GRPC_AGENT`, `LIVE_API`, or
`GATEWAY_DATASET`) which the UI surfaces as a tag on each page, so it is always
visible where a number came from.

## Running locally

One command from the repo root starts the agent, the gateway and the dashboard:

```bash
./run.sh
```

Then open http://localhost:5173. Ctrl+C once stops everything. Logs go to
`.logs/{agent,gateway,frontend}.log`. If a service is already running on its
port, `run.sh` reuses it instead of starting a second copy.

<details>
<summary>Or start the three pieces by hand — one terminal each</summary>

```bash
# terminal 1 — crop health agent
cd python/agents/crop_health_agent && python3 crop_health_agent.py

# terminal 2 — REST gateway
python3 gateway/main.py

# terminal 3 — dashboard
cd frontend && npm install && npm run dev
```

Each of these runs in the foreground, so they need separate terminals.
</details>

The Vite dev server proxies `/api` to `localhost:8000`, so no CORS setup is needed.

With Docker instead: `docker compose up` and open http://localhost:8501.

## Pages

| Route | Agent | Data source today |
|---|---|---|
| `/` | Orchestrator fan-in | merges all five |
| `/soil` | Soil :50052 | `Karnataka_Paddy_Main_Dataset.csv` |
| `/weather` | Weather :50053 | live Open-Meteo API |
| `/crop-health` | Crop Health :50054 | **live gRPC agent** |
| `/market` | Market :50055 | `Karnataka_Market_Prices.csv` |
| `/pest-risk` | Pest Risk :50056 | KVK rules + live weather |
| `/system` | — | live probe of all six ports |

## Configuration

`VITE_GATEWAY_URL` overrides the gateway origin (default: same-origin `/api`,
proxied to `http://localhost:8000` in dev).
