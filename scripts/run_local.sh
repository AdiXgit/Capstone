#!/usr/bin/env bash
# Bring up the full KrishiSense mesh locally (no Docker):
#   orchestrator (:50051) → soil (:50052) · weather (:50053) · crop-health (:50054)
#   → REST gateway (:8000)
#
# Agents register with the orchestrator on startup, so once this is up the
# Orchestrator page in the dashboard shows real cross-agent fan-out.
#
# Usage:  bash scripts/run_local.sh
# Stop:   Ctrl-C (all child processes are killed together)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

PY="${PYTHON:-python3}"
PIDS=()
cleanup() { echo; echo "Stopping mesh…"; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

start() {  # name  dir  script
  echo "→ starting $1"
  ( cd "$2" && exec "$PY" "$3" ) &
  PIDS+=($!)
  sleep 1
}

# 1) Orchestrator first so agents can register against it.
start "orchestrator" "python/orchestrator" "orchestrator_server.py"
sleep 1
# 2) Domain agents.
start "soil-agent"        "python/agents/soil_agent"        "soil_agent.py"
start "weather-agent"     "python/agents/weather_agent"     "weather_agent.py"
start "crop-health-agent" "python/agents/crop_health_agent" "crop_health_agent.py"
start "market-agent"      "python/agents/market_agent"      "market_agent.py"
start "pest-agent"        "python/agents/pest_agent"        "pest_agent.py"
sleep 2
# 3) REST gateway (foreground-ish; keeps the script alive).
echo "→ starting gateway on :8000"
( cd "$REPO" && exec "$PY" gateway/main.py ) &
PIDS+=($!)

echo
echo "Mesh is up:"
echo "  orchestrator  localhost:50051"
echo "  soil          localhost:50052"
echo "  weather       localhost:50053"
echo "  crop-health   localhost:50054"
echo "  market        localhost:50055"
echo "  pest          localhost:50056"
echo "  gateway       http://localhost:8000  (try /api/orchestrator?district=Mandya&season=Kharif)"
echo
echo "Frontend:  cd frontend && npm install && npm run dev   (proxy VITE_GATEWAY_URL=http://localhost:8000)"
echo "Press Ctrl-C to stop everything."
wait
