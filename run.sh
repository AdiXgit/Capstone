#!/usr/bin/env bash
# Start the whole KrishiSense stack: crop health agent + REST gateway + React dashboard.
# Ctrl+C once stops everything.
set -uo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
LOGS="$ROOT/.logs"
mkdir -p "$LOGS"

PY="${PYTHON:-python3}"
PIDS=()
STARTED_PORTS=()

# Kill a process and everything it spawned (npm spawns vite, etc.)
kill_tree() {
  local pid=$1
  local child
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    kill_tree "$child"
  done
  kill "$pid" 2>/dev/null || true
}

CLEANED=0
cleanup() {
  [ "$CLEANED" = 1 ] && return
  CLEANED=1
  echo ""
  echo "→ shutting down…"
  for pid in "${PIDS[@]:-}"; do
    kill_tree "$pid"
  done
  # Final sweep — only ports this script started, never one it merely reused.
  sleep 1
  for port in "${STARTED_PORTS[@]:-}"; do
    for pid in $(lsof -i ":$port" -sTCP:LISTEN -t 2>/dev/null); do
      kill "$pid" 2>/dev/null || true
    done
  done
  echo "→ stopped. Logs kept in .logs/"
}
trap 'cleanup; exit 0' INT TERM
trap cleanup EXIT

port_busy() { lsof -i ":$1" -sTCP:LISTEN -t >/dev/null 2>&1; }

wait_for_port() {
  local port=$1 name=$2 tries=${3:-60}
  for ((i = 0; i < tries; i++)); do
    port_busy "$port" && return 0
    sleep 1
  done
  echo "   ✗ $name did not come up on :$port — see $LOGS/$name.log"
  return 1
}

echo "KrishiSense — starting stack"
echo "────────────────────────────"

# 1. Crop Health agent (:50054)
if port_busy 50054; then
  echo "→ crop health agent already running on :50054, reusing it"
else
  echo "→ starting crop health agent on :50054 (trains the classifier, ~10s)…"
  ( cd python/agents/crop_health_agent && exec "$PY" crop_health_agent.py ) > "$LOGS/agent.log" 2>&1 &
  PIDS+=($!); STARTED_PORTS+=(50054)
  wait_for_port 50054 agent || true
fi

# 2. REST gateway (:8000)
if port_busy 8000; then
  echo "→ gateway already running on :8000, reusing it"
else
  echo "→ starting REST gateway on :8000…"
  "$PY" gateway/main.py > "$LOGS/gateway.log" 2>&1 &
  PIDS+=($!); STARTED_PORTS+=(8000)
  wait_for_port 8000 gateway || true
fi

# 3. React dashboard (:5173)
if [ ! -d frontend/node_modules ]; then
  echo "→ installing frontend dependencies (first run only)…"
  ( cd frontend && npm install ) > "$LOGS/npm-install.log" 2>&1 || {
    echo "   ✗ npm install failed — see $LOGS/npm-install.log"; cleanup; }
fi

echo "→ starting dashboard on :5173…"
( cd frontend && exec npm run dev ) > "$LOGS/frontend.log" 2>&1 &
PIDS+=($!); STARTED_PORTS+=(5173)
wait_for_port 5173 frontend || true

echo ""
echo "  Dashboard   http://localhost:5173"
echo "  Gateway     http://localhost:8000/api/system/status"
echo "  Agent       localhost:50054 (gRPC)"
echo ""
echo "  Logs: .logs/{agent,gateway,frontend}.log"
echo "  Press Ctrl+C to stop everything."
echo ""

wait
