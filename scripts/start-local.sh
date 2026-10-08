#!/bin/bash
#
# Start the backend together with a local desktop-server worker.
#
# Meant to be run inside `firebase emulators:exec` (see `make run-local`): the
# backend and the worker talk to each other only through Firestore (the job
# queue + the heartbeat doc), so a worker pointed at real Firestore is invisible
# to a backend running against the emulator. This starts a throwaway worker in
# the same emulator, then runs the backend; the worker is killed when the
# backend exits.
#
# Environment (each is also read from .env, which app.py picks up
# via load_dotenv but this shell doesn't, with a real env var taking precedence):
#   DESKTOP_SERVER_HOST        - run the worker on this host over SSH (e.g.
#                                gpu-box.local) instead of locally.
#                                The emulator is reverse-tunnelled to it and the
#                                worker's port forwarded back, so neither has to
#                                listen on the network.
#   DESKTOP_SERVER_DIR         - desktop-server checkout (required); on
#                                DESKTOP_SERVER_HOST when that is set
#   DESKTOP_SERVER_LOCAL_PORT  - port for the throwaway worker's HTTP server
#                                (default 9447; kept off 9247 so a real
#                                desktop-server instance isn't disturbed, and
#                                off 9347 which tests/*/run.sh use)
#   PORT                       - port the backend listens on (default 5555, as
#                                in app.py)

set -euo pipefail

# Run from the repo root: .env and the backend's start script are relative to
# it, not to scripts/.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

. scripts/setting.sh
. scripts/remote-worker.sh

DESKTOP_SERVER_HOST="$(setting DESKTOP_SERVER_HOST "")"
DESKTOP_SERVER_DIR="$(required_setting DESKTOP_SERVER_DIR)"
LOCAL_PORT="$(setting DESKTOP_SERVER_LOCAL_PORT 9447)"
APP_PORT="$(setting PORT 5555)"

# The worker shells out to the `lms` CLI to load models, and LM Studio's bin
# directory isn't on PATH for non-login shells.
if [ -d "$HOME/.lmstudio/bin" ]; then
  export PATH="$HOME/.lmstudio/bin:$PATH"
fi

# Kill whatever is listening on a given TCP port: SIGTERM, then SIGKILL for
# anything that survives. desktop-server traps SIGTERM to cancel its worker
# context but stays blocked in ListenAndServe, so it does survive a plain kill —
# and it treats a failed bind as fatal, so a leftover from an earlier run would
# otherwise silently take the new worker down. Flask/werkzeug likewise refuses
# to start when its port is taken.
free_port() {
  local port="$1" label="$2" pids
  pids="$(lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null || true)"
  [ -n "$pids" ] || return 0
  echo "🧹 Killing leftover $label on port $port: $pids"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 1
  pids="$(lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null || true)"
  # shellcheck disable=SC2086
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
  return 0
}

free_port "$LOCAL_PORT" "worker"
free_port "$APP_PORT" "backend"

if [ -n "$DESKTOP_SERVER_HOST" ]; then
  # The worker reaches the emulator through a reverse tunnel, on the same port
  # it has here.
  EMULATOR_PORT="${FIRESTORE_EMULATOR_HOST##*:}"
  echo "🖥️  Starting desktop-server worker on $DESKTOP_SERVER_HOST (project ${GOOGLE_CLOUD_PROJECT:-unset}, emulator tunnelled on port $EMULATOR_PORT)..."
  # -tt so the remote worker gets SIGHUP when this connection drops.
  ssh -tt -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
    -R "$EMULATOR_PORT:localhost:$EMULATOR_PORT" \
    -L "$LOCAL_PORT:localhost:$LOCAL_PORT" \
    "$DESKTOP_SERVER_HOST" "$(remote_worker_command "$DESKTOP_SERVER_DIR" "$LOCAL_PORT" \
      "FIRESTORE_EMULATOR_HOST=localhost:$EMULATOR_PORT GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:-}")" </dev/null &
  WORKER_PID=$!
  # The remote build happens after this point, so allow it more time.
  STARTUP_WAIT=180
else
  BINARY="$(mktemp -d)/desktop-server"
  echo "🔨 Building desktop-server worker from $DESKTOP_SERVER_DIR..."
  (cd "$DESKTOP_SERVER_DIR" && go build -o "$BINARY" .)

  # GOOGLE_CLOUD_PROJECT comes from the Makefile and FIRESTORE_EMULATOR_HOST from
  # emulators:exec; both are inherited here, so the worker joins the same emulator
  # as the backend.
  echo "🖥️  Starting desktop-server worker on $LOCAL_PORT (project ${GOOGLE_CLOUD_PROJECT:-unset}, emulator ${FIRESTORE_EMULATOR_HOST:-none})..."
  DESKTOP_SERVER_ADDR="0.0.0.0:$LOCAL_PORT" "$BINARY" &
  WORKER_PID=$!
  STARTUP_WAIT=20
fi

# Signals are trapped as well as EXIT, because bash skips the EXIT trap when it
# dies on an untrapped SIGINT/SIGTERM.
cleanup() {
  local pids=("$WORKER_PID")
  [ -n "${APP_PID:-}" ] && pids+=("$APP_PID")
  kill "${pids[@]}" 2>/dev/null || true
  sleep 1
  kill -9 "${pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Fail fast (rather than letting every analysis fall back to Gemini) if the
# worker didn't come up.
for _ in $(seq "$STARTUP_WAIT"); do
  curl -sf --max-time 2 "http://localhost:$LOCAL_PORT/" >/dev/null 2>&1 && break
  kill -0 "$WORKER_PID" 2>/dev/null || { echo "❌ desktop-server worker exited during startup"; exit 1; }
  sleep 1
done
curl -sf --max-time 2 "http://localhost:$LOCAL_PORT/" >/dev/null 2>&1 \
  || { echo "❌ desktop-server worker never became reachable on port $LOCAL_PORT"; exit 1; }
echo "✅ desktop-server worker ready on port $LOCAL_PORT"

# The backend runs in the background and is waited on, rather than in the
# foreground: bash defers trap handling until the current foreground command
# returns, which would delay the cleanup above until start.sh happened to exit.
PORT="$APP_PORT" scripts/start.sh &
APP_PID=$!
set +e
wait "$APP_PID"
APP_EXIT=$?
set -e
exit "$APP_EXIT"
