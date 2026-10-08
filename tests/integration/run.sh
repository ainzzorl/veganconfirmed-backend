#!/usr/bin/env bash
#
# Run the desktop-server analysis integration tests end-to-end.
#
# Starts a throwaway Firestore emulator (via desktop-server's firebase tooling),
# launches the desktop-server worker against it, then runs pytest. The worker
# and the Python client share the same emulator + project, so jobs written by
# the tests are picked up and processed.
#
# Prerequisites:
#   - firebase CLI + Go toolchain (for the emulator and the worker)
#   - DESKTOP_SERVER_DIR naming a desktop-server checkout (env or .env)
#   - LM Studio running (`lms server start`) with openai/gpt-oss-20b available;
#     the tests skip cleanly if it is not reachable.
#
# Usage: tests/integration/run.sh [--keep-emulator] [extra pytest args]
#
#   --keep-emulator  Leave the Firestore emulator running after the tests finish
#                    (with the Emulator UI enabled) so you can inspect its
#                    contents. The script blocks until you press Enter or Ctrl-C.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$BACKEND_DIR"  # setting() reads .env from here
. scripts/setting.sh
DESKTOP_SERVER_DIR="$(required_setting DESKTOP_SERVER_DIR)"
PROJECT_ID="${PROJECT_ID:-desktop-server-test}"
LMS_MODEL="${LMS_MODEL:-openai/gpt-oss-20b}"

# Bind the test worker to a dedicated port so it never collides with a real
# desktop-server instance (which defaults to 9247). Only this port is ever
# touched by the cleanup below.
TEST_PORT="${DESKTOP_SERVER_TEST_PORT:-9347}"
TEST_ADDR="0.0.0.0:$TEST_PORT"

# Parse the --keep-emulator flag out of the args; everything else is forwarded
# to pytest. When set, the Firestore emulator is left running after the tests
# finish (with the Emulator UI on http://localhost:4000) so its contents can be
# inspected, and the script blocks until you press Enter (or Ctrl-C).
KEEP_EMULATOR=false
PYTEST_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --keep-emulator) KEEP_EMULATOR=true ;;
    *) PYTEST_ARGS+=("$arg") ;;
  esac
done
set -- "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"

# Kill whatever is listening on a given TCP port (SIGTERM, then SIGKILL if it
# survives). Best-effort: used to clear leftovers from a previous run that didn't
# shut down cleanly (Ctrl-C, timeouts). firebase's "firepit" binary in
# particular tends to orphan the emulator hub and its Firestore child.
free_port() {
  local port="$1" label="$2" pids
  pids="$(lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null || true)"
  [ -n "$pids" ] || return 0
  echo "Killing leftover $label on port $port: $pids"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 1
  pids="$(lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null || true)"
  # shellcheck disable=SC2086
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
}

# Clear leftovers from a previous run. The test worker lives on its own
# dedicated port, so the real desktop-server is never touched. The emulator uses
# fixed default ports; freeing the hub (4400) first lets it shut its Firestore
# child down gracefully, then we force-free anything that lingers. This also
# means a deliberately running `make emulator` session will be torn down.
free_port "$TEST_PORT" "test worker"
free_port 4400 "emulator hub"
free_port 8080 "Firestore emulator"
free_port 4000 "Emulator UI"

# Build the worker binary up front so `go run`'s compile time doesn't eat into
# the emulator session, and so we can kill it cleanly by PID.
BINARY="$(mktemp -d)/desktop-server"
echo "Building desktop-server worker..."
(cd "$DESKTOP_SERVER_DIR" && go build -o "$BINARY" .)

export FIREBASE_EMULATOR_PROJECT="$PROJECT_ID"

# `emulators:exec` is CI-oriented and leaves the Emulator UI off by default; the
# `--ui` flag turns it on (default port 4000). We only enable it when keeping the
# emulator around, since that's the only time there's a human to look at it.
EXEC_UI_ARGS=()
if [ "$KEEP_EMULATOR" = "true" ]; then
  EXEC_UI_ARGS=(--ui)
fi

cd "$DESKTOP_SERVER_DIR"
exec firebase emulators:exec --only firestore "${EXEC_UI_ARGS[@]}" --project "$PROJECT_ID" "
  set -e
  export GOOGLE_CLOUD_PROJECT='$PROJECT_ID'
  export LMS_MODEL='$LMS_MODEL'
  export DESKTOP_SERVER_ADDR='$TEST_ADDR'

  # Start the worker (also writes the heartbeat) against the emulator.
  '$BINARY' &
  WORKER_PID=\$!
  trap 'kill \$WORKER_PID 2>/dev/null || true' EXIT

  # Best-effort: preload the model so the first job doesn't pay a cold-load cost.
  sleep 2
  curl -s -X POST http://localhost:$TEST_PORT/lms/load \
    -H 'Content-Type: application/json' \
    -d '{\"model\": \"$LMS_MODEL\"}' --max-time 600 >/dev/null 2>&1 || true

  cd '$BACKEND_DIR'
  set +e
  USE_DESKTOP_SERVER=true \
  DISABLE_GEMINI_FALLBACK=true \
  LMS_MODEL='$LMS_MODEL' \
  GOOGLE_CLOUD_PROJECT='$PROJECT_ID' \
  DESKTOP_COMPLETION_TIMEOUT_SECONDS='${DESKTOP_COMPLETION_TIMEOUT_SECONDS:-600}' \
    uv run python -m pytest -m integration tests/integration $*
  PYTEST_EXIT=\$?
  set -e

  if [ '$KEEP_EMULATOR' = 'true' ]; then
    echo
    echo '=================================================================='
    echo \"Tests finished (pytest exit code: \$PYTEST_EXIT).\"
    echo 'Firestore emulator left running for inspection:'
    echo '  Emulator UI: http://localhost:4000'
    echo '  Firestore:   localhost:8080'
    echo 'Press Enter (or Ctrl-C) to shut it down...'
    echo '=================================================================='
    read -r _ || true
  fi

  exit \$PYTEST_EXIT
"
