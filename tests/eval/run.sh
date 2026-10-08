#!/usr/bin/env bash
#
# Run the analysis classification eval against the local desktop-server LLM.
#
# Mirrors tests/integration/run.sh: starts a throwaway Firestore emulator,
# launches the desktop-server worker against it on a dedicated test port, then
# runs the eval (`python -m tests.eval` with a desktop combo). Reports
# classification metrics; it never fails on a wrong prediction.
#
# With DESKTOP_SERVER_HOST set (env or .env, as for `make run-local`), the worker
# is built and run on that host over SSH, from $DESKTOP_SERVER_DIR there; the
# emulator is reverse-tunnelled to it and the worker's port forwarded back. The
# emulator and the eval stay on this machine.
#
# For gemini-only combos you do NOT need this script (no emulator/worker):
#   GEMINI_API_KEY=... python -m tests.eval --run gemini:<model>
#
# Prerequisites:
#   - firebase CLI (emulator) + Go toolchain (worker; on DESKTOP_SERVER_HOST if set)
#   - LM Studio running (on the worker's machine) (`lms server start`) with the model available
#   - Node + the extension's jsdom harness (`npm install` in the extension repo)
#
# Combos are passed explicitly via --run (no defaults), at least one of which
# should be a desktop combo to make the emulator/worker worthwhile.
# Usage: tests/eval/run.sh [--keep-emulator] --run PROVIDER:MODEL[:EFFORT] [more args]
#   e.g. tests/eval/run.sh --run desktop:openai/gpt-oss-20b
#        tests/eval/run.sh --run desktop:qwen/qwen3.5-9b --case vegan_food --html
#        # compare reasoning effort levels of one model:
#        tests/eval/run.sh --html \
#          --run desktop:openai/gpt-oss-20b:low --run desktop:openai/gpt-oss-20b:high
#        # analyze 4 cases at a time:
#        tests/eval/run.sh --run desktop:openai/gpt-oss-20b --max-batch-size 4
#
# --max-batch-size N (forwarded to the eval) is also read here, because all three
# ends of the desktop path have to allow N at once: the worker is given
# DESKTOP_SERVER_MAX_CONCURRENT_JOBS=N, the model is loaded with N prediction
# slots (LMS_PARALLEL=N), and it is loaded with N times the context, because the
# slots divide the loaded window between them rather than each getting one of
# their own. Without that last part every concurrent request fails with
# "Context size has been exceeded" — the model would be serving four requests
# out of one 16384-token window. The eval still plans each request against
# LMS_CONTEXT_LENGTH, exactly as production does; only the load is scaled.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# From the repo root, where setting() looks for .env.
cd "$BACKEND_DIR"
. scripts/setting.sh
. scripts/remote-worker.sh
DESKTOP_SERVER_HOST="$(setting DESKTOP_SERVER_HOST "")"
DESKTOP_SERVER_DIR="$(required_setting DESKTOP_SERVER_DIR)"
PROJECT_ID="${PROJECT_ID:-desktop-server-test}"

# Keep these two in step with DesktopService's defaults: they decide how much of
# a page the model gets to read (context window, less the reserved completion
# and the instructions), so an eval run at other values is not measuring what
# production does.
LMS_CONTEXT_LENGTH="${LMS_CONTEXT_LENGTH:-16384}"
LMS_MAX_TOKENS="${LMS_MAX_TOKENS:-8192}"

TEST_PORT="${DESKTOP_SERVER_TEST_PORT:-9347}"
TEST_ADDR="0.0.0.0:$TEST_PORT"
# The emulator's default port (there is no firebase.json to change it).
EMULATOR_PORT=8080

# Parse --keep-emulator out; forward everything else to `python -m tests.eval`
# verbatim. No provider/model is injected — the eval requires them explicitly.
KEEP_EMULATOR=false
EVAL_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --keep-emulator) KEEP_EMULATOR=true ;;
    *) EVAL_ARGS+=("$arg") ;;
  esac
done
set -- "${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}"

# Work out which desktop model(s) the eval will request (from --run desktop:...),
# so we can preload the first one — a cold-start optimisation. The worker loads
# each job's model on demand (EnsureModel), so we only preload one and skip the
# preload entirely for runs with no desktop combo.
ARGV=( "${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}" )
DESKTOP_MODELS=()
BATCH=1
add_runs() {
  local IFS=','; local c m
  for c in $1; do
    case "$c" in
      desktop:*)
        m="${c#desktop:}"
        # Drop a trailing :EFFORT so the preload gets a bare model id.
        case "$m" in *:low|*:medium|*:high) m="${m%:*}" ;; esac
        DESKTOP_MODELS+=("$m")
        ;;
    esac
  done
}
set_batch() {
  # Anything that isn't a positive integer is left for the eval to reject.
  case "$1" in
    ''|*[!0-9]*) ;;
    *) BATCH="$1" ;;
  esac
}
for ((i = 0; i < ${#ARGV[@]}; i++)); do
  case "${ARGV[i]}" in
    --run) add_runs "${ARGV[i + 1]:-}" ;;
    --run=*) add_runs "${ARGV[i]#--run=}" ;;
    --max-batch-size) set_batch "${ARGV[i + 1]:-}" ;;
    --max-batch-size=*) set_batch "${ARGV[i]#--max-batch-size=}" ;;
  esac
done

# What one request may use stays LMS_CONTEXT_LENGTH (production's window, which
# the eval plans every prompt against); what the model is loaded with is that
# times the number of concurrent slots, which share it.
LOAD_CONTEXT_LENGTH=$((LMS_CONTEXT_LENGTH * BATCH))

# Batched submissions all land at once, so the client's default 10s pickup
# window can expire on the ones the worker claims last.
if [ "$BATCH" -gt 1 ]; then
  DESKTOP_PICKUP_TIMEOUT_SECONDS="${DESKTOP_PICKUP_TIMEOUT_SECONDS:-30}"
else
  DESKTOP_PICKUP_TIMEOUT_SECONDS="${DESKTOP_PICKUP_TIMEOUT_SECONDS:-10}"
fi
# Model to preload, if any. Empty means no desktop combo — no preload.
PRELOAD_MODEL="${DESKTOP_MODELS[0]:-}"

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

free_port "$TEST_PORT" "test worker"
free_port 4400 "emulator hub"
free_port "$EMULATOR_PORT" "Firestore emulator"
free_port 4000 "Emulator UI"

# The worker's environment, wherever it runs. No LMS_MODEL here: the worker
# loads each job's requested model on demand, and the eval sets LMS_MODEL per
# combo before constructing AnalysisCore.
#
# LMS_CONTEXT_LENGTH pins the load context so the worker's `lms load` doesn't
# default to the model's huge max context and crash on VRAM (Vulkan
# ErrorDeviceLost). Batched runs load N times the per-request window, since the
# N slots share it. To also pin GPU offload, export LMS_GPU_OFFLOAD before
# running this script.
WORKER_ENV="GOOGLE_CLOUD_PROJECT=$PROJECT_ID LMS_CONTEXT_LENGTH=$LOAD_CONTEXT_LENGTH"
if [ -n "${LMS_GPU_OFFLOAD:-}" ]; then
  WORKER_ENV+=" LMS_GPU_OFFLOAD=$LMS_GPU_OFFLOAD"
fi
# Batched runs only: let the worker claim that many jobs at once, and load the
# model with that many prediction slots (an instance carried over from an
# earlier run with fewer slots, or less context, is reloaded). Left unset
# otherwise, so an unbatched run is unchanged.
if [ "$BATCH" -gt 1 ]; then
  WORKER_ENV+=" DESKTOP_SERVER_MAX_CONCURRENT_JOBS=$BATCH LMS_PARALLEL=$BATCH"
fi

if [ -n "$DESKTOP_SERVER_HOST" ]; then
  # Read from the environment by the emulators:exec script below, which spares
  # quoting them into it.
  export DESKTOP_SERVER_HOST
  export REMOTE_WORKER_COMMAND="$(remote_worker_command "$DESKTOP_SERVER_DIR" "$TEST_PORT" \
    "FIRESTORE_EMULATOR_HOST=localhost:$EMULATOR_PORT $WORKER_ENV")"
  # The remote build happens after the emulator is up, so allow it more time.
  STARTUP_WAIT=180
else
  BINARY="$(mktemp -d)/desktop-server"
  echo "Building desktop-server worker..."
  (cd "$DESKTOP_SERVER_DIR" && go build -o "$BINARY" .)
  STARTUP_WAIT=20
fi

export FIREBASE_EMULATOR_PROJECT="$PROJECT_ID"

EXEC_UI_ARGS=()
if [ "$KEEP_EMULATOR" = "true" ]; then
  EXEC_UI_ARGS=(--ui)
fi

exec firebase emulators:exec --only firestore ${EXEC_UI_ARGS[@]+"${EXEC_UI_ARGS[@]}"} --project "$PROJECT_ID" "
  set -e

  # Start the worker (also writes the heartbeat) against the emulator. The remote
  # one reaches it through a reverse tunnel, on the same port it has here; -tt so
  # it gets SIGHUP when this connection drops.
  if [ -n \"\${DESKTOP_SERVER_HOST:-}\" ]; then
    echo \"Starting desktop-server worker on \$DESKTOP_SERVER_HOST...\"
    ssh -tt -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \\
      -R '$EMULATOR_PORT:localhost:$EMULATOR_PORT' \\
      -L '$TEST_PORT:localhost:$TEST_PORT' \\
      \"\$DESKTOP_SERVER_HOST\" \"\$REMOTE_WORKER_COMMAND\" </dev/null &
  else
    DESKTOP_SERVER_ADDR='$TEST_ADDR' $WORKER_ENV '${BINARY:-}' &
  fi
  WORKER_PID=\$!
  trap 'kill \$WORKER_PID 2>/dev/null || true' EXIT

  for _ in \$(seq $STARTUP_WAIT); do
    curl -sf --max-time 2 http://localhost:$TEST_PORT/ >/dev/null 2>&1 && break
    kill -0 \$WORKER_PID 2>/dev/null || { echo 'desktop-server worker exited during startup'; exit 1; }
    sleep 1
  done
  curl -sf --max-time 2 http://localhost:$TEST_PORT/ >/dev/null 2>&1 \\
    || { echo 'desktop-server worker never became reachable on port $TEST_PORT'; exit 1; }

  # Best-effort: preload the first desktop model so the first job doesn't pay a
  # cold-load cost. The worker's /lms/load handler no-ops if it's already loaded
  # with the slots LMS_PARALLEL wants, and reloads it otherwise (it reads
  # lms ps --json). Skipped when no desktop model is being benchmarked (e.g.
  # a pure-gemini run); other desktop models load on demand per job.
  PRELOAD_MODEL='$PRELOAD_MODEL'
  if [ -n \"\$PRELOAD_MODEL\" ]; then
    curl -s -X POST http://localhost:$TEST_PORT/lms/load \
      -H 'Content-Type: application/json' \
      -d \"{\\\"model\\\": \\\"\$PRELOAD_MODEL\\\"}\" --max-time 600 >/dev/null 2>&1 || true
  fi

  cd '$BACKEND_DIR'
  set +e
  GOOGLE_CLOUD_PROJECT='$PROJECT_ID' \
  LMS_MAX_TOKENS='$LMS_MAX_TOKENS' \
  LMS_CONTEXT_LENGTH='$LMS_CONTEXT_LENGTH' \
  DESKTOP_COMPLETION_TIMEOUT_SECONDS='${DESKTOP_COMPLETION_TIMEOUT_SECONDS:-600}' \
  DESKTOP_PICKUP_TIMEOUT_SECONDS='$DESKTOP_PICKUP_TIMEOUT_SECONDS' \
    uv run python -m tests.eval $*
  EVAL_EXIT=\$?
  set -e

  if [ '$KEEP_EMULATOR' = 'true' ]; then
    echo
    echo '=================================================================='
    echo \"Eval finished (exit code: \$EVAL_EXIT).\"
    echo 'Firestore emulator left running for inspection:'
    echo '  Emulator UI: http://localhost:4000'
    echo '  Firestore:   localhost:8080'
    echo 'Press Enter (or Ctrl-C) to shut it down...'
    echo '=================================================================='
    read -r _ || true
  fi

  exit \$EVAL_EXIT
"
