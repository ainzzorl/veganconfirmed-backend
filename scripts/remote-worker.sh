# Sourced by scripts that can run their desktop-server worker on another host
# over SSH (DESKTOP_SERVER_HOST).
#
# Prints the remote command, for `ssh HOST "$(remote_worker_command ...)"`, that
# builds the worker in checkout $1 there and runs it on 127.0.0.1:$2 with the
# environment assignments in $3 (space-separated NAME=VALUE, no quoting needed).
# Go's and LM Studio's bin directories are added by hand: bash -l misses them
# when the remote user's shell rc (e.g. .zshrc) is what sets them. Frees the
# worker's port first: SIGHUP from the closing SSH session should stop the
# worker, but a leftover would otherwise make the new one fail to bind.
remote_worker_command() {
  local dir="$1" port="$2" env="$3" script
  script="$(cat <<EOF
set -e
export PATH="/usr/local/go/bin:\$HOME/go/bin:\$HOME/.lmstudio/bin:\$PATH"
pids="\$( (lsof -ti tcp:$port -sTCP:LISTEN || fuser -n tcp $port) 2>/dev/null || true)"
[ -z "\$pids" ] || { echo "Killing leftover worker on port $port: \$pids"; kill -9 \$pids || true; }
cd $(printf %q "$dir")
bin="\$(mktemp -d)/desktop-server"
echo "Building desktop-server worker in \$PWD..."
go build -o "\$bin" .
DESKTOP_SERVER_ADDR=127.0.0.1:$port $env exec "\$bin"
EOF
)"
  echo "bash -lc $(printf %q "$script")"
}
