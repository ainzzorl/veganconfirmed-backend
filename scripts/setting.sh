# Sourced by scripts that, unlike app.py (load_dotenv), don't load .env.
#
# The value of a setting: the env var if set, else its last assignment in .env
# in the current directory (surrounding quotes stripped), else the given default.
setting() {
  local name="$1" default="$2" value="${!1:-}"
  if [ -z "$value" ] && [ -f .env ]; then
    value="$(sed -n "s/^[[:space:]]*\(export[[:space:]]\{1,\}\)\{0,1\}$name[[:space:]]*=[[:space:]]*//p" .env \
      | tail -1 | sed 's/[[:space:]]*$//; s/^"\(.*\)"$/\1/; s/^'"'"'\(.*\)'"'"'$/\1/')"
  fi
  echo "${value:-$default}"
}

# Same, for a setting with no default: exits with an error when unset.
required_setting() {
  local value
  value="$(setting "$1" "")"
  if [ -z "$value" ]; then
    echo "Error: $1 is not set (in the environment or .env); see env.example" >&2
    exit 1
  fi
  echo "$value"
}
