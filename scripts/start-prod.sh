#!/bin/bash

# Vegan Confirmed Backend Startup Script with uv and Gunicorn

set -e

# Run from the repo root: everything below (.env, app.py, the deploy source
# directory) is relative to it, not to scripts/.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "🚀 Starting Vegan Confirmed Backend..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from example..."
    if [ -f env.example ]; then
        cp env.example .env
        echo "✅ Created .env from env.example"
        echo "📝 Please edit .env file with your actual configuration"
    else
        echo "❌ env.example not found. Please create a .env file manually"
        exit 1
    fi
fi

# Install dependencies using uv. `--no-dev` skips the dev group (pytest, black,
# flake8, mypy), which production has no use for.
echo "📦 Installing dependencies with uv..."
uv sync --no-dev

# Get port from environment or use default
PORT=${PORT:-5555}
WORKERS=${WORKERS:-4}

# Run the application with Gunicorn
echo "🌟 Starting Flask application with Gunicorn..."
echo "   Port: $PORT"
echo "   Workers: $WORKERS"
uv run --no-dev gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers $WORKERS \
    --worker-class sync \
    --timeout 120 \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    app:app 