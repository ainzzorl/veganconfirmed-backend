#!/bin/bash

# Vegan Confirmed Backend Startup Script with uv

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

# Install dependencies using uv
echo "📦 Installing dependencies with uv..."
uv sync

# Run the application with uv
echo "🌟 Starting Flask application..."
uv run python app.py
