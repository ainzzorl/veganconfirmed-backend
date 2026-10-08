#!/bin/bash

# Deploy Vegan Confirmed to Google Cloud Functions
# Usage: ./scripts/deploy_cloud_function.sh [function_name] [region]

set -e

# Run from the repo root: everything below (.env, app.py, the deploy source
# directory) is relative to it, not to scripts/.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Prod releases only run from a release branch: release/YYYYMMDD/<number>, the
# date the release was cut plus a counter for that day starting at 1.
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)
if [[ ! "$BRANCH" =~ ^release/[0-9]{8}/[0-9]+$ ]]; then
    echo "Error: prod releases must run from a release branch named release/YYYYMMDD/<number>"
    echo "Current branch: ${BRANCH:-<not a git repository>}"
    exit 1
fi

# Default values
FUNCTION_NAME=${1:-"vegassist-analyze"}
REGION=${2:-"us-central1"}
if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID is not set (in .env or the environment)"
    exit 1
fi

echo "Deploying Vegan Confirmed Cloud Function..."
echo "Project: $PROJECT_ID"
echo "Function: $FUNCTION_NAME"
echo "Region: $REGION"

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY environment variable is required"
    echo "Please set it with: export GEMINI_API_KEY=your_api_key"
    exit 1
fi

if [ -z "$IP_HASH_SECRET" ]; then
    echo "Error: IP_HASH_SECRET environment variable is required"
    exit 1
fi

./scripts/fetch_geoip.sh

# Deploy the function
gcloud functions deploy $FUNCTION_NAME \
    --project=$PROJECT_ID \
    --gen2 \
    --runtime=python311 \
    --region=$REGION \
    --source=. \
    --entry-point=vegassist_analyze \
    --trigger-http \
    --allow-unauthenticated \
    --memory=512MB \
    --timeout=540s \
    --set-env-vars=GEMINI_API_KEY=$GEMINI_API_KEY,IP_HASH_SECRET=$IP_HASH_SECRET \
    --max-instances=3

echo "Deployment completed!"
echo "Function URL: https://$REGION-$PROJECT_ID.cloudfunctions.net/$FUNCTION_NAME" 