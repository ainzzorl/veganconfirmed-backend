#!/bin/bash

# Deploy Weekly Report Cloud Function
# This script deploys the weekly report function to Google Cloud Functions.
# Its Cloud Scheduler trigger and invoker permissions are in terraform/
# (`make tf-apply`), which reads the URL of the deployed function.

set -e

# Run from the repo root: everything below (.env, app.py, the deploy source
# directory) is relative to it, not to scripts/.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Configuration
FUNCTION_NAME="vegassist-weekly-report"
SCHEDULER_JOB_NAME="vegassist-weekly-report-scheduler"
REGION="us-central1"
RUNTIME="python311"
ENTRY_POINT="generate_weekly_report"
SOURCE_DIR="."

echo "🚀 Deploying Vegan Confirmed Weekly Report Cloud Function..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Error: Not authenticated with gcloud. Please run 'gcloud auth login' first."
    exit 1
fi

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: PROJECT_ID is not set (in .env or the environment)"
    exit 1
fi

# main.py builds the analysis core at import time, so the report function needs
# a provider configured even though it never analyzes anything: Gemini's key is
# required, and the desktop provider is switched off in --set-env-vars below.
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ Error: GEMINI_API_KEY environment variable is required"
    echo "Please set it with: export GEMINI_API_KEY=your_api_key"
    exit 1
fi

# Set the project
echo "📋 Setting project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Deploy the function
echo "📦 Deploying function: $FUNCTION_NAME"
gcloud functions deploy $FUNCTION_NAME \
    --gen2 \
    --runtime=$RUNTIME \
    --region=$REGION \
    --source=$SOURCE_DIR \
    --entry-point=$ENTRY_POINT \
    --memory=512MB \
    --timeout=540s \
    --project=$PROJECT_ID \
    --set-env-vars="SENDER_EMAIL=$SENDER_EMAIL,SENDER_LOGIN=$SENDER_LOGIN,SENDER_PASSWORD=$SENDER_PASSWORD,RECIPIENT_EMAIL=$RECIPIENT_EMAIL,GEMINI_API_KEY=$GEMINI_API_KEY,USE_DESKTOP_SERVER=false" \
    --max-instances=1 \
    --trigger-http \
    --no-allow-unauthenticated

# Drop public access left by earlier deploys. Gen2 invoker bindings live on the
# function's Cloud Run service.
if gcloud run services get-iam-policy $FUNCTION_NAME --project=$PROJECT_ID --region=$REGION \
    --format="value(bindings.members)" | grep -q allUsers; then
    echo "🔒 Removing public access from $FUNCTION_NAME..."
    gcloud functions remove-invoker-policy-binding $FUNCTION_NAME \
        --project=$PROJECT_ID --region=$REGION --member=allUsers
fi

echo "✅ Function deployed successfully!"

echo ""
echo "📅 The schedule lives in terraform/ (make tf-apply)."
echo ""
echo "🔧 To send a report now, run:"
echo "gcloud scheduler jobs run $SCHEDULER_JOB_NAME --project=$PROJECT_ID --location=$REGION"
echo ""
echo "📊 To view scheduled function logs:"
echo "gcloud functions logs read --region=$REGION --filter='resource.type=cloud_function'"
