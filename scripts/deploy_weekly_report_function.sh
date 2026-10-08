#!/bin/bash

# Deploy Weekly Report Cloud Function
# This script deploys the weekly report function to Google Cloud Functions
# and sets up a Cloud Scheduler trigger

set -e

# Run from the repo root: everything below (.env, app.py, the deploy source
# directory) is relative to it, not to scripts/.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Configuration
FUNCTION_NAME="vegassist-weekly-report"
SCHEDULER_JOB_NAME="vegassist-weekly-report-scheduler"
# The scheduler's identity, and the function's only non-owner invoker.
SCHEDULER_SA_NAME="vegassist-report-scheduler"
REGION="us-central1"
RUNTIME="python311"
ENTRY_POINT="generate_weekly_report"
SOURCE_DIR="."
SCHEDULE="0 7 * * 6"  # Every Saturday at 7 AM UTC (11 AM EST)

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
SCHEDULER_SA="$SCHEDULER_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

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

if ! gcloud iam service-accounts describe $SCHEDULER_SA --project=$PROJECT_ID &>/dev/null; then
    echo "🆕 Creating scheduler service account: $SCHEDULER_SA"
    gcloud iam service-accounts create $SCHEDULER_SA_NAME --project=$PROJECT_ID \
        --display-name="Vegan Confirmed weekly report scheduler"
fi
gcloud functions add-invoker-policy-binding $FUNCTION_NAME \
    --project=$PROJECT_ID --region=$REGION --member="serviceAccount:$SCHEDULER_SA"

# Get the function URL
FUNCTION_URL=$(gcloud functions describe $FUNCTION_NAME --project=$PROJECT_ID --region=$REGION --gen2 --format="value(serviceConfig.uri)")

echo "🌐 Function URL: $FUNCTION_URL"

# Create or update the Cloud Scheduler job
echo "📅 Setting up Cloud Scheduler job: $SCHEDULER_JOB_NAME"

# Check if scheduler job already exists
if gcloud scheduler jobs describe $SCHEDULER_JOB_NAME --project=$PROJECT_ID --location=$REGION &>/dev/null; then
    echo "🔄 Updating existing scheduler job..."
    gcloud scheduler jobs update http $SCHEDULER_JOB_NAME \
        --project=$PROJECT_ID \
        --location=$REGION \
        --schedule="$SCHEDULE" \
        --uri="$FUNCTION_URL" \
        --http-method=POST \
        --update-headers="Content-Type=application/json" \
        --message-body='{"trigger": "scheduled"}' \
        --oidc-service-account-email="$SCHEDULER_SA"
else
    echo "🆕 Creating new scheduler job..."
    gcloud scheduler jobs create http $SCHEDULER_JOB_NAME \
        --project=$PROJECT_ID \
        --location=$REGION \
        --schedule="$SCHEDULE" \
        --uri="$FUNCTION_URL" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body='{"trigger": "scheduled"}' \
        --oidc-service-account-email="$SCHEDULER_SA"
fi

echo "✅ Scheduler job created/updated successfully!"

echo ""
echo "📅 Scheduling Information:"
echo "The function is now scheduled to run every Saturday at 11 AM."
echo "Scheduler job name: $SCHEDULER_JOB_NAME"
echo "Schedule: $SCHEDULE"
echo ""
echo "🔧 To send a report now, run:"
echo "gcloud scheduler jobs run $SCHEDULER_JOB_NAME --project=$PROJECT_ID --location=$REGION"
echo ""
echo "📊 To view scheduled function logs:"
echo "gcloud functions logs read --region=$REGION --filter='resource.type=cloud_function'"
echo ""
echo "🔧 To manage the scheduler job:"
echo "gcloud scheduler jobs describe $SCHEDULER_JOB_NAME --project=$PROJECT_ID --location=$REGION"
echo "gcloud scheduler jobs delete $SCHEDULER_JOB_NAME --project=$PROJECT_ID --location=$REGION"
