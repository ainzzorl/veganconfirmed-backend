#!/bin/bash

# Deploy the database viewer (viewer_main.py) to Cloud Run, behind Google
# sign-in. One-time console setup, before the first run:
#   1. APIs & Services > OAuth consent screen: External, left in Testing, with
#      only the DB_VIEWER_ALLOWED_EMAILS accounts as test users.
#   2. Credentials > OAuth client ID (Web application). Put its ID in .env as
#      GOOGLE_OAUTH_CLIENT_ID and its secret as GOOGLE_OAUTH_CLIENT_SECRET; the
#      secret goes into Secret Manager on the first run and can then be
#      removed from .env.
#   3. After the first deploy, add the redirect URI this script prints to the
#      client's "Authorized redirect URIs".

set -e

# Run from the repo root: the deploy source directory is relative to it.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SERVICE_NAME="vegassist-db-viewer"
REGION="us-central1"
SA_NAME="vegassist-db-viewer"
CLIENT_SECRET_NAME="db-viewer-oauth-client-secret"
SESSION_KEY_NAME="db-viewer-secret-key"

for name in PROJECT_ID GOOGLE_OAUTH_CLIENT_ID DB_VIEWER_ALLOWED_EMAILS; do
    if [ -z "${!name}" ]; then
        echo "❌ Error: $name is not set (in .env or the environment)"
        exit 1
    fi
done
SA="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com secretmanager.googleapis.com \
    --project="$PROJECT_ID"

# The viewer's own identity, which can only read Firestore.
if ! gcloud iam service-accounts describe "$SA" --project="$PROJECT_ID" &> /dev/null; then
    gcloud iam service-accounts create "$SA_NAME" --project="$PROJECT_ID" \
        --display-name="Database viewer (read-only)"
fi
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SA" --role=roles/datastore.viewer \
    --condition=None --quiet > /dev/null

# create_secret NAME: creates it from stdin unless it already exists.
create_secret() {
    if gcloud secrets describe "$1" --project="$PROJECT_ID" &> /dev/null; then
        cat > /dev/null
    else
        gcloud secrets create "$1" --project="$PROJECT_ID" --data-file=-
    fi
}
if ! gcloud secrets describe "$CLIENT_SECRET_NAME" --project="$PROJECT_ID" &> /dev/null \
    && [ -z "$GOOGLE_OAUTH_CLIENT_SECRET" ]; then
    echo "❌ Error: secret $CLIENT_SECRET_NAME doesn't exist yet; set GOOGLE_OAUTH_CLIENT_SECRET to create it"
    exit 1
fi
printf %s "$GOOGLE_OAUTH_CLIENT_SECRET" | create_secret "$CLIENT_SECRET_NAME"
# Signs session cookies. Delete the secret and redeploy to sign everyone out.
python3 -c "import secrets; print(secrets.token_hex(32), end='')" \
    | create_secret "$SESSION_KEY_NAME"
for secret in "$CLIENT_SECRET_NAME" "$SESSION_KEY_NAME"; do
    gcloud secrets add-iam-policy-binding "$secret" --project="$PROJECT_ID" \
        --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor \
        --quiet > /dev/null
done

# Public URL on purpose: viewer_main.py does the sign-in and refuses to start
# without an allowlist. The ^;^ prefix makes ";" the separator, since the
# allowlist itself may contain commas. The base image matches .python-version;
# the default builder only has newer Pythons.
gcloud run deploy "$SERVICE_NAME" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --source=. \
    --base-image=python311 \
    --service-account="$SA" \
    --allow-unauthenticated \
    --set-build-env-vars="GOOGLE_ENTRYPOINT=gunicorn --workers 1 --threads 8 --timeout 0 viewer_wsgi:app" \
    --set-env-vars="^;^GOOGLE_CLOUD_PROJECT=$PROJECT_ID;GOOGLE_OAUTH_CLIENT_ID=$GOOGLE_OAUTH_CLIENT_ID;DB_VIEWER_ALLOWED_EMAILS=$DB_VIEWER_ALLOWED_EMAILS" \
    --set-secrets="GOOGLE_OAUTH_CLIENT_SECRET=$CLIENT_SECRET_NAME:latest,DB_VIEWER_SECRET_KEY=$SESSION_KEY_NAME:latest" \
    --memory=512Mi \
    --min-instances=0 \
    --max-instances=1

# Cloud Run serves the service under two URLs. Sign-in returns to whichever was
# opened, so each one used needs its redirect URI registered on the client.
URLS=$(gcloud run services describe "$SERVICE_NAME" --project="$PROJECT_ID" \
    --region="$REGION" --format='value(metadata.annotations["run.googleapis.com/urls"])' \
    | python3 -c "import json, sys; print(*json.load(sys.stdin), sep='\\n')")
echo "✅ Deployed. Viewer URLs, each with the OAuth redirect URI it needs:"
for url in $URLS; do
    echo "   $url/debug/db/  ->  $url/auth/callback"
done
