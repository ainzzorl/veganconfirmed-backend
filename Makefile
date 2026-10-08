.PHONY: run-local-remote run-local run-prod deploy-service deploy-weekly-report deploy-db-viewer geoip

# Project id for the local Firestore emulator. It matches the one desktop-server's
# tooling uses (its `make emulator`, tests/integration/run.sh), so a locally
# running worker and this service share the same emulator data.
EMULATOR_PROJECT ?= desktop-server-test

# IP-to-country database (services/privacy.py). `make geoip` refreshes it.
GEOIP_DB := data/dbip-country-lite.mmdb

$(GEOIP_DB) geoip:
	./scripts/fetch_geoip.sh

run-local-remote: $(GEOIP_DB)
	./scripts/start.sh

# Run the service against a throwaway Firestore emulator instead of the real
# database. `emulators:exec` starts the emulator (with its UI, on port 4000),
# runs scripts/start-local.sh, and shuts the emulator down as soon as it exits.
# The firebase CLI exports FIRESTORE_EMULATOR_HOST into the child process, which
# both firebase_admin and google-cloud-firestore pick up on their own;
# GOOGLE_CLOUD_PROJECT has to be set to match the emulator's project. Neither is
# overwritten by .env, since load_dotenv() leaves real env vars alone.
# scripts/start-local.sh also brings up a desktop-server worker inside the same
# emulator: the two only ever talk through Firestore, so a worker attached to
# the real database can't serve analysis jobs written to the emulator. It first
# frees both the worker's port and the backend's PORT, so leftovers from a run
# that didn't shut down cleanly don't block startup.
run-local: $(GEOIP_DB)
	GOOGLE_CLOUD_PROJECT=$(EMULATOR_PROJECT) \
		firebase emulators:exec --only firestore --ui --project $(EMULATOR_PROJECT) ./scripts/start-local.sh

run-prod: $(GEOIP_DB)
	./scripts/start-prod.sh

# Deployments take their secrets from .env (git-ignored) instead of expecting
# them exported in the caller's shell: `set -a` exports every assignment the
# file makes, so the deploy script and the gcloud calls under it see the same
# values python-dotenv gives the running service. Lines that already say
# `export` work either way.
ENV_FILE ?= .env

# The analysis service (vegassist-analyze). Deploys only from a release branch;
# scripts/deploy_cloud_function.sh enforces that.
deploy-service: $(ENV_FILE)
	set -a && . ./$(ENV_FILE) && set +a && ./scripts/deploy_cloud_function.sh

# The weekly report function, plus its Cloud Scheduler job.
deploy-weekly-report: $(ENV_FILE)
	set -a && . ./$(ENV_FILE) && set +a && ./scripts/deploy_weekly_report_function.sh

# The database viewer on Cloud Run, behind Google sign-in (viewer_main.py).
deploy-db-viewer: $(ENV_FILE)
	set -a && . ./$(ENV_FILE) && set +a && ./scripts/deploy_db_viewer.sh

$(ENV_FILE):
	@echo "Error: $(ENV_FILE) not found. Copy env.example to $(ENV_FILE) and fill it in." >&2; exit 1
