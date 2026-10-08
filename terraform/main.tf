# Everything this repo needs in GCP except the Cloud Functions and Cloud Run
# services themselves, which the scripts/deploy_*.sh scripts deploy.

provider "google" {
  project = var.project_id
  region  = var.region
}

# Lists shared with imports.tf.
locals {
  apis = toset([
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudfunctions.googleapis.com",
    "cloudscheduler.googleapis.com",
    "firestore.googleapis.com",
    "iam.googleapis.com",
    "logging.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
  ])
  db_viewer_secrets = toset(["db-viewer-oauth-client-secret", "db-viewer-secret-key"])
}

resource "google_project_service" "apis" {
  for_each = local.apis
  service  = each.key
  # The project is shared with other repos; never switch an API off from here.
  disable_on_destroy = false
}

# Holds this configuration's state. `make tf-init` creates it with gcloud
# (Terraform can't create its own backend) and Terraform manages it from then on.
resource "google_storage_bucket" "tfstate" {
  name                        = "${var.project_id}-tfstate"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_firestore_database" "default" {
  name                        = "(default)"
  location_id                 = "nam5"
  type                        = "FIRESTORE_NATIVE"
  concurrency_mode            = "PESSIMISTIC"
  app_engine_integration_mode = "DISABLED"
  delete_protection_state     = "DELETE_PROTECTION_DISABLED"
  # Removing it from Terraform only forgets it.
  deletion_policy = "ABANDON"

  lifecycle {
    prevent_destroy = true
  }
}

# --- Database viewer (scripts/deploy_db_viewer.sh) ---

resource "google_service_account" "db_viewer" {
  account_id   = "vegassist-db-viewer"
  display_name = "Database viewer (read-only)"
}

resource "google_project_iam_member" "db_viewer_firestore" {
  project = var.project_id
  role    = "roles/datastore.viewer"
  member  = google_service_account.db_viewer.member
}

# Containers only: the deploy script adds the values, so they stay out of the
# state (scripts/deploy_db_viewer.sh).
resource "google_secret_manager_secret" "db_viewer" {
  for_each  = local.db_viewer_secrets
  secret_id = each.key

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_iam_member" "db_viewer" {
  for_each  = google_secret_manager_secret.db_viewer
  secret_id = each.value.id
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.db_viewer.member
}

# --- Weekly report (scripts/deploy_weekly_report_function.sh) ---

resource "google_service_account" "report_scheduler" {
  account_id   = "vegassist-report-scheduler"
  display_name = "Vegan Confirmed weekly report scheduler"
}

# The function is private; the scheduler is its only non-owner invoker. Gen2
# invoker bindings live on the function's Cloud Run service.
resource "google_cloud_run_v2_service_iam_member" "report_scheduler_invoker" {
  name     = "vegassist-weekly-report"
  location = var.region
  role     = "roles/run.invoker"
  member   = google_service_account.report_scheduler.member
}

# Read for its URL, so the function has to be deployed before the first apply.
data "google_cloudfunctions2_function" "weekly_report" {
  name     = "vegassist-weekly-report"
  location = var.region
}

locals {
  # With the trailing slash the Scheduler API stores the URI with, so the OIDC
  # audience matches it.
  weekly_report_url = "${trimsuffix(data.google_cloudfunctions2_function.weekly_report.service_config[0].uri, "/")}/"
}

resource "google_cloud_scheduler_job" "weekly_report" {
  name      = "vegassist-weekly-report-scheduler"
  region    = var.region
  schedule  = "0 7 * * 6" # Saturdays, 7:00 UTC
  time_zone = "Etc/UTC"

  http_target {
    uri         = local.weekly_report_url
    http_method = "POST"
    headers     = { "Content-Type" = "application/json" }
    body        = base64encode("{\"trigger\": \"scheduled\"}")

    oidc_token {
      service_account_email = google_service_account.report_scheduler.email
      audience              = local.weekly_report_url
    }
  }

  # The API's defaults, spelled out so Terraform doesn't try to unset them.
  retry_config {
    min_backoff_duration = "5s"
    max_backoff_duration = "3600s"
    max_doublings        = 5
    max_retry_duration   = "0s"
  }

  depends_on = [google_project_service.apis]
}
