# Adopt the resources the deploy scripts created before this configuration
# existed. Safe to delete once they've been applied.

import {
  for_each = local.apis
  to       = google_project_service.apis[each.key]
  id       = "${var.project_id}/${each.key}"
}

import {
  to = google_storage_bucket.tfstate
  id = "${var.project_id}-tfstate"
}

import {
  to = google_firestore_database.default
  id = "projects/${var.project_id}/databases/(default)"
}

import {
  to = google_service_account.db_viewer
  id = "projects/${var.project_id}/serviceAccounts/vegassist-db-viewer@${var.project_id}.iam.gserviceaccount.com"
}

import {
  to = google_project_iam_member.db_viewer_firestore
  id = "${var.project_id} roles/datastore.viewer serviceAccount:vegassist-db-viewer@${var.project_id}.iam.gserviceaccount.com"
}

import {
  for_each = local.db_viewer_secrets
  to       = google_secret_manager_secret.db_viewer[each.key]
  id       = "projects/${var.project_id}/secrets/${each.key}"
}

import {
  for_each = local.db_viewer_secrets
  to       = google_secret_manager_secret_iam_member.db_viewer[each.key]
  id       = "projects/${var.project_id}/secrets/${each.key} roles/secretmanager.secretAccessor serviceAccount:vegassist-db-viewer@${var.project_id}.iam.gserviceaccount.com"
}

import {
  to = google_service_account.report_scheduler
  id = "projects/${var.project_id}/serviceAccounts/vegassist-report-scheduler@${var.project_id}.iam.gserviceaccount.com"
}

import {
  to = google_cloud_run_v2_service_iam_member.report_scheduler_invoker
  id = "projects/${var.project_id}/locations/${var.region}/services/vegassist-weekly-report roles/run.invoker serviceAccount:vegassist-report-scheduler@${var.project_id}.iam.gserviceaccount.com"
}

import {
  to = google_cloud_scheduler_job.weekly_report
  id = "projects/${var.project_id}/locations/${var.region}/jobs/vegassist-weekly-report-scheduler"
}
