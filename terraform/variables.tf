variable "project_id" {
  description = "GCP project; `make` passes PROJECT_ID from .env."
  type        = string
}

variable "region" {
  type    = string
  default = "us-central1"
}
