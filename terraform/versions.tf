terraform {
  # 1.7: for_each in import blocks (imports.tf).
  required_version = ">= 1.7"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
  }

  # The bucket is passed in by `make tf-init`, which also creates it.
  backend "gcs" {
    prefix = "veganconfirmed-backend"
  }
}
