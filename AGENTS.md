# AGENTS.md

Shared instructions for any AI coding agent working in this repository.

## Git workflow

- Committing directly to `main` is fine by default.
- Do not create branches unless explicitly asked to.
- Do not commit or push unless explicitly asked to.
- Commit message should be brief. One sentence is usually enough. One paragraph at the most.

## Infrastructure

- GCP infrastructure lives in `terraform/`; Cloud Functions and Cloud Run services are deployed by the `scripts/deploy_*.sh` scripts instead.
- Never run mutating Terraform commands without explicit user approval: `make tf-apply`, `terraform apply`, `destroy`, `import`, `state` subcommands that write, `taint`/`untaint`, `force-unlock`, and `make tf-init` while the state bucket doesn't exist yet (it creates it).
- `make tf-plan`, `terraform validate` and `terraform fmt` are fine to run.

## Code style

- Keep comments and documentation concise; don't restate what the code already says.
- Put comments and docs next to the code they describe so they don't drift out of date.

## Tests

- Only add tests that earn their keep; don't write tests just to have tests.
