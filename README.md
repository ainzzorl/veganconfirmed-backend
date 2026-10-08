# Vegan Confirmed Backend

Flask backend for the Vegan Confirmed browser extension. It takes a page the
extension extracted, decides what kind of page it is, and answers the question
that kind poses: whether a product is vegan, or which dishes on a menu are.
Analysis runs on a local LLM through desktop-server, with Google Gemini as the
fallback, and every call is stored in Firestore. It runs on Google Cloud
Platform. The same core runs behind a local Flask server (`app.py`) and behind
Cloud Functions (`main.py`).

## Running it

Prerequisites: [uv](https://docs.astral.sh/uv/), a Firebase project with
Firestore, and a Gemini API key. The local LLM path also needs LM Studio and a
desktop-server checkout (`$DESKTOP_SERVER_DIR`).

```bash
uv sync
cp env.example .env   # then fill it in; every setting is documented there
```

| Command | What it runs against |
| --- | --- |
| `make run-local` | A throwaway Firestore emulator (UI on `http://localhost:4000`) plus its own desktop-server worker. |
| `make run-local-remote` | The real database, Flask dev server. |
| `make run-prod` | The real database, gunicorn, database viewer off. |

The server listens on `$PORT` (default 5555). The `Makefile` explains each
target.

Which provider serves a request, and which is the fallback, is decided by
`PROVIDER_BY_TRIGGER` in [`analysis_core.py`](analysis_core.py).

## Deployment

| Command | What it deploys |
| --- | --- |
| `make deploy-service` | The analysis function, `vegassist-analyze`. Only from a branch named `release/YYYYMMDD/<number>`. |
| `make deploy-weekly-report` | The weekly usage report email, a private function run by Cloud Scheduler. |

Both read their settings from `.env`. Cloud Functions installs from
`requirements.txt`, not `uv.lock`, so refresh it after changing dependencies:

```bash
uv export --no-dev --no-hashes --no-emit-project --output-file requirements.txt
```

## API

Both endpoints are unauthenticated.

### `POST /api/analyze`

**Request:** `PageAnalysisRequest` in
[`models/content_model.py`](models/content_model.py). Only `url`, `content` and
`timestamp` are required; the rest are hints from the extension.

```json
{
  "url": "https://example.com",
  "title": "Page Title",
  "content": "Extracted text/markdown content of the page...",
  "timestamp": "2024-01-01T12:00:00Z",
  "user_avoided_ingredients": ["palm oil"],
  "place_id": "0x47a84e2e:0x1b9f7c",
  "source": "google_maps_menu_tab",
  "restaurant_name": "Cafe Verde",
  "page_signals": { "og_type": "product", "schema_types": ["Product", "Offer"] }
}
```

Some hints narrow which kinds of page the model is asked about; the rules are
in [`services/page_scope.py`](services/page_scope.py).

**Response:** `PageAnalysis` in
[`models/analysis_model.py`](models/analysis_model.py), under `analysis`, plus
an `analysis_id` naming the stored record (null if the database write failed).
`page_kind` (`shopping_item`, `restaurant_menu` or `other`) says which of
`shopping_item` and `menu` is filled. The verdict vocabulary is defined in the
prompts: [`services/analysis_prompt.py`](services/analysis_prompt.py) and
[`services/menu_prompt.py`](services/menu_prompt.py).

### `POST /api/feedback`

A user's rating of an analysis: `{analysis_id, rating, comment}`
(`AnalysisFeedbackRequest` in [`models/content_model.py`](models/content_model.py)).
It is stored on the analysis's own record. An unknown `analysis_id` is a 404.

## Data

Every analysis, failures included, is stored in the `api_calls` collection as
an `APICallRecord` ([`models/database_model.py`](models/database_model.py)).
Records keep no IP address or URL query string; see
[`services/privacy.py`](services/privacy.py). Country data is
[IP Geolocation by DB-IP](https://db-ip.com) (CC BY 4.0).

To browse the records locally, open `http://localhost:5555/debug/db/`
([`db_viewer.py`](db_viewer.py)). It shows whichever database the server is
pointed at.

## Tests

```bash
uv run pytest              # unit tests
tests/integration/run.sh   # end-to-end: emulator + desktop-server worker + LM Studio
```

The classification eval and the LM Studio speed benchmark have their own docs in
[`tests/eval/README.md`](tests/eval/README.md) and
[`tests/perf/README.md`](tests/perf/README.md).

Formatting, linting and type checking: `uv run black .`, `uv run flake8 .`,
`uv run mypy .`.

## Layout

```
analysis_core.py    # Classify → analyze → persist; shared by both entry points
app.py              # Local Flask entry point
main.py             # Cloud Functions entry points (analyze, weekly report)
db_viewer.py        # Local-only view of the stored API calls
models/             # Request, analysis and stored-record schemas (pydantic)
services/           # Providers, prompts, page scope, Firestore, weekly report
tests/              # Unit tests, tests/integration/, tests/eval/, tests/perf/
```
