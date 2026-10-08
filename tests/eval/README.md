# Analysis classification eval

A standalone **benchmark** for page analysis — *not* a pass/fail test. It runs a
corpus of saved pages through the real analysis stack and **reports** per-field
accuracy and other metrics. A wrong prediction lowers the score; it never raises.

It scores both halves of what the backend does: the `page_kind` classification
(`shopping_item` / `restaurant_menu` / `other`), and then the branch that kind
selects — the product verdict for an item, the per-dish verdicts for a menu.

## What it does

For each case it:

1. Runs the extension's **real** extractor over the saved fixture via a Node
   harness — the exact payload the extension would send to the backend. Either
   `extractPageContent` (from `veganconfirmed-extension/content.js`) over the
   saved HTML via jsdom, or `extractPdfText` (from
   `veganconfirmed-extension/pdf_extract.mjs`) over a saved PDF via pdf.js.
2. Feeds it through `AnalysisCore.analyze_page` (with the database disabled)
   using the configured provider (desktop LM Studio or real Gemini) and model.
3. Compares the predicted analysis against the case's declared expectations and
   accumulates metrics.

It then prints a report and writes a machine-readable
`results/results-<provider>-<model>-<timestamp>.json` (one per model), plus an
optional combined HTML report (`--html`).

## The corpus

`cases/` holds one pair of files per case — a fixture and a manifest:

- `cases/<id>.html` — a saved page snapshot (include nav/footer etc.; the
  extension's extractor strips them, so this exercises that path), **or**
- `cases/<id>.pdf` — a saved PDF menu, run through pdf.js instead of jsdom.
  Only the PDF's text layer is read; see "PDF cases" below.
- `cases/<id>.json` — expectations + metadata:

  ```json
  {
    "url": "https://example.com/oat-milk",
    "title": null,
    "category": "food",
    "user_avoided_ingredients": ["honey"],
    "expect": {
      "page_kind": "shopping_item",
      "is_shopping_item": true,
      "is_vegan": true,
      "is_cruelty_free": null,
      "user_avoided_ingredients_include": ["honey"]
    },
    "notes": "..."
  }
  ```

  - `url` is required (the saved fixture doesn't carry its origin URL). `title`
    defaults to the page's `<title>` unless overridden here.
  - `skip` (optional) parks a case: give it a reason string and the case is left
    out of the corpus entirely rather than scored as a failure, so a case the
    stack cannot yet serve doesn't drag every aggregate down. `--case <id>` runs
    it anyway, which is how you work on one.
  - Only fields listed under `expect` are scored — including an explicit `null`
    (a real expectation, e.g. `is_vegan: null` for a non-shopping page). A field
    left out is "don't care".
  - Scored fields: `page_kind`, `is_shopping_item`, `is_vegan`,
    `is_cruelty_free`, `vegan_friendliness`. Any of them may be given **a list
    of accepted values** instead of one value — `"vegan_friendliness": ["medium",
    "high"]` — for the genuinely subjective ones, where insisting on a single
    answer would score a defensible one as wrong.
  - `user_avoided_ingredients` (top level) is passed into the analysis;
    `user_avoided_ingredients_include` (under `expect`) is the recall check that
    those show up in the result.
  - `ingredients` (under `expect`, shopping items) checks the two ingredient
    lists the analysis returns: each term under `explicit` / `inferred` must
    appear in `shopping_item.explicit_ingredients` / `inferred_ingredients`, and
    no term under `excluded` may appear in either — which is how a vegan
    product case catches an invented animal ingredient. A term may be a list of
    accepted spellings; matching is a case-insensitive substring.

    ```json
    "ingredients": {
      "explicit": ["wheat gluten", ["merino", "wool"]],
      "excluded": ["pork", "milk"]
    }
    ```
  - An optional top-level `request` block is merged into the request the backend
    receives, for the hints the extension sends when it has them:

    ```json
    "request": {
      "source": "google_maps_menu_tab",
      "place_id": "ChIJ...",
      "restaurant_name": "Kembali Warung"
    }
    ```

    `restaurant_name` is the only one that changes what the model sees —
    `AnalysisCore` passes it in place of the title — so it is how a case
    exercises the Google Maps path (see `cases/google_maps_menu.json`). The rest
    are recorded, not prompted.

### Menu expectations

A `restaurant_menu` case can declare the dishes it expects under `expect.menu`,
which is scored by `menu_scoring.py`:

```json
"expect": {
  "page_kind": "restaurant_menu",
  "vegan_friendliness": ["medium", "high"],
  "menu": {
    "restaurant_name": "The Olive Branch",
    "dishes": [
      { "name": "Falafel Plate",     "verdict": "vegan" },
      { "name": "Roasted Aubergine", "verdict": "not_vegan" },
      { "name": "Linguine Pomodoro", "verdict": ["likely_vegan", "unclear"] }
    ]
  }
}
```

- **Dish names are matched, not compared.** The model returns the name as the
  menu writes it, so "Falafel Plate" matches "Falafel Plate (VG)" and
  "Hummus & Flatbread" matches "Hummus and Flatbread". Matching is exact-first,
  then whole-token containment.
- `verdict` is one of `vegan` / `likely_vegan` / `veganizable` / `not_vegan` /
  `unclear`, or a list of the ones you'd accept. An unknown verdict raises
  rather than scoring every dish wrong. `veganizable` is for a dish the menu
  itself offers in a vegan version (a choice of beef, chicken or tofu) — see
  `cases/cafe_menu_no_prices.json`, where oat milk is free on any drink.
- The two ingredient lists the model returns (`explicit_ingredients`,
  `inferred_ingredients`) are shown in the HTML dish breakdown but not scored: a
  case declares verdicts, and they are the evidence the verdict should follow
  from.
- **A dish the model lists that the case doesn't declare is never scored.** The
  dish list is whatever the case wants checked — usually a spot-check of a menu
  that runs to sixty dishes — so counting the rest against the model would
  punish it for reading them. They are still reported: `+n` in the `menu`
  column, an "extra dishes" line in the aggregates, and their own rows (verdict,
  ingredients, reason) at the foot of the HTML dish breakdown, which is where
  you notice a model inventing a dish or splitting one in two.

Recall (dishes found) and verdict accuracy (over the dishes that were found) are
reported separately, so a model that lists three dishes and rates them perfectly
does not outscore one that lists all nine and gets one wrong.

### PDF cases

Restaurants publish menus as PDFs at least as often as HTML, so a case can be a
saved `cases/<id>.pdf` instead of a snapshot. The extension reads the PDF's own
text layer with pdf.js and sends it in the ordinary `content` field, so these
cases exercise the same backend path as every other page — including the local
desktop provider, which no image could reach.

That only works when the text layer *is* the menu. Design tools routinely
convert menu type to vector outlines, which leaves the prices as the only real
text in the file: enough to look like a menu, nowhere near enough to be one. The
extractor counts words of three or more letters and gives up below 40 per page
(`MIN_LETTER_WORDS_PER_PAGE` in `pdf_extract.mjs`), reporting the case as an
error rather than inviting the model to invent dish names for the prices it can
see. `cases/hummus-mountain-view.json` is exactly that PDF, and is `skip`ped
until there is a vision path — a two-page menu whose text layer holds 24 words.

**Adding a case:** save the page's HTML to `cases/<id>.html` (or the menu to
`cases/<id>.pdf`), write `cases/<id>.json`, run the eval.

## Testing the harness itself

The scoring is pure functions over dicts, so it has its own offline tests — no
provider, no network, no emulator:

```sh
uv run pytest tests/eval/test_metrics.py
```

Worth running before a corpus run: a dish matcher that fails to pair "Falafel
Plate" with "Falafel Plate (VG)" reports a missed dish and blames the model.

## Running

### One-time setup
Install the Node harnesses' dependencies (jsdom, pdf.js) in the extension repo:

```sh
cd ../veganconfirmed-extension && npm install
```

Combos to benchmark are always given explicitly as `--run provider:model`
(providers: `gemini`, `desktop`) — there are no default providers or models.
Repeat `--run` or comma-separate to benchmark several combos in one run.

### Gemini (no GCP, no emulator)
Calls the real Gemini API directly; nothing touches Firestore/GCP.

```sh
GEMINI_API_KEY=... python -m tests.eval --run gemini:gemini-2.5-flash-lite
```

### Desktop LM Studio (Firestore emulator + worker)
Needs the emulator + desktop-server worker + LM Studio; `run.sh` brings the
first two up (mirrors `tests/integration/run.sh`):

```sh
tests/eval/run.sh --run desktop:openai/gpt-oss-20b
# extra flags are forwarded, e.g.:  tests/eval/run.sh --run desktop:qwen/qwen3.5-9b --case vegan_food
# --keep-emulator leaves the emulator/UI up afterwards for inspection
```

`run.sh` preloads the first desktop model named in your `--run` combos (the
worker loads any others on demand), so it won't pull an unrelated model.

With `DESKTOP_SERVER_HOST` set (in `.env` or the environment, as for
`make run-local`), the worker is built and run on that host over SSH instead,
from `$DESKTOP_SERVER_DIR` there. The emulator and the eval stay on this machine.

### Analyzing several cases at once
By default the corpus runs one case at a time, so a run costs the sum of every
case's latency. `--max-batch-size N` analyzes up to N cases concurrently, a
batch at a time:

```sh
tests/eval/run.sh --run desktop:openai/gpt-oss-20b --max-batch-size 4
```

Cases are fed in **category order**, so the cases in flight come from the same
category wherever the corpus allows it and only mix at a category boundary. A
slot that frees takes the next case immediately rather than waiting for the
others it started with — cases differ in cost by an order of magnitude (a
product page in 13s against a menu in 114s), and on this corpus making them
wait cost 111s of a 277s run in idle slots. Results are always reported in
corpus order, whichever order they finished in.

Three things have to allow N at once, and `run.sh` arranges all of them from the
flag:

- the eval keeps N analyses in flight (this flag);
- the desktop-server worker claims N jobs at once
  (`DESKTOP_SERVER_MAX_CONCURRENT_JOBS`) — without it the worker runs jobs
  strictly one at a time and the extra ones just wait;
- the model is loaded with N prediction slots (`LMS_PARALLEL`, passed to
  `lms load --parallel`) — without it LM Studio queues the extra requests
  internally;
- the model is loaded with **N times the context** (`16384 × N`), because the
  slots divide the loaded window between them rather than each getting one of
  their own. Without this, four requests share one 16384-token window and every
  one of them dies with `Context size has been exceeded`. Each request is still
  planned against the ordinary 16384, exactly as production does; only the load
  is scaled. An instance carried over from an earlier run with fewer slots or
  less context is reloaded.

KV cache use therefore scales with N (measured: gpt-oss-20b at 65536/4 slots
loads in ~11.3 GiB). The pickup timeout is also raised (10s → 30s) for batched
runs, since N jobs land at once and the worker claims them in turn.

Per-case latencies under batching include contention with the other cases in
the batch, so they are not comparable with single-stream timings — compare
totals across batch sizes, not per-case numbers. The batch size is recorded in
the report header and in the results JSON (`config.max_batch_size`) so a
batched run is never mistaken for a sequential one.

For a gemini-only run the flag works the same way (N API calls in flight) and
needs none of the desktop scaffolding.

`GEMINI_API_KEY` (and other vars) are read from the repo-root `.env` if present,
so you can drop the key there instead of exporting it each time. Anything already
set in the environment wins over `.env`.

### Several combos in one run
Repeat `--run` (or comma-separate) to benchmark several combos — including a mix
of providers, e.g. a Gemini model against a local LM Studio model. Each page is
extracted once and replayed against every combo, a text report is printed per
combo, and each combo switches the env `AnalysisCore` reads, exactly as in
production:

```sh
# Needs the desktop scaffolding for the desktop combo, so go through run.sh
# (which also satisfies the gemini combo via .env / GEMINI_API_KEY):
tests/eval/run.sh --html \
    --run gemini:gemini-2.5-flash-lite --run desktop:openai/gpt-oss-20b
```

For desktop combos the chosen model must be loadable by the running LM Studio
worker. Gemini-only runs don't need `run.sh`.

### Response cache
Both providers are expensive to re-run — Gemini charges per call, a local model
spends minutes of GPU time on a menu — so every response is cached on disk under
`tests/eval/.cache/<provider>/<model>/<hash>.json` (gitignored). A re-run that
asks for the same thing is served from disk: no API call, and for desktop no job
at all (the worker, LM Studio and even the heartbeat check are skipped).

The cache key is the **exact request**, so nothing that decides the answer can
be left out of it:

- **gemini** — model name, system instruction (which includes the prompt and the
  user's avoided ingredients), page content, and generation config/response
  schema.
- **desktop** — the whole OpenAI-format request: model, messages (system prompt,
  page content, avoided ingredients), `reasoning_effort`, `temperature` and
  `max_tokens` (which moves with `LMS_CONTEXT_LENGTH`/`LMS_MAX_TOKENS` and with
  how much of the page fits). So `--run desktop:m:low` and `--run desktop:m:high`
  never share an entry.

Edit the prompt, re-extract a page, switch model or effort and the key changes —
a stale answer can never be served. What is *not* in the key is the machine
behind it: LM Studio's version, the quantization it loaded, the sampling seed,
Gemini's server-side model revision. Entries are only as comparable as that
setup is stable.

Only the call itself is cached; prompt building, `normalize_page_analysis` and
scoring run for real on every case, and failures (an unavailable worker, a
failed or unparseable job, an API error) are never cached. The desktop wrap sits
just above response parsing, so the *parsed* payload is what is stored — a
change to that parsing (including the repair of a cut-short answer) needs
`--refresh-cache`.

Each entry also stores how long the real call took, and a cached case reports
that stored duration as its latency — so a replayed run shows the timings the
calls actually had rather than ~0s. Entries cached before timings were recorded
still report 0s until they are refreshed. For a true speed benchmark (fresh
network conditions, a cold model, current contention) use `--no-cache` or
`--refresh-cache`. The run ends with a per-provider line, e.g.
`desktop cache: 12/14 served from disk, 2 real desktop-server job(s)`.

```sh
python -m tests.eval --run gemini:gemini-2.5-flash-lite            # uses the cache
python -m tests.eval --run gemini:... --refresh-cache              # re-call, overwrite
python -m tests.eval --run gemini:... --no-cache                   # bypass entirely
python -m tests.eval --run gemini:... --cache-dir /path/to/cache   # elsewhere
```

Desktop runs still go through `run.sh` even when every case is cached: the
runner requires `FIRESTORE_EMULATOR_HOST` before it knows which cases will hit.

Drop cached answers for a model by deleting its directory:
`rm -rf tests/eval/.cache/gemini/gemini-2.5-flash-lite`,
`rm -rf tests/eval/.cache/desktop/openai_gpt-oss-20b`.

### HTML report
Add `--html` to also write a self-contained HTML report. It opens with a legend
explaining how to read it, and under each case row surfaces the model's own
**confidence and explanation text** (summary, cruelty-free rationale, avoided
ingredients found) so a verdict can be understood, not just scored. Menu cases additionally get an expandable **Menu breakdown** — every
expected dish against what the model called it, with the animal-derived
ingredients the model named for it (plus any avoid-list ingredient it found) and
the model's own reason, followed by the same columns for every dish the model
listed that the case doesn't declare (shown for reference, never scored). With more than one combo it also includes a cross-model
comparison table and a per-case pass/fail matrix.

```sh
python -m tests.eval --run gemini:gemini-2.5-flash-lite --html              # auto-named under results/
python -m tests.eval --run gemini:gemini-2.5-flash-lite --html report.html  # explicit path
```

### Useful flags
- `--case <id>` — run a single case (even one the manifest `skip`s).
- `--type <item|menu|other>` — run only cases of that type, i.e. whose
  `expect.page_kind` is `shopping_item` / `restaurant_menu` / `other`. Repeat
  for several.
- `--cases <dir>` — point at a different corpus.
- `--node <path>` — path to the extension's `tools/extract_content.js`.
- `--node-pdf <path>` — path to the extension's `tools/extract_pdf.js`.
- `--run provider:model` — combo(s) to benchmark (required); repeat or
  comma-separate to run several, mixing providers, in one pass.
- `--max-batch-size <n>` — analyze up to n cases concurrently, a batch at a
  time, grouping same-category cases together (default 1: one at a time).
- `--out <file>` — results JSON destination (single-combo only).
- `--html [path]` — also write an HTML report (auto-named if no path).
- `--cache-dir <dir>` — where cached responses live, one subdirectory per
  provider (default `tests/eval/.cache`).
- `--refresh-cache` — re-run every case against the provider and overwrite the
  cached responses.
- `--no-cache` — ignore the cache and don't write to it.

## Reading the report

- Per-case table: the page-kind scope the case ran under, then actual value +
  ✓/✗ per scored field (`kind` first, since every other field is downstream of
  it), avoided-ingredient recall, the menu summary, overall ✓/✗, latency.
- The `scope` column is what `services/page_scope.py` let the model choose from:
  `all`, or `item`/`menu` when a rule dropped the other branch (the rule's name
  is in the notes column, and the tooltip in the HTML report). It is not scored
  — it is decided from the request, before any model runs — but a wrong `kind`
  reads very differently when the right branch was never on offer.
- The `menu` column is *dishes found / dishes expected*, then per-dish verdict
  accuracy, then `+n` if the model returned dishes the case doesn't list (shown,
  not scored).
- Per-field accuracy with a confusion breakdown (`expected->actual`: count).
- Menu analysis: mean dish recall, verdict accuracy, extra dishes (unscored),
  and a verdict
  confusion breakdown — the last is what exposes a model that plays it safe by
  calling everything `unclear`, or one that softens `not_vegan` into
  `likely_vegan`.
- By type (the case's expected `page_kind`: item / menu / other): fully-correct
  count, errors, mean latency and per-field accuracy over that type's cases
  alone. In the HTML report this is a table above each run's cases, plus a
  fully-correct column per type in the model comparison. The results JSON has
  the full aggregate per type under `aggregate.by_type`.
- Per-category fully-correct counts and error counts.
- Page-kind scope: how many cases were asked each question, broken down by the
  rule that narrowed them.

`results/*.json` records the config, aggregates, and every case's prediction so
runs can be compared over time. The HTML report (`--html`) renders the same data
for browsing and, for multi-model runs, side-by-side comparison.
