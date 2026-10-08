# LM Studio performance benchmark

Measures how fast the local models **serve** a request — time to first token and
tokens/second — and nothing else. Whether the answers are any good is
[`tests/eval/`](../eval/README.md)'s job; this never looks at them.

It talks straight to LM Studio. No Firestore emulator, no desktop-server worker,
no `run.sh`.

```sh
uv run python -m tests.perf --model qwen3-30b-a3b-instruct-2507
uv run python -m tests.perf --all --workload all --reps 3
```

## What it measures

Every number comes from LM Studio's **native** REST API (`/api/v0`, not `/v1`),
which returns a `stats` block on each non-streaming completion:

| reported | from |
|---|---|
| `ttft_s` | `stats.time_to_first_token` |
| `prefill/s` | `prompt_tokens / ttft_s` |
| `decode/s` | `completion_tokens / (generation_time - ttft)` |
| `gen_s` | `stats.generation_time` — server total, prefill included |
| `ptok` / `ctok` / `rtok` | prompt, completion and reasoning tokens |
| `load_s` | wall clock of `lms load` |

Two things worth knowing about that block:

- `generation_time` **includes prefill** and `tokens_per_second` is the **pure
  decode rate**. The identity `tokens_per_second == completion_tokens /
  (generation_time - ttft)` holds exactly. Dividing by `generation_time` instead
  is wrong by roughly half on a long prompt. The report recomputes the rate and
  warns if it stops matching what LM Studio reports.
- **Rate is not volume.** A thinking model can decode as fast as a non-thinking
  one and still take five times as long, because it writes five times as many
  tokens. `decode/s` is the hardware, `ctok`/`rtok` is the model's appetite, and
  a single "seconds per page" figure would confuse the two. Both are in the
  table.

## What it controls

**Load parameters, deliberately.** The instance this box had loaded when the
benchmark was written (16384 context, `--parallel 4`) is not the same machine as
the same model at `--parallel 1`, so every model is unloaded, reloaded at a
stated context and slot count, and verified against `lms ps` before a single
number is taken. `--no-load` measures whatever is already loaded — useful for a
quick check, not for comparing models.

**The KV cache.** LM Studio serves a repeated prompt from cache: measured, a
second identical request came back with TTFT **2.84s → 0.019s**, independent of
prompt size. Every repetition therefore gets a fresh nonce prepended to the
*head of the first message*, which is the only placement that busts it — a
nonce anywhere later leaves the shared prefix cached. `--no-nonce` turns it off,
and is the fastest way to see the effect for yourself:

```sh
uv run python -m tests.perf --no-load --no-nonce --model openai/gpt-oss-20b \
    --workload synthetic --prompt-sizes 4096 --output-sizes 32 --reps 3 -v
```

**Warm-up.** One request after each load is timed and thrown away; it pays
one-off costs that no later request repeats.

## Workloads

### `real` (default)
The production page-analysis request, built by
`DesktopService.build_page_request` — the repo's single source of truth for that
payload — over page text frozen in [`corpus.json`](corpus.json).

The corpus is **committed**, so the benchmark needs neither Node nor the sibling
`veganconfirmed-extension` checkout, and the prompt stays byte-identical over
time. That is the only way a TTFT measured today means anything next to one
measured after a runtime upgrade. Four cases spanning the sizes the extension
really sends:

| case | extracted | what it exercises |
|---|---|---|
| `non_shopping_article` | 0.3k chars | fixed per-request overhead |
| `nuts-for-cheese-black-garlic` | 1.9k chars | a typical shopping item |
| `eureka-restaurant` | 5.1k chars | a restaurant menu |
| `merit-vegan-menu` | 15k chars | fills the page-content budget |

`--refresh-corpus` re-extracts through the extension's own harnesses (needs
`npm install` in `../veganconfirmed-extension`) and rewrites the file, so a
change in extracted sizes shows up as a reviewable diff.

**On `--max-tokens`.** `DesktopService._completion_cap` treats `LMS_MAX_TOKENS`
as a *floor* that grows into whatever the page left unused — on a short page it
returns ~10k tokens, which at the slower models' rates is half an hour for one
request. So the real workload caps the completion at **256** by default: the
prompt is production-shaped, the decode rate is real, but the wall clock is not
production's end-to-end latency. `--max-tokens 0` restores production's own cap
for a (much slower) end-to-end run.

### `synthetic`
Filler prose at controlled prompt sizes, crossed with output caps. Self-contained
— no corpus, no prompt coupling — and it isolates prefill cost from decode cost.

Sizes are hit by **measuring**, not estimating: `usage.prompt_tokens` is exact
ground truth, so two one-token probes per model give the fixed prompt overhead
and that tokenizer's characters-per-token, and the filler is cut to fit. The
table always shows the size the server actually counted, never the target.

```sh
uv run python -m tests.perf --all --workload synthetic \
    --prompt-sizes 512,2048,8192 --output-sizes 128,512
```

## Reading the output

Two tables. The first is per cell, **medians** across repetitions — N is small
and the noise is one-sided (a background process lengthens a run, nothing
shortens it), which is exactly where a mean misleads. Every raw repetition is in
the results JSON, so a mean or a percentile can be recomputed without re-running.

The second is per model. Its `prefill/s` and `decode/s` are **token-weighted**
(`sum(completion_tokens) / sum(decode_s)`), not an average of the per-cell rates,
which would weight a 128-token cell the same as a 512-token one. `flakes` counts
retried requests — gpt-oss fails 5–8% of requests with a harmony-parser error,
and that rate is itself a property of the model.

`stop` abbreviates why generation ended: `cap` (hit `max_tokens` — the expected
outcome for synthetic cells), `eos`, `ctx`.

A run writes `results/perf-<workload>-<timestamp>.json` with the config, the
environment (runtime name and version, host), per-model rollups and every raw
sample. Results are gitignored, as the eval's are: the numbers belong to a
driver version and a thermal state, and committing them invites comparisons that
aren't valid. Keep one deliberately with `git add -f` if it's worth a baseline.

## Sweeps

`--all` benchmarks every text model LM Studio has (embeddings are excluded
automatically). Some are slow enough to dominate a sweep, so:

- `--model-budget` (default 900s) caps wall clock per model; cells that don't
  fit are reported `budget_exceeded` rather than dropped silently;
- `--timeout` (default 300s) caps one request, and three consecutive timeouts
  abandon the model;
- a model that won't load is reported `load_failed` and the sweep continues;
- Ctrl-C writes partial results before exiting.

`--dry-run` prints the models, the cells and the total request count without
calling anything — worth doing before a multi-hour sweep.

By default a run prints one line per model plus anything that went wrong. `-v`
adds the progress log — load, warm-up, the measured tokenizer ratio, every
repetition and a running per-cell summary — on stderr, so `... -v > report.txt`
still leaves the report alone on stdout.

## Testing the harness

```sh
uv run pytest tests/perf
```

Offline; no LM Studio, no Node. It covers the decode-rate arithmetic and the
prompt-cache buster — the two things that would be wrong *and* look entirely
plausible in the report.
