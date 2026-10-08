"""Offline tests for the perf harness. No LM Studio, no Node, no network.

Only the two things that would be wrong *and* look plausible are covered: the
decode-rate arithmetic, and the prompt-cache buster. Everything else here is a
thin wrapper over ``requests`` or ``subprocess``, where a mocked test would
restate the code and protect nothing.
"""

import statistics

from tests.perf import lmstudio, workloads
from tests.perf.report import CellResult
from tests.perf.workloads import Prompt

# A real /api/v0/chat/completions response, captured from gpt-oss-20b.
CAPTURED = {
    "choices": [{"finish_reason": "length", "message": {"content": "..."}}],
    "usage": {
        "prompt_tokens": 41,
        "completion_tokens": 300,
        "total_tokens": 341,
        "completion_tokens_details": {"reasoning_tokens": 19},
    },
    "stats": {
        "tokens_per_second": 22.04643213770143,
        "time_to_first_token": 0.069882,
        "generation_time": 13.677526,
        "stop_reason": "maxPredictedTokensReached",
    },
    "model_info": {"arch": "gpt-oss", "quant": "MXFP4"},
    "runtime": {"name": "llama.cpp-linux-x86_64-vulkan-avx2", "version": "2.28.2"},
}


def test_decode_rate_matches_lm_studios_own():
    """``generation_time`` includes prefill, so the decode rate must subtract TTFT.

    Dividing by ``generation_time`` instead looks reasonable and is wrong by
    roughly half on a long prompt, which is why this is pinned to a captured
    response rather than to the formula.
    """
    sample = lmstudio.Sample.from_response(CAPTURED, wall_s=13.9)

    assert sample.decode_tok_s == 300 / (13.677526 - 0.069882)
    assert abs(sample.decode_tok_s - sample.server_tok_s) < 1e-9
    assert not sample.stats_mismatch
    assert sample.prefill_tok_s == 41 / 0.069882
    assert sample.reasoning_tokens == 19


def test_stats_mismatch_flags_a_changed_server_definition():
    body = {**CAPTURED, "stats": {**CAPTURED["stats"], "tokens_per_second": 18.0}}
    assert lmstudio.Sample.from_response(body, wall_s=13.9).stats_mismatch


def test_nonce_lands_at_the_head_of_the_first_message():
    """LM Studio serves a repeated prompt from its KV cache: measured, a second
    identical request came back with TTFT 2.84s -> 0.019s. Only busting the very
    start of the prompt defeats that; a nonce anywhere later leaves the shared
    prefix cached and TTFT still measures the cache."""
    prompt = Prompt(
        id="t",
        family="synthetic",
        messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USR"},
        ],
        max_tokens=8,
    )
    body = prompt.body("m", nonce="deadbeef")

    assert body["messages"][0]["content"].startswith("[perf deadbeef] ")
    assert body["messages"][0]["content"].endswith("SYS")
    assert body["messages"][1]["content"] == "USR"
    # The prompt itself is never mutated, so every rep starts from the same text.
    assert prompt.messages[0]["content"] == "SYS"


def test_median_ignores_a_single_outlier():
    cell = CellResult(id="t", family="synthetic")
    for ttft in (1.0, 1.1, 10.0):
        cell.samples.append(
            lmstudio.Sample(
                ok=True,
                wall_s=ttft,
                ttft_s=ttft,
                generation_s=ttft + 1,
                completion_tokens=100,
                prompt_tokens=100,
                server_tok_s=100.0,
            )
        )
    assert cell.median()["ttft_s"] == 1.1
    assert statistics.mean([1.0, 1.1, 10.0]) > 4  # what a mean would have said


def test_synthetic_prompt_lands_on_its_token_target(monkeypatch):
    """The filler is sized from a measured chars-per-token ratio, not an estimate.

    The repo's ``estimate_tokens`` deliberately over-estimates (3.0 chars/token),
    so a grid built straight off it would undershoot every target.
    """
    truth = 4.2  # the fake tokenizer's real ratio
    overhead = 17

    def fake_chat(request, *, timeout, session=None):
        chars = sum(len(m["content"]) for m in request["messages"])
        return lmstudio.Sample(
            ok=True, wall_s=0.0, prompt_tokens=overhead + int(chars / truth)
        )

    monkeypatch.setattr(lmstudio, "chat", fake_chat)
    tokenizer = workloads.calibrate("fake", timeout=1)
    prompts = workloads.synthetic_prompts(
        [2048], [128], tokenizer=tokenizer, context_length=16384
    )

    measured = fake_chat(prompts[0].body("fake"), timeout=1).prompt_tokens
    assert abs(measured - 2048) / 2048 < 0.02


def test_synthetic_grid_skips_cells_that_cannot_fit():
    tokenizer = workloads.Tokenizer(overhead_tokens=0, chars_per_token=4.0)
    prompts = workloads.synthetic_prompts(
        [512, 8192], [128], tokenizer=tokenizer, context_length=4096
    )
    assert [p.id for p in prompts] == ["synth:p512-o128"]
