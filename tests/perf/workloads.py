"""What gets sent to each model.

Two families. ``real`` is the production page-analysis request, built by
``DesktopService.build_page_request`` — the repo's documented single source of
truth for that payload — over the frozen page text in ``corpus.json``.
``synthetic`` is filler text at controlled prompt sizes, which needs nothing
outside this module and isolates prefill cost from decode cost.

A payload carries no ``model`` field: the runner fills it in per model, so every
model is measured on byte-identical bytes.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from tests.perf import lmstudio

# Prepended to the head of the *first* message. Placement is load-bearing:
# LM Studio keeps the KV cache between requests, so a repeated prompt comes back
# with a TTFT ~90x too fast and independent of prompt size. Only busting the
# very start of the prompt defeats it — a nonce appended anywhere later leaves
# the shared prefix cached and the TTFT still meaningless.
NONCE_TEMPLATE = "[perf {nonce}] "

SYNTHETIC_SYSTEM = "You are a helpful assistant."
SYNTHETIC_TASK = "\n\nSummarize the text above."

# Ordinary English, not repeated lorem ipsum: highly repetitive text tokenizes
# far denser than prose and would understate every prefill number.
_LEXICON = (
    "the quick brown fox jumps over a lazy dog while morning light spills across "
    "narrow streets where vendors arrange bread olives and citrus under faded awnings "
    "children argue about football results a tram rattles past the old post office "
    "and somewhere behind the market a radio plays something nobody recognises "
    "afternoon brings rain that empties the square within minutes leaving puddles "
    "reflecting shuttered windows and the slow return of pigeons to the cobbles "
    "by evening the cafes fill again conversation rising over cutlery and glasses "
    "until the last tables are folded away and the street lamps hum alone"
).split()


def filler(n_chars: int, seed: int = 0) -> str:
    """A deterministic pseudo-random word stream of about ``n_chars`` characters."""
    if n_chars <= 0:
        return ""
    rng = random.Random(seed)
    out: list[str] = []
    total = 0
    while total < n_chars:
        word = rng.choice(_LEXICON)
        out.append(word)
        total += len(word) + 1
    return " ".join(out)[:n_chars]


@dataclass(frozen=True)
class Prompt:
    id: str
    family: str
    messages: list[dict[str, str]]
    max_tokens: int
    temperature: float = 0.0
    reasoning_effort: Optional[str] = None
    nominal_prompt_tokens: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def body(self, model: str, nonce: Optional[str] = None) -> dict[str, Any]:
        messages = [dict(m) for m in self.messages]
        if nonce:
            messages[0]["content"] = (
                NONCE_TEMPLATE.format(nonce=nonce) + messages[0]["content"]
            )
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        return request


# ---------------------------------------------------------------- synthetic


@dataclass(frozen=True)
class Tokenizer:
    """How this model's tokenizer maps filler characters to prompt tokens.

    There is no local tokenizer and LM Studio exposes no tokenize endpoint, but
    ``usage.prompt_tokens`` is exact, so the ratio is measured with a couple of
    one-token probes instead of guessed.
    """

    overhead_tokens: int
    chars_per_token: float

    def chars_for(self, target_tokens: int) -> int:
        return max(
            0, int((target_tokens - self.overhead_tokens) * self.chars_per_token)
        )


def _probe(model: str, text: str, *, timeout: float, session: Any) -> Optional[int]:
    prompt = Prompt(
        id="calibrate",
        family="synthetic",
        messages=[
            {"role": "system", "content": SYNTHETIC_SYSTEM},
            {"role": "user", "content": text + SYNTHETIC_TASK},
        ],
        max_tokens=1,
    )
    sample = lmstudio.chat(prompt.body(model), timeout=timeout, session=session)
    return sample.prompt_tokens if sample.ok else None


def calibrate(model: str, *, timeout: float, session: Any = None) -> Tokenizer:
    """Measure the fixed prompt overhead and characters-per-token for ``model``."""
    from services.page_prompt import CHARS_PER_TOKEN

    overhead = _probe(model, "", timeout=timeout, session=session)
    if overhead is None:
        return Tokenizer(overhead_tokens=0, chars_per_token=CHARS_PER_TOKEN)

    ratio = CHARS_PER_TOKEN
    probe_chars = int(2048 * ratio)
    measured = _probe(model, filler(probe_chars), timeout=timeout, session=session)
    if measured and measured > overhead:
        ratio = probe_chars / (measured - overhead)
    return Tokenizer(overhead_tokens=overhead, chars_per_token=ratio)


def synthetic_prompts(
    prompt_sizes: list[int],
    output_sizes: list[int],
    *,
    tokenizer: Tokenizer,
    context_length: int,
) -> list[Prompt]:
    """The (prompt size x output cap) grid, minus any cell that cannot fit."""
    prompts = []
    for p_tokens in prompt_sizes:
        text = filler(tokenizer.chars_for(p_tokens))
        for o_tokens in output_sizes:
            if p_tokens + o_tokens + 256 > context_length:
                continue
            prompts.append(
                Prompt(
                    id=f"synth:p{p_tokens}-o{o_tokens}",
                    family="synthetic",
                    messages=[
                        {"role": "system", "content": SYNTHETIC_SYSTEM},
                        {"role": "user", "content": text + SYNTHETIC_TASK},
                    ],
                    max_tokens=o_tokens,
                    nominal_prompt_tokens=p_tokens,
                    meta={"filler_chars": len(text)},
                )
            )
    return prompts


# --------------------------------------------------------------------- real


def _desktop_service(
    model: str, *, context_length: int, max_tokens: int, reasoning_effort: str
) -> Any:
    """A DesktopService that can build page requests with nothing running.

    Its constructor reaches firestore.Client(); pointing that at a dead emulator
    keeps the client lazy and credential-free, and build_page_request never
    touches it. tests/eval/runner.py sets GOOGLE_CLOUD_PROJECT the same way.
    """
    os.environ.update(
        {
            "GOOGLE_CLOUD_PROJECT": os.getenv("GOOGLE_CLOUD_PROJECT", "perf-bench"),
            "FIRESTORE_EMULATOR_HOST": os.getenv(
                "FIRESTORE_EMULATOR_HOST", "localhost:9"
            ),
            "LMS_MODEL": model,
            "LMS_CONTEXT_LENGTH": str(context_length),
            "LMS_MAX_TOKENS": str(max_tokens),
            "LMS_REASONING_EFFORT": reasoning_effort,
        }
    )
    from services.desktop_service import DesktopService

    return DesktopService()


def real_prompts(
    cases: list[dict[str, Any]],
    *,
    context_length: int,
    page_max_tokens: int,
    output_cap: Optional[int],
    reasoning_effort: str,
) -> list[Prompt]:
    """Production page-analysis requests over the frozen corpus.

    ``page_max_tokens`` is what the page budget is planned against (production's
    LMS_MAX_TOKENS), so the prompt is shaped exactly as it would be in the app.
    ``output_cap`` then overrides the completion cap the builder computed: that
    cap is a *floor* which grows into the unused window and reaches ~10k tokens
    on a short page, which at the slower models' rates is half an hour for one
    request. Capping it measures production-shaped prefill and a real decode
    rate; ``--max-tokens 0`` restores the production cap for a full end-to-end run.
    """
    from services.page_scope import scope_for_request

    service = _desktop_service(
        "benchmark",
        context_length=context_length,
        max_tokens=page_max_tokens,
        reasoning_effort=reasoning_effort,
    )
    prompts = []
    for case in cases:
        kinds = scope_for_request(
            case["url"],
            trigger_type=case.get("trigger_type"),
            source=case.get("source"),
            page_signals=case.get("page_signals"),
        )
        request = service.build_page_request(
            url=case["url"],
            title=case["title"],
            content=case["content"],
            user_avoided_ingredients=case.get("user_avoided_ingredients"),
            page_kinds=kinds,
        )
        prompts.append(
            Prompt(
                id=f"real:{case['id']}",
                family="real",
                messages=request["messages"],
                max_tokens=output_cap or request["max_tokens"],
                temperature=request["temperature"],
                reasoning_effort=request.get("reasoning_effort"),
                meta={
                    "case": case["id"],
                    "chars": case.get("chars", len(case["content"])),
                    "page_kinds": list(kinds),
                    "production_max_tokens": request["max_tokens"],
                },
            )
        )
    return prompts
