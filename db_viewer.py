"""Local-only web viewer for the Firestore data this service writes.

A debugging aid: it renders the ``api_calls`` collection as a browsable list of
requests with their stored page content, verdicts, how long each call took and
provider/token details, plus a detail page per record showing every field of
the raw document.

It is deliberately unauthenticated, and therefore only ever served to a caller
on the same machine.

A record served by the desktop-server carries the ID of the job it ran as, so
the detail page follows that link into the ``lms_jobs`` collection and shows the
job beside the record: its status, the prompt that actually went to LM Studio,
and the completion that came back.

Reads are intentionally simple: the most recent N documents are fetched ordered
by ``created_at`` and every filter is applied in Python over that window. That
keeps the tool free of composite-index requirements (which an equality filter
combined with an ordering would otherwise need), at the cost of only ever
searching the scanned window — which the page states.
"""

import ipaddress
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, Flask, abort, jsonify, render_template_string, request

from models.database_model import STATUS_OK
from services.desktop_service import jobs_collection_name
from services.firestore_service import STORED_CONTENT_CHAR_LIMIT
from services.menu_prompt import ITEM_VERDICTS
from services.page_scope import describe_scope

logger = logging.getLogger(__name__)

URL_PREFIX = "/debug/db"

# How many of the most recent documents a request reads before filtering.
DEFAULT_SCAN_LIMIT = 500
MAX_SCAN_LIMIT = 5000
# How many of the matching documents one page of the table shows.
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 500

# Fields the free-text search looks through.
SEARCH_FIELDS = (
    "item_url",
    "origin_url",
    "title",
    "summary",
    "content",
    "installation_id",
    "ip_hash",
    "country",
    "model",
    "lms_job_id",
    "error",
)

# Columns of the list table, in order: (document field, header).
LIST_COLUMNS = (
    ("created_at", "Time"),
    ("page_kind", "Kind"),
    ("page_scope_kinds", "Scope"),
    ("verdict", "Verdict"),
    ("shopping_item.confidence_level", "Confidence"),
    ("title", "Page"),
    ("service", "Service"),
    ("model", "Model"),
    ("tokens", "Tokens"),
    ("duration", "Took"),
    ("extension_version", "Ext"),
    ("installation_id", "Installation"),
    ("feedback.rating", "Feedback"),
)

# Fields shown in the detail page, grouped. Anything not listed here still
# shows up in the raw-document section, so a new field is never invisible.
DETAIL_GROUPS = (
    (
        "Request",
        (
            "created_at",
            "item_url",
            "origin_url",
            "title",
            "language",
            "user_avoided_ingredients",
            "trigger_type",
            "trigger_element_text",
            "trigger_element_selector",
            "page_signals",
            "page_scope_rule",
            "page_scope_kinds",
            "extension_version",
            "installation_id",
            "ip_hash",
            "country",
            "user_agent",
        ),
    ),
    (
        "Result",
        (
            "status",
            "error",
            "page_kind",
            "summary",
        ),
    ),
    # The two per-kind halves. Only one of them is filled on any record, and the
    # other renders as a group of dashes — which is the point: the shape says
    # which question this page was asked.
    (
        "Shopping item",
        (
            "shopping_item.explicit_ingredients",
            "shopping_item.inferred_ingredients",
            "shopping_item.animal_derived_ingredients",
            "shopping_item.is_vegan",
            "shopping_item.is_cruelty_free",
            "shopping_item.cruelty_free_explanation",
            "shopping_item.confidence_level",
        ),
    ),
    # "menu.items" is deliberately absent, for the reason DISH_COLUMNS gives.
    ("Menu", ("menu.restaurant_name", "menu.vegan_friendliness")),
    (
        "Feedback",
        (
            "feedback.rating",
            "feedback.comment",
            "feedback.created_at",
            "feedback.updated_at",
        ),
    ),
    ("Provider", ("service", "model", "token_usage", "lms_job_id")),
    # "duration" is not a stored field: it is the span between the two that are.
    ("Timing", ("started_at", "finished_at", "duration")),
)

# Columns of the per-dish table on the detail page, in order: (row key, header).
# `items` is deliberately absent from DETAIL_GROUPS above: a list of dish maps
# is a table, not a field value.
DISH_COLUMNS = (
    ("name", "Dish"),
    ("section", "Section"),
    ("verdict", "Verdict"),
    ("reason", "Reason"),
    ("explicit_ingredients", "Listed ingredients"),
    ("inferred_ingredients", "Inferred ingredients"),
)

# The avoid-list column is only shown when a dish actually names something, so
# the common case (no avoid-list on the request) keeps a narrower table.
AVOIDED_COLUMN = ("user_avoided_ingredients", "Avoided")

# Labels for fields whose name does not survive ``_label``'s prettifying.
FIELD_LABELS = {
    "lms_job_id": "Desktop-server job",
    "page_scope_kinds": "Page-kind scope",
    "page_scope_rule": "Page-kind scope rule",
    "error": "Error",
    "duration": "Took",
    "shopping_item.explicit_ingredients": "Listed ingredients",
    "shopping_item.inferred_ingredients": "Inferred ingredients",
    "shopping_item.animal_derived_ingredients": "Animal-derived ingredients",
    "shopping_item.is_vegan": "Vegan",
    "shopping_item.is_cruelty_free": "Cruelty free",
    "shopping_item.cruelty_free_explanation": "Cruelty-free explanation",
    "shopping_item.confidence_level": "Confidence",
    "menu.restaurant_name": "Restaurant",
    "menu.vegan_friendliness": "Vegan friendliness",
}

# Fields of a desktop-server job document shown above its prompt, in order.
# Everything else the worker wrote is still in the raw job document below them.
JOB_FIELDS = ("status", "model", "created_at", "updated_at", "error")

# Fields offered as filter selects, in order: (document field, label).
FILTER_SELECTS = (
    ("status", "Status"),
    ("page_kind", "Kind"),
    ("page_scope_kinds", "Scope"),
    ("service", "Service"),
    ("model", "Model"),
    ("shopping_item.confidence_level", "Confidence"),
    ("installation_id", "Installation"),
    # The point of collecting feedback at all: filtering to "down" is how the
    # analyses a user disagreed with get found.
    ("feedback.rating", "Feedback"),
)

# Per-dish verdict to CSS class. "unclear" is left unstyled along with anything
# a future prompt adds.
VERDICT_CLASSES = {
    "vegan": "v-vegan",
    "likely_vegan": "v-maybe",
    "veganizable": "v-maybe",
    "not_vegan": "v-not",
}


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment variable (mirrors analysis_core._env_flag)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def viewer_enabled() -> bool:
    """Whether the viewer should be registered at all."""
    return _env_flag("ENABLE_DB_VIEWER", True)


def _request_is_local() -> bool:
    """True when the caller is on this machine and no proxy sits in between.

    Both halves matter: behind a reverse proxy the peer address *is* loopback,
    so the forwarding headers are what distinguish "someone on the internet"
    from "someone at this keyboard".
    """
    for header in ("X-Forwarded-For", "X-Real-IP", "Forwarded"):
        if request.headers.get(header):
            return False
    try:
        return ipaddress.ip_address(request.remote_addr or "").is_loopback
    except ValueError:
        return False


def _jsonable(value: Any) -> Any:
    """Convert Firestore values into something ``jsonify`` can render."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _format_time(value: Any) -> str:
    """Render a timestamp in this machine's timezone.

    Firestore hands back UTC; a tool that is only ever read at the keyboard it
    runs on is easier to line up against the logs in local time.
    """
    if isinstance(value, datetime):
        local = value.astimezone() if value.tzinfo else value
        return local.strftime("%Y-%m-%d %H:%M:%S")
    return "" if value is None else str(value)


def _duration_s(doc: Dict[str, Any]) -> Optional[float]:
    """How long a call took, as the span between its two stored timestamps.

    ``None`` when either end is missing — records written before requests were
    timed have neither.
    """
    started, finished = doc.get("started_at"), doc.get("finished_at")
    if not isinstance(started, datetime) or not isinstance(finished, datetime):
        return None
    return (finished - started).total_seconds()


def _format_duration(seconds: Optional[float]) -> str:
    """Render a duration for a reader.

    Sub-second calls are the fast path worth seeing in milliseconds; anything
    longer is easier to compare in seconds.
    """
    if seconds is None:
        return ""
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"


def _label(field: str) -> str:
    """Turn a document field name into a column/row label."""
    return (
        FIELD_LABELS.get(field)
        or field.rsplit(".", 1)[-1].replace("_", " ").capitalize()
    )


def _dish_maps(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The stored dishes of a record, skipping anything malformed.

    Documents are read raw, so ``menu`` is whatever was written — including
    nothing at all on a record that is not a menu.
    """
    items = (doc.get("menu") or {}).get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _status(doc: Dict[str, Any]) -> str:
    """How the request ended.

    Records written before failures were stored carry no status, and every one
    of them was answered — so an absent status reads as ``ok``, which also
    keeps the status filter from hiding them.
    """
    return str(doc.get("status") or STATUS_OK)


def _scope_label(doc: Dict[str, Any]) -> str:
    """The page kinds the model was offered, as the short name the logs use.

    The document stores the kinds themselves; the label is derived so the
    filter and the table read as one word rather than as a list. Records
    written before the scope was stored have none, which reads as empty
    everywhere.
    """
    kinds = doc.get("page_scope_kinds")
    if not isinstance(kinds, (list, tuple)) or not kinds:
        return ""
    return describe_scope(tuple(str(kind) for kind in kinds))


def _path_value(doc: Dict[str, Any], field: str) -> Any:
    """The raw value at a field name, which may be a dotted path.

    The per-kind halves of the analysis live in the ``shopping_item`` and
    ``menu`` maps, so a field is addressed as "shopping_item.is_vegan".
    """
    value: Any = doc
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _field_value(doc: Dict[str, Any], field: str) -> Any:
    """The value a filter compares against, normalized where the field needs it."""
    if field == "status":
        return _status(doc)
    if field == "page_scope_kinds":
        return _scope_label(doc)
    return _path_value(doc, field)


def _verdict(doc: Dict[str, Any]) -> str:
    """One-word verdict for the list table.

    A request that produced no analysis has no verdict to show, so it says so
    instead; the reason is on the detail page (and under the URL here).

    Otherwise: only the shopping branch has a boolean verdict; a menu has one
    per dish, so it is shown as its dish count, with the verdicts themselves on
    the detail page.
    """
    if _status(doc) != STATUS_OK:
        return "failed"
    if doc.get("page_kind") == "restaurant_menu":
        count = len(_dish_maps(doc))
        if count:
            return f"menu · {count} dish{'' if count == 1 else 'es'}"
        return "menu"
    is_vegan = _path_value(doc, "shopping_item.is_vegan")
    if is_vegan is True:
        return "vegan"
    if is_vegan is False:
        return "not vegan"
    return "—"


def _feedback_label(doc: Dict[str, Any]) -> str:
    """The user's verdict for the list table, with a mark when they wrote more.

    Most analyses have no feedback at all, so the column is empty far more
    often than not — the thumb is what should catch the eye when scanning.
    """
    rating = _path_value(doc, "feedback.rating")
    if rating not in ("up", "down"):
        return "—"
    thumb = "\N{THUMBS UP SIGN}" if rating == "up" else "\N{THUMBS DOWN SIGN}"
    return (
        f"{thumb} \N{SPEECH BALLOON}" if _path_value(doc, "feedback.comment") else thumb
    )


def _tokens(doc: Dict[str, Any]) -> Optional[int]:
    usage = doc.get("token_usage")
    return usage.get("total_tokens") if isinstance(usage, dict) else None


def _int_arg(name: str, default: int, maximum: int) -> int:
    """Read a positive integer query parameter, clamped and fault-tolerant."""
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(0, min(value, maximum))


def _matches(doc: Dict[str, Any], filters: Dict[str, str], query: str) -> bool:
    """Whether a document passes the filter selects and the free-text search."""
    for field, wanted in filters.items():
        if not wanted:
            continue
        if str(_field_value(doc, field) or "") != wanted:
            return False
    if not query:
        return True
    needle = query.lower()
    return any(needle in str(doc.get(field) or "").lower() for field in SEARCH_FIELDS)


def _distinct(docs: List[Dict[str, Any]], field: str) -> List[str]:
    """Sorted distinct non-empty values of a field, to populate a select."""
    return sorted(
        {
            str(_field_value(d, field))
            for d in docs
            if _field_value(d, field) not in (None, "")
        }
    )


def _summarize(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts shown above the table, computed over the matching documents."""
    by_kind: Dict[str, int] = {}
    by_service: Dict[str, int] = {}
    total_tokens = 0
    total_seconds = 0.0
    timed = 0
    failed = 0
    for doc in docs:
        if _status(doc) != STATUS_OK:
            failed += 1
        kind = str(doc.get("page_kind") or "unknown")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        service = str(doc.get("service") or "unknown")
        by_service[service] = by_service.get(service, 0) + 1
        total_tokens += _tokens(doc) or 0
        duration = _duration_s(doc)
        if duration is not None:
            total_seconds += duration
            timed += 1
    return {
        "by_kind": sorted(by_kind.items()),
        "by_service": sorted(by_service.items()),
        "total_tokens": total_tokens,
        # Averaged over the records that carry a duration, so records written
        # before it was measured don't drag the mean down.
        "mean_duration": _format_duration(total_seconds / timed) if timed else "",
        "failed": failed,
    }


STYLE = """
:root {
  --bg: #ffffff; --fg: #1b1b1f; --muted: #6b6b76; --line: #e2e2e8;
  --accent: #2f6f4f; --chip: #f2f2f5; --warn: #a1421f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16161a; --fg: #e8e8ec; --muted: #9a9aa6; --line: #2e2e36;
    --accent: #7fbf9a; --chip: #24242b; --warn: #e0866a;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px; background: var(--bg); color: var(--fg);
  font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
a { color: var(--accent); }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 28px 0 8px; text-transform: uppercase;
     letter-spacing: .06em; color: var(--muted); }
h3 { font-size: 13px; margin: 18px 0 6px; color: var(--muted); font-weight: 600; }
.note { color: var(--muted); margin: 0 0 16px; }
form.filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-end;
               margin-bottom: 14px; }
form.filters label { display: flex; flex-direction: column; gap: 3px;
                     font-size: 12px; color: var(--muted); }
input, select, button {
  font: inherit; padding: 5px 8px; background: var(--bg); color: var(--fg);
  border: 1px solid var(--line); border-radius: 6px;
}
button { cursor: pointer; background: var(--chip); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
.chip { background: var(--chip); border-radius: 999px; padding: 2px 10px;
        font-size: 12px; color: var(--muted); }
.chip.warn { color: var(--warn); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line);
         vertical-align: top; }
th { color: var(--muted); font-weight: 600; white-space: nowrap; }
tbody tr:hover { background: var(--chip); }
td.time, td.num { white-space: nowrap; }
td.page { max-width: 420px; }
td.page .url { color: var(--muted); font-size: 12px; word-break: break-all; }
.v-vegan { color: var(--accent); font-weight: 600; }
.v-maybe { color: var(--accent); }
.v-not { color: var(--warn); font-weight: 600; }
table.dishes td { vertical-align: top; }
table.dishes td.dish { font-weight: 600; }
table.dishes td.reason { max-width: 420px; }
.pager { display: flex; gap: 12px; align-items: center; margin-top: 14px; }
dl.fields { display: grid; grid-template-columns: 200px 1fr; gap: 6px 16px;
            margin: 0; }
dl.fields dt { color: var(--muted); }
dl.fields dd { margin: 0; word-break: break-word; white-space: pre-wrap; }
pre { background: var(--chip); padding: 12px; border-radius: 8px;
      overflow-x: auto; white-space: pre-wrap; word-break: break-word;
      max-height: 480px; overflow-y: auto; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.empty { color: var(--muted); padding: 24px 0; }
"""

LIST_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>API calls — {{ collection }}</title>
  <style>{{ style }}</style>
</head>
<body>
  <h1>API calls</h1>
  <p class="note">
    Collection <code>{{ collection }}</code> · scanned the {{ scanned }} most
    recent record{{ '' if scanned == 1 else 's' }} · {{ matching }} match{{ '' if matching == 1 else 'es' }}
    the filters · times are local · <a href="{{ json_url }}">JSON</a>
  </p>

  <form class="filters" method="get">
    <label>Search
      <input type="search" name="q" value="{{ query }}" placeholder="url, title, content…" size="28">
    </label>
    {% for field, label, options in selects %}
    <label>{{ label }}
      <select name="{{ field }}">
        <option value="">any</option>
        {% for option in options %}
        <option value="{{ option }}" {{ 'selected' if filters[field] == option }}>{{ option }}</option>
        {% endfor %}
      </select>
    </label>
    {% endfor %}
    <label>Scan
      <input type="number" name="scan" value="{{ scan_limit }}" min="1" max="{{ max_scan }}" size="6">
    </label>
    <label>Per page
      <input type="number" name="page_size" value="{{ page_size }}" min="1" max="{{ max_page_size }}" size="5">
    </label>
    <button type="submit">Apply</button>
    <a class="chip" href="{{ base_url }}">Reset</a>
  </form>

  {% if summary.by_kind or summary.by_service %}
  <div class="chips">
    {% for kind, count in summary.by_kind %}<span class="chip">{{ kind }}: {{ count }}</span>{% endfor %}
    {% for service, count in summary.by_service %}<span class="chip">{{ service }}: {{ count }}</span>{% endfor %}
    <span class="chip">tokens: {{ '{:,}'.format(summary.total_tokens) }}</span>
    {% if summary.mean_duration %}<span class="chip">mean: {{ summary.mean_duration }}</span>{% endif %}
    {% if summary.failed %}<span class="chip warn">failed: {{ summary.failed }}</span>{% endif %}
  </div>
  {% endif %}

  {% if rows %}
  <div class="scroll">
  <table>
    <thead>
      <tr>{% for _, header in columns %}<th>{{ header }}</th>{% endfor %}</tr>
    </thead>
    <tbody>
      {% for row in rows %}
      <tr>
        <td class="time"><a href="{{ row.detail_url }}">{{ row.created_at }}</a></td>
        <td>{{ row.page_kind }}</td>
        <td>{{ row.page_scope }}</td>
        <td class="{{ row.verdict_class }}">{{ row.verdict }}</td>
        <td>{{ row.confidence }}</td>
        <td class="page">
          <div>{{ row.title or '(no title)' }}</div>
          <div class="url">{{ row.url }}</div>
          {% if row.summary %}<div class="url">{{ row.summary }}</div>{% endif %}
        </td>
        <td>{{ row.service }}</td>
        <td>{{ row.model }}</td>
        <td class="num">{{ row.tokens }}</td>
        <td class="num">{{ row.duration }}</td>
        <td>{{ row.extension_version }}</td>
        <td>{{ row.installation_id }}</td>
        <td class="{{ row.feedback_class }}">{{ row.feedback }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  </div>

  <div class="pager">
    {% if prev_url %}<a href="{{ prev_url }}">← Newer</a>{% endif %}
    <span class="chip">{{ offset + 1 }}–{{ offset + rows|length }} of {{ matching }}</span>
    {% if next_url %}<a href="{{ next_url }}">Older →</a>{% endif %}
  </div>
  {% else %}
  <p class="empty">No records match.</p>
  {% endif %}
</body>
</html>
"""

DETAIL_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ doc.get('title') or doc.get('item_url') or record_id }}</title>
  <style>{{ style }}</style>
</head>
<body>
  <p class="note"><a href="{{ list_url }}">← All calls</a> ·
    <code>{{ collection }}/{{ record_id }}</code> ·
    <a href="{{ json_url }}">JSON</a></p>
  <h1>{{ doc.get('title') or '(no title)' }}</h1>
  <p class="note"><a href="{{ doc.get('item_url') }}" rel="noreferrer">{{ doc.get('item_url') }}</a></p>

  {% for group, fields in groups %}
  <h2>{{ group }}</h2>
  <dl class="fields">
    {% for label, value in fields %}
    <dt>{{ label }}</dt><dd>{{ value }}</dd>
    {% endfor %}
  </dl>
  {% endfor %}

  {% if dishes %}
  <h2>Menu dishes</h2>
  <div class="chips">
    {% for label, count in tally %}<span class="chip">{{ label }}: {{ count }}</span>{% endfor %}
  </div>
  <div class="scroll">
  <table class="dishes">
    <thead>
      <tr>{% for _, header in dish_columns %}<th>{{ header }}</th>{% endfor %}</tr>
    </thead>
    <tbody>
      {% for dish in dishes %}
      <tr>
        <td class="dish">{{ dish.name }}</td>
        <td>{{ dish.section }}</td>
        <td class="{{ dish.verdict_class }}">{{ dish.verdict }}</td>
        <td class="reason">{{ dish.reason }}</td>
        <td>{{ dish.explicit_ingredients }}</td>
        <td>{{ dish.inferred_ingredients }}</td>
        {% if show_avoided %}<td>{{ dish.user_avoided_ingredients }}</td>{% endif %}
      </tr>
      {% endfor %}
    </tbody>
  </table>
  </div>
  {% endif %}

  {% if job_ref %}
  <h2>Desktop-server job</h2>
  <p class="note"><code>{{ job_ref }}</code></p>
  {% if job %}
  <dl class="fields">
    {% for label, value in job_fields %}
    <dt>{{ label }}</dt><dd>{{ value }}</dd>
    {% endfor %}
  </dl>
  {% for role, text in job_messages %}
  <h3>Prompt · {{ role }}</h3>
  <pre>{{ text }}</pre>
  {% endfor %}
  {% for label, text in job_completion %}
  <h3>Completion · {{ label }}</h3>
  <pre>{{ text }}</pre>
  {% endfor %}
  <h3>Raw job document</h3>
  <pre>{{ job_raw }}</pre>
  {% else %}
  <p class="empty">{{ job_note }}</p>
  {% endif %}
  {% endif %}

  <h2>Stored page content{% if content_truncated %} (truncated on write){% endif %}</h2>
  {% if content %}<pre>{{ content }}</pre>{% else %}<p class="empty">Empty.</p>{% endif %}

  <h2>Raw document</h2>
  <pre>{{ raw }}</pre>
</body>
</html>
"""


def _list_rows(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Shape documents into the values the list template prints."""
    rows = []
    for doc in docs:
        verdict = _verdict(doc)
        rows.append(
            {
                "detail_url": f"{URL_PREFIX}/{doc['id']}",
                "created_at": _format_time(doc.get("created_at")),
                "page_kind": doc.get("page_kind") or "—",
                "page_scope": _scope_label(doc) or "—",
                "verdict": verdict,
                "verdict_class": {
                    "vegan": "v-vegan",
                    "not vegan": "v-not",
                    "failed": "v-not",
                }.get(verdict, ""),
                "confidence": _path_value(doc, "shopping_item.confidence_level")
                or "—",
                "title": doc.get("title") or "",
                "url": doc.get("item_url") or "",
                # A failed record has no summary; its error is the
                # one line worth showing in its place.
                "summary": doc.get("summary") or doc.get("error") or "",
                "service": doc.get("service") or "—",
                "model": doc.get("model") or "—",
                "tokens": _tokens(doc) if _tokens(doc) is not None else "—",
                "duration": _format_duration(_duration_s(doc)) or "—",
                "extension_version": doc.get("extension_version") or "—",
                "installation_id": doc.get("installation_id") or "—",
                "feedback": _feedback_label(doc),
                "feedback_class": {
                    "up": "v-vegan",
                    "down": "v-not",
                }.get(str(_path_value(doc, "feedback.rating") or ""), ""),
            }
        )
    return rows


def _detail_groups(doc: Dict[str, Any]) -> List[Any]:
    """Render the grouped field lists of the detail page."""
    groups = []
    for group, fields in DETAIL_GROUPS:
        rendered = []
        for field in fields:
            value = _path_value(doc, field)
            if field.endswith("_at"):
                shown = _format_time(value)
            elif field == "duration":
                shown = _format_duration(_duration_s(doc))
            elif isinstance(value, dict):
                shown = ", ".join(
                    f"{k}={v}" for k, v in sorted(value.items()) if v is not None
                )
            elif isinstance(value, (list, tuple)):
                shown = ", ".join(str(v) for v in value)
            elif value is None or value == "":
                shown = "—"
            else:
                shown = str(value)
            rendered.append((_label(field), shown or "—"))
        groups.append((group, rendered))
    return groups


def _join(values: Any) -> str:
    """Render a stored string list as one cell."""
    if isinstance(values, (list, tuple)) and values:
        return ", ".join(str(value) for value in values)
    return "—"


def _dishes(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Shape the stored per-dish verdicts into rows of the dish table."""
    rows = []
    for item in _dish_maps(doc):
        verdict = str(item.get("verdict") or "unclear")
        rows.append(
            {
                "name": item.get("name") or "—",
                "section": item.get("section") or "—",
                # The stored verdicts are the prompt's enum values; the
                # underscore is for the model, not for a reader.
                "verdict": verdict.replace("_", " "),
                "verdict_class": VERDICT_CLASSES.get(verdict, ""),
                "reason": item.get("reason") or "—",
                "explicit_ingredients": _join(item.get("explicit_ingredients")),
                "inferred_ingredients": _join(item.get("inferred_ingredients")),
                "user_avoided_ingredients": _join(item.get("user_avoided_ingredients")),
            }
        )
    return rows


def _shows_avoided(dishes: List[Dict[str, Any]]) -> bool:
    """Whether any dish names an avoid-list ingredient, so the column earns a place."""
    return any(dish["user_avoided_ingredients"] != "—" for dish in dishes)


def _verdict_tally(doc: Dict[str, Any]) -> List[Any]:
    """Count the dishes by verdict, in the order the prompt defines them."""
    counts: Dict[str, int] = {}
    for item in _dish_maps(doc):
        verdict = str(item.get("verdict") or "unclear")
        counts[verdict] = counts.get(verdict, 0) + 1
    known = [(v, counts.pop(v)) for v in ITEM_VERDICTS if v in counts]
    # Anything the prompt does not define still gets a chip rather than vanishing.
    ordered = known + sorted(counts.items())
    return [(verdict.replace("_", " "), count) for verdict, count in ordered]


def _first_choice(job: Dict[str, Any]) -> Dict[str, Any]:
    """The first choice of a job's OpenAI-format response, or an empty map.

    Job documents are read raw and were written by another process, so every
    step down into them tolerates a missing or differently-shaped level.
    """
    response = job.get("response")
    choices = response.get("choices") if isinstance(response, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _job_fields(job: Dict[str, Any]) -> List[Any]:
    """Render the job's own fields, plus the settings the call ran under."""
    rendered = []
    for field in JOB_FIELDS:
        value = job.get(field)
        shown = _format_time(value) if field.endswith("_at") else value
        rendered.append((_label(field), "—" if shown in (None, "") else str(shown)))
    request_body = job.get("request")
    if isinstance(request_body, dict):
        for field in ("temperature", "max_tokens"):
            if request_body.get(field) is not None:
                rendered.append((_label(field), str(request_body[field])))
    finish_reason = _first_choice(job).get("finish_reason")
    if finish_reason:
        rendered.append(("Finish reason", str(finish_reason)))
    return rendered


def _job_messages(job: Dict[str, Any]) -> List[Any]:
    """The prompt as it was actually sent: one (role, text) pair per message.

    This is the whole prompt — instructions included — where the record's own
    "stored page content" is only the page part of it.
    """
    request_body = job.get("request")
    messages = request_body.get("messages") if isinstance(request_body, dict) else None
    if not isinstance(messages, list):
        return []
    return [
        (str(message.get("role") or "message"), str(message.get("content") or ""))
        for message in messages
        if isinstance(message, dict)
    ]


def _job_completion(job: Dict[str, Any]) -> List[Any]:
    """What the model answered, reasoning first where the response carries it."""
    message = _first_choice(job).get("message")
    if not isinstance(message, dict):
        return []
    return [
        (_label(field), str(message[field]))
        for field in ("reasoning", "reasoning_content", "content")
        if message.get(field)
    ]


def _load_job(
    firestore_service, job_id: str
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Read the desktop-server job a record links to.

    Returns (job, note): the note explains an absent job, which is a normal
    thing to see — the jobs collection is the desktop-server's, with its own
    lifetime, so a record can outlive the job it points at. A failed read is
    reported the same way rather than costing the whole record its page.
    """
    collection = jobs_collection_name()
    try:
        job = firestore_service.get_raw_document(job_id, collection_name=collection)
    except Exception as e:
        logger.warning("Failed to read job %s/%s: %s", collection, job_id, e)
        return None, f"Could not read {collection}/{job_id}: {e}"
    if job is None:
        return None, f"No such document — {collection}/{job_id} is gone."
    return job, ""


def make_blueprint(firestore_service) -> Blueprint:
    """Build the viewer blueprint over an already-constructed FirestoreService."""
    bp = Blueprint("db_viewer", __name__, url_prefix=URL_PREFIX)

    @bp.before_request
    def _local_only():
        if not _request_is_local():
            logger.warning(
                "Refusing non-local database-viewer request from %s",
                request.remote_addr,
            )
            abort(404)

    @bp.route("/")
    def list_calls():
        scan_limit = _int_arg("scan", DEFAULT_SCAN_LIMIT, MAX_SCAN_LIMIT) or 1
        page_size = _int_arg("page_size", DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE) or 1
        offset = _int_arg("offset", 0, 10**6)
        query = (request.args.get("q") or "").strip()
        filters = {
            field: (request.args.get(field) or "").strip()
            for field, _ in FILTER_SELECTS
        }

        scanned = firestore_service.get_raw_documents(limit=scan_limit)
        matching = [doc for doc in scanned if _matches(doc, filters, query)]
        page = matching[offset : offset + page_size]

        if request.args.get("format") == "json":
            return jsonify(
                {
                    "records": [_jsonable(doc) for doc in page],
                    "scanned": len(scanned),
                    "matching": len(matching),
                    "offset": offset,
                    "page_size": page_size,
                }
            )

        def page_url(new_offset: int) -> str:
            args = {k: v for k, v in request.args.items() if v}
            args["offset"] = new_offset
            return f"{URL_PREFIX}/?" + "&".join(f"{k}={v}" for k, v in args.items())

        selects = [
            (field, label, _distinct(scanned, field)) for field, label in FILTER_SELECTS
        ]

        return render_template_string(
            LIST_TEMPLATE,
            style=STYLE,
            collection=firestore_service.collection_name,
            columns=LIST_COLUMNS,
            rows=_list_rows(page),
            summary=_summarize(matching),
            scanned=len(scanned),
            matching=len(matching),
            offset=offset,
            page_size=page_size,
            scan_limit=scan_limit,
            max_scan=MAX_SCAN_LIMIT,
            max_page_size=MAX_PAGE_SIZE,
            query=query,
            filters=filters,
            selects=selects,
            base_url=f"{URL_PREFIX}/",
            json_url=page_url(offset) + "&format=json",
            prev_url=page_url(max(0, offset - page_size)) if offset else None,
            next_url=page_url(offset + page_size)
            if offset + page_size < len(matching)
            else None,
        )

    @bp.route("/<record_id>")
    def show_call(record_id: str):
        doc = firestore_service.get_raw_document(record_id)
        if doc is None:
            abort(404)

        job_id = doc.get("lms_job_id")
        job, job_note = _load_job(firestore_service, job_id) if job_id else (None, "")

        if request.args.get("format") == "json":
            payload = _jsonable(doc)
            if job is not None:
                payload["lms_job"] = _jsonable(job)
            return jsonify(payload)

        content = doc.get("content") or ""
        dishes = _dishes(doc)
        show_avoided = _shows_avoided(dishes)

        return render_template_string(
            DETAIL_TEMPLATE,
            style=STYLE,
            collection=firestore_service.collection_name,
            record_id=record_id,
            doc=doc,
            groups=_detail_groups(doc),
            dishes=dishes,
            dish_columns=DISH_COLUMNS + ((AVOIDED_COLUMN,) if show_avoided else ()),
            show_avoided=show_avoided,
            tally=_verdict_tally(doc),
            job=job,
            job_ref=f"{jobs_collection_name()}/{job_id}" if job_id else "",
            job_note=job_note,
            job_fields=_job_fields(job) if job else [],
            job_messages=_job_messages(job) if job else [],
            job_completion=_job_completion(job) if job else [],
            job_raw=json.dumps(_jsonable(job), indent=2, ensure_ascii=False)
            if job
            else "",
            content=content,
            content_truncated=len(content) >= STORED_CONTENT_CHAR_LIMIT,
            raw=json.dumps(_jsonable(doc), indent=2, ensure_ascii=False),
            list_url=f"{URL_PREFIX}/",
            json_url=f"{URL_PREFIX}/{record_id}?format=json",
        )

    return bp


def register_db_viewer(app: Flask, firestore_service) -> bool:
    """Register the viewer on ``app`` unless it is switched off or unusable.

    Returns whether it was registered, so the caller can log it.
    """
    if not viewer_enabled():
        logger.info("Database viewer disabled (ENABLE_DB_VIEWER)")
        return False
    if firestore_service is None:
        logger.info("Database viewer not registered: no database configured")
        return False
    app.register_blueprint(make_blueprint(firestore_service))
    logger.info(f"Database viewer available at {URL_PREFIX}/ (local requests only)")
    return True
