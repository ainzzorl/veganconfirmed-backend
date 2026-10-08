"""
Report generation service for Vegan Confirmed analytics.

This module contains shared logic for generating usage reports.
"""

import os
import html
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from collections import defaultdict

from urllib.parse import urlparse

from models.database_model import STATUS_OK, TokenUsage
from services.firestore_service import FirestoreService
from services.cloud_logging_service import CloudLoggingService
from services.extension_store_service import ExtensionStoreService, STORE_LABELS

logger = logging.getLogger(__name__)


# How the report names things the records name differently. "desktop" is the
# provider running on the local machine, which is what a reader cares about.
PROVIDER_LABELS = {"desktop": "Local", "gemini": "Gemini"}
# A request that never got an answer names no provider, and neither do records
# written before the provider was stored. They are still calls the week made,
# so they get a row rather than being dropped.
NO_PROVIDER = "none"
PROVIDER_LABELS[NO_PROVIDER] = "Not recorded"

STATUS_LABELS = {
    "ok": "Answered",
    "invalid_request": "Invalid request",
    "analysis_failed": "Analysis failed",
}

PAGE_KIND_LABELS = {
    "shopping_item": "Item",
    "restaurant_menu": "Menu",
    "other": "Other",
    "unknown": "Unclassified",
}


def _label(labels: Dict[str, str], key: str) -> str:
    """Display name for a stored value, falling back to the value itself."""
    return labels.get(key, key.replace("_", " ").title())


def _pct(count: float, total: float) -> float:
    """Percentage of ``total``, or 0 when there is nothing to divide by."""
    return (count / total * 100) if total else 0.0


def _empty_call_stats() -> Dict[str, int]:
    """Counters for one group of calls: how many, how they went, what they spent."""
    return {
        "calls": 0,
        "ok": 0,
        # Calls whose provider reported any token usage. Averages divide by
        # this rather than by ``calls``, so records written before a provider
        # reported usage do not drag them down.
        "calls_with_usage": 0,
        "read_tokens": 0,
        "write_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }


def _add_token_usage(stats: Dict[str, int], usage: Optional[TokenUsage]) -> None:
    """Add one record's reported tokens to a group's counters.

    Read (prompt) and write (completion) are kept apart because they cost
    differently and move for different reasons. ``total`` is what the provider
    reported — which includes reasoning tokens it did not break out — and falls
    back to read + write when it reported no total at all.
    """
    if usage is None:
        return

    reported = (
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.reasoning_tokens,
        usage.total_tokens,
    )
    if all(value is None for value in reported):
        return

    stats["calls_with_usage"] += 1
    stats["read_tokens"] += usage.prompt_tokens or 0
    stats["write_tokens"] += usage.completion_tokens or 0
    stats["reasoning_tokens"] += usage.reasoning_tokens or 0
    total = usage.total_tokens
    if total is None:
        total = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
    stats["total_tokens"] += total


def _per_call(total: float, calls: int) -> float:
    """Average per call, or 0 when nothing was counted."""
    return (total / calls) if calls else 0.0


FEEDBACK_LABELS = {"up": "\N{THUMBS UP SIGN} Up", "down": "\N{THUMBS DOWN SIGN} Down"}


def _feedback_entry(record) -> Dict[str, Any]:
    """One rating as the report lists it, with enough of the analysis to place it."""
    feedback = record.feedback
    given_at = feedback.updated_at or feedback.created_at or record.created_at
    verdict = None
    if record.shopping_item:
        verdict = {True: "Vegan", False: "Non-vegan"}.get(
            record.shopping_item.is_vegan, "No verdict"
        )
    elif record.menu:
        verdict = record.menu.vegan_friendliness
    return {
        "analysis_id": record.id,
        "rating": feedback.rating,
        "comment": feedback.comment,
        "given_at": given_at.isoformat() if given_at else None,
        "title": record.title,
        "item_url": record.item_url,
        "page_kind": record.page_kind or "unknown",
        "verdict": verdict,
        "service": record.service or NO_PROVIDER,
        "model": record.model,
    }


def _feedback_rows(entries: List[Dict[str, Any]]) -> str:
    """A row per rating. Comments and titles come from users and pages, so
    everything is escaped."""
    rows = ""
    for entry in entries:
        given_at = (entry["given_at"] or "")[:16].replace("T", " ")
        title = html.escape(entry["title"] or entry["item_url"] or "(untitled)")
        if entry["item_url"]:
            title = f'<a href="{html.escape(entry["item_url"])}">{title}</a>'
        provider = html.escape(_label(PROVIDER_LABELS, entry["service"]))
        if entry["model"]:
            provider += f" ({html.escape(entry['model'])})"
        css = "vegan" if entry["rating"] == "up" else "non-vegan"
        rows += f"""
                <tr>
                    <td>{given_at}</td>
                    <td class="{css}">{html.escape(_label(FEEDBACK_LABELS, entry["rating"]))}</td>
                    <td>{html.escape(entry["comment"] or "")}</td>
                    <td>{title}</td>
                    <td>{_label(PAGE_KIND_LABELS, entry["page_kind"])}</td>
                    <td>{html.escape(entry["verdict"] or "")}</td>
                    <td>{provider}</td>
                    <td><code>{html.escape(entry["analysis_id"] or "")}</code></td>
                </tr>
        """
    return rows


def _count_rows(
    counts: Dict[str, int], total: int, labels: Optional[Dict[str, str]] = None
) -> str:
    """Table rows of name / count / share, largest first."""
    rows = ""
    for key, count in sorted(counts.items(), key=lambda pair: pair[1], reverse=True):
        name = _label(labels or {}, key)
        rows += f"""
                <tr>
                    <td>{name}</td>
                    <td>{count}</td>
                    <td>{_pct(count, total):.1f}%</td>
                </tr>
        """
    return rows


def _signed(value: float, fmt: str = ",.0f") -> str:
    return f"{'+' if value > 0 else ''}{value:{fmt}}"


def _extension_user_row(name: str, users, previous, css: str = "") -> str:
    """One store's count now, at the previous snapshot, and the change between."""
    row_class = f' class="{css}"' if css else ""
    change = ""
    if users is not None and previous is not None:
        change = _signed(users - previous)
        if previous:
            change += f" ({_signed((users - previous) / previous * 100, '.1f')}%)"
    return f"""
                <tr{row_class}>
                    <td>{name}</td>
                    <td>{"unavailable" if users is None else f"{users:,}"}</td>
                    <td>{"" if previous is None else f"{previous:,}"}</td>
                    <td>{change}</td>
                </tr>
        """


def _extension_user_rows(stores: Dict[str, Dict[str, Any]]) -> str:
    """A row per store, then both together when every store could be read."""
    rows = "".join(
        _extension_user_row(_label(STORE_LABELS, store), s["users"], s["previous"])
        for store, s in stores.items()
    )
    if all(s["users"] is not None for s in stores.values()):
        previous = [s["previous"] for s in stores.values()]
        rows += _extension_user_row(
            "Both stores",
            sum(s["users"] for s in stores.values()),
            None if None in previous else sum(previous),
            css="total",
        )
    return rows


def _extension_users_section(extension_users: Optional[Dict[str, Any]]) -> str:
    if not extension_users or extension_users.get("error"):
        return ""
    previous_at = (extension_users.get("previous_at") or "")[:10]
    since = (
        f"The previous count is from the snapshot taken {previous_at}."
        if previous_at
        else "There is no snapshot from a week ago yet to compare with."
    )
    return f"""
        <div class="section">
            <h2>🧩 Extension Users</h2>
            <table>
                <tr>
                    <th>Store</th>
                    <th>Users</th>
                    <th>Week Before</th>
                    <th>Change</th>
                </tr>
                {_extension_user_rows(extension_users["stores"])}
            </table>
            <p><em>As the stores report them at the time of this report. Firefox
            counts average daily users; Chrome counts roughly weekly active users
            and rounds large numbers. {since}</em></p>
        </div>
    """


def _provider_row(
    name: str, stats: Dict[str, int], total_calls: int, css: str = ""
) -> str:
    """One row of the provider table: volume, success and what it spent."""
    row_class = f' class="{css}"' if css else ""
    return f"""
                <tr{row_class}>
                    <td>{name}</td>
                    <td>{stats['calls']}</td>
                    <td>{_pct(stats['calls'], total_calls):.1f}%</td>
                    <td>{_pct(stats['ok'], stats['calls']):.1f}%</td>
                    <td>{stats['read_tokens']:,}</td>
                    <td>{stats['write_tokens']:,}</td>
                    <td>{stats['reasoning_tokens']:,}</td>
                    <td>{stats['total_tokens']:,}</td>
                    <td>{_per_call(stats['total_tokens'], stats['calls_with_usage']):,.0f}</td>
                </tr>
        """


def _provider_rows(
    provider_stats: Dict[str, Dict[str, int]],
    call_stats: Dict[str, int],
    total_calls: int,
) -> str:
    """The provider table: a row each, then the week as a whole.

    Token usage lives here rather than in a table of its own — the per-provider
    rows already carry the read/write split, and a separate one would only be
    their sum.
    """
    ordered = sorted(
        provider_stats.items(), key=lambda pair: pair[1]["calls"], reverse=True
    )
    rows = "".join(
        _provider_row(_label(PROVIDER_LABELS, service), stats, total_calls)
        for service, stats in ordered
    )
    return rows + _provider_row("All providers", call_stats, total_calls, css="total")


def save_report_locally(report_data: Dict[str, Any], output_path: str = None) -> str:
    """
    Generate the weekly report and save it locally as an HTML file.

    Args:
        report_data: Dictionary containing report statistics
        output_path: Optional path for the output file. If not provided,
                    saves to 'local/weekly_report_YYYY-MM-DD.html'

    Returns:
        str: Path to the saved file, or None if failed
    """
    try:
        # Generate default filename if not provided
        if output_path is None:
            filename = f"weekly_report_{report_data['week_start']}_to_{report_data['week_end']}.html"
            # Sanitize filename (replace slashes and spaces)
            filename = filename.replace("/", "-").replace(" ", "_")
            # Save to local/ directory
            output_path = os.path.join("local", filename)

        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Generate HTML content
        html_content = create_html_report(report_data)

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Weekly report saved locally to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to save weekly report locally: {e}")
        return None


def create_html_report(report_data: Dict[str, Any]) -> str:
    """
    Create HTML formatted report.

    Args:
        report_data: Dictionary containing report statistics

    Returns:
        str: HTML content of the report
    """
    total_calls = report_data["total_calls"]
    # Records written before a field existed simply do not have it, so every
    # new section reads through a default rather than assuming the shape.
    call_stats = report_data.get("call_stats") or _empty_call_stats()
    item_stats = report_data.get("item_stats") or _empty_call_stats()
    menu_stats = report_data.get("menu_stats") or _empty_call_stats()
    provider_stats = report_data.get("provider_stats") or {}
    success_rate = report_data.get("success_rate", 0.0)
    vegan_breakdown = report_data["vegan_breakdown"]
    vegan_unknown = vegan_breakdown.get("unknown", 0)
    menu_dishes = menu_stats.get("dishes", 0)

    status_rows = _count_rows(
        report_data.get("status_breakdown") or {}, total_calls, STATUS_LABELS
    )
    kind_rows = _count_rows(
        report_data.get("page_kind_breakdown") or {}, total_calls, PAGE_KIND_LABELS
    )
    friendliness_rows = _count_rows(
        menu_stats.get("friendliness") or {}, menu_stats["calls"]
    )
    provider_rows = _provider_rows(provider_stats, call_stats, total_calls)
    feedback = report_data.get("feedback") or []
    feedback_up = sum(1 for entry in feedback if entry["rating"] == "up")
    feedback_commented = sum(1 for entry in feedback if entry["comment"])
    feedback_rows = _feedback_rows(feedback)
    extension_users_section = _extension_users_section(
        report_data.get("extension_users")
    )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
            .metric-label {{ font-size: 12px; color: #666; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .vegan {{ color: #4CAF50; }}
            .non-vegan {{ color: #f44336; }}
            tr.total td {{ font-weight: bold; border-top: 2px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌱 Vegan Confirmed Weekly Usage Report</h1>
            <p>{report_data['week_start']} to {report_data['week_end']}</p>
        </div>
        
        <div class="section">
            <h2>📊 Overview</h2>
            <div class="metric">
                <div class="metric-value">{total_calls}</div>
                <div class="metric-label">Total Calls</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report_data.get('unique_clients', 0)}</div>
                <div class="metric-label">Unique Clients</div>
            </div>
            <div class="metric">
                <div class="metric-value">{success_rate:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{item_stats['calls']}</div>
                <div class="metric-label">Item Analyses</div>
            </div>
            <div class="metric">
                <div class="metric-value">{menu_stats['calls']}</div>
                <div class="metric-label">Menu Analyses</div>
            </div>
            <div class="metric">
                <div class="metric-value">{call_stats['total_tokens']:,}</div>
                <div class="metric-label">Total Tokens</div>
            </div>
        </div>

        {extension_users_section}

        <div class="section">
            <h2>✅ Request Outcomes</h2>
            <div class="metric">
                <div class="metric-value">{report_data.get('successful_calls', 0)}</div>
                <div class="metric-label">Answered</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report_data.get('failed_calls', 0)}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{success_rate:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <table>
                <tr>
                    <th>Outcome</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                {status_rows}
            </table>
        </div>

        <div class="section">
            <h2>🧭 Analysis Kinds</h2>
            <table>
                <tr>
                    <th>Page Kind</th>
                    <th>Calls</th>
                    <th>Percentage</th>
                </tr>
                {kind_rows}
            </table>
        </div>

        <div class="section">
            <h2>🛒 Item Analysis</h2>
            <div class="metric">
                <div class="metric-value">{item_stats['calls']}</div>
                <div class="metric-label">Analyses</div>
            </div>
            <div class="metric">
                <div class="metric-value">{_pct(item_stats['calls'], total_calls):.1f}%</div>
                <div class="metric-label">Of All Calls</div>
            </div>
            <div class="metric">
                <div class="metric-value">{_per_call(item_stats['total_tokens'], item_stats['calls_with_usage']):,.0f}</div>
                <div class="metric-label">Tokens per Analysis</div>
            </div>
            <table>
                <tr>
                    <th>Verdict</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                <tr class="vegan">
                    <td>Vegan</td>
                    <td>{vegan_breakdown['vegan']}</td>
                    <td>{_pct(vegan_breakdown['vegan'], item_stats['calls']):.1f}%</td>
                </tr>
                <tr class="non-vegan">
                    <td>Non-Vegan</td>
                    <td>{vegan_breakdown['non_vegan']}</td>
                    <td>{_pct(vegan_breakdown['non_vegan'], item_stats['calls']):.1f}%</td>
                </tr>
                <tr>
                    <td>No Verdict</td>
                    <td>{vegan_unknown}</td>
                    <td>{_pct(vegan_unknown, item_stats['calls']):.1f}%</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>🍽️ Menu Analysis</h2>
            <div class="metric">
                <div class="metric-value">{menu_stats['calls']}</div>
                <div class="metric-label">Analyses</div>
            </div>
            <div class="metric">
                <div class="metric-value">{_pct(menu_stats['calls'], total_calls):.1f}%</div>
                <div class="metric-label">Of All Calls</div>
            </div>
            <div class="metric">
                <div class="metric-value">{menu_dishes}</div>
                <div class="metric-label">Dishes Rated</div>
            </div>
            <div class="metric">
                <div class="metric-value">{_per_call(menu_dishes, menu_stats['calls']):.1f}</div>
                <div class="metric-label">Dishes per Menu</div>
            </div>
            <div class="metric">
                <div class="metric-value">{_per_call(menu_stats['total_tokens'], menu_stats['calls_with_usage']):,.0f}</div>
                <div class="metric-label">Tokens per Analysis</div>
            </div>
            <table>
                <tr>
                    <th>Vegan Friendliness</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                {friendliness_rows}
            </table>
        </div>

        <div class="section">
            <h2>💬 User Feedback</h2>
            <div class="metric">
                <div class="metric-value">{len(feedback)}</div>
                <div class="metric-label">Ratings</div>
            </div>
            <div class="metric">
                <div class="metric-value">{feedback_up}</div>
                <div class="metric-label">Thumbs Up</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(feedback) - feedback_up}</div>
                <div class="metric-label">Thumbs Down</div>
            </div>
            <div class="metric">
                <div class="metric-value">{feedback_commented}</div>
                <div class="metric-label">With Comment</div>
            </div>
            <div class="metric">
                <div class="metric-value">{_pct(len(feedback), report_data.get('successful_calls', 0)):.1f}%</div>
                <div class="metric-label">Of Answered Calls Rated</div>
            </div>
            <table>
                <tr>
                    <th>Given (UTC)</th>
                    <th>Rating</th>
                    <th>Comment</th>
                    <th>Page</th>
                    <th>Kind</th>
                    <th>Verdict</th>
                    <th>Provider</th>
                    <th>Analysis ID</th>
                </tr>
                {feedback_rows}
            </table>
            <p><em>Every rating left on this week's analyses, newest first,
            whenever it was given.</em></p>
        </div>

        <div class="section">
            <h2>🤖 Providers and Token Usage</h2>
            <table>
                <tr>
                    <th>Provider</th>
                    <th>Calls</th>
                    <th>Percentage</th>
                    <th>Success Rate</th>
                    <th>Read Tokens</th>
                    <th>Write Tokens</th>
                    <th>Reasoning Tokens</th>
                    <th>Total Tokens</th>
                    <th>Total per Call</th>
                </tr>
                {provider_rows}
            </table>
            <p><em>Read is the prompt, write the completion; reasoning is part
            of the total rather than on top of it. Tokens were reported on
            {call_stats['calls_with_usage']} of {total_calls} calls and the
            per-call figures divide by those. "Not recorded" is a request that
            named no provider: it failed after both had been tried, or predates
            the field.</em></p>
        </div>

        <div class="section">
            <h2>🎯 Trigger Type Breakdown</h2>
            <table>
                <tr>
                    <th>Trigger Type</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
    """

    for trigger_type, count in report_data["trigger_breakdown"].items():
        percentage = (count / total_calls * 100) if total_calls > 0 else 0
        html += f"""
                <tr>
                    <td>{trigger_type.title()}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
        """

    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>🌐 Top Clients</h2>
            <table>
                <tr>
                    <th>Client</th>
                    <th>Calls</th>
                    <th>Percentage</th>
                    <th>Country</th>
                </tr>
    """

    client_countries = report_data.get("client_countries", {})
    top_clients = sorted(
        report_data.get("client_breakdown", {}).items(),
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    for client, count in top_clients:
        html += f"""
                <tr>
                    <td>{client}</td>
                    <td>{count}</td>
                    <td>{_pct(count, total_calls):.1f}%</td>
                    <td>{client_countries.get(client) or "Unknown"}</td>
                </tr>
        """

    html += """
            </table>
        </div>

        <div class="section">
            <h2>🌍 Geographic Distribution</h2>
            <table>
                <tr>
                    <th>Country</th>
                    <th>Calls</th>
                    <th>Percentage</th>
                </tr>
    """

    sorted_countries = sorted(
        report_data.get("country_breakdown", {}).items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for country, count in sorted_countries:
        html += f"""
                <tr>
                    <td>{country}</td>
                    <td>{count}</td>
                    <td>{_pct(count, total_calls):.1f}%</td>
                </tr>
        """

    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>🔗 Top Item Domains</h2>
            <table>
                <tr>
                    <th>Domain</th>
                    <th>Calls</th>
                    <th>Percentage</th>
                </tr>
    """

    # Show top 10 item domains
    item_domain_breakdown = report_data.get("item_domain_breakdown", {})
    sorted_domains = sorted(
        item_domain_breakdown.items(), key=lambda x: x[1], reverse=True
    )[:10]
    for domain, count in sorted_domains:
        percentage = (count / total_calls * 100) if total_calls > 0 else 0
        html += f"""
                <tr>
                    <td>{domain}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
        """

    html += """
            </table>
        </div>
    """

    # Add page visit statistics section if available
    page_visit_stats = report_data.get("page_visit_stats")
    if page_visit_stats and not page_visit_stats.get("error"):
        total_page_views = page_visit_stats.get("total_page_views", 0)
        unique_visitors = page_visit_stats.get("unique_visitors", 0)
        page_breakdown = page_visit_stats.get("page_breakdown", {})
        domain_breakdown = page_visit_stats.get("domain_breakdown", {})

        html += f"""
        <div class="section">
            <h2>📄 Website Page Visits (Firebase Hosting)</h2>
            <div class="metric">
                <div class="metric-value">{total_page_views}</div>
                <div class="metric-label">Total Page Views</div>
            </div>
            <div class="metric">
                <div class="metric-value">{unique_visitors}</div>
                <div class="metric-label">Unique Visitors</div>
            </div>
        """

        if domain_breakdown:
            html += """
            <h3>Domains</h3>
            <table>
                <tr>
                    <th>Domain</th>
                    <th>Views</th>
                    <th>Percentage</th>
                </tr>
            """

            # Show all domains
            sorted_domains = sorted(
                domain_breakdown.items(), key=lambda x: x[1], reverse=True
            )
            for domain, count in sorted_domains:
                percentage = (
                    (count / total_page_views * 100) if total_page_views > 0 else 0
                )
                html += f"""
                <tr>
                    <td>{domain}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
                """

            html += """
            </table>
            """

        if page_breakdown:
            html += """
            <h3>Top Pages</h3>
            <table>
                <tr>
                    <th>Page Path</th>
                    <th>Views</th>
                    <th>Percentage</th>
                </tr>
            """

            # Show top 10 pages
            sorted_pages = sorted(
                page_breakdown.items(), key=lambda x: x[1], reverse=True
            )[:10]
            for path, count in sorted_pages:
                percentage = (
                    (count / total_page_views * 100) if total_page_views > 0 else 0
                )
                html += f"""
                <tr>
                    <td>{path}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
                """

            html += """
            </table>
            """

        html += """
        </div>
        """

    html += """
        <div class="section">
            <p><em>Report generated automatically by Vegan Confirmed Analytics</em></p>
            <p><a href="https://db-ip.com">IP Geolocation by DB-IP</a></p>
        </div>
    </body>
    </html>
    """

    return html


def get_records_for_period(
    firestore_service: FirestoreService, start_date: datetime, end_date: datetime
) -> List:
    """
    Get all records from Firestore for the specified date range.

    Args:
        firestore_service: FirestoreService instance
        start_date: Start date for the range (inclusive)
        end_date: End date for the range (inclusive)

    Returns:
        List of APICallRecord objects
    """
    # Get all records (filter by date in memory since Firestore queries are limited)
    all_records = firestore_service.get_api_calls(limit=10000)

    # Filter records by date range
    filtered_records = []
    for record in all_records:
        if record.created_at and start_date <= record.created_at <= end_date:
            filtered_records.append(record)

    logger.info(f"Found {len(filtered_records)} records for the period")
    return filtered_records


def generate_report_data(
    records: List,
    start_date: datetime,
    end_date: datetime,
    cloud_logging_service: CloudLoggingService = None,
    extension_store_service: ExtensionStoreService = None,
) -> Dict[str, Any]:
    """
    Generate report data from API call records.

    Args:
        records: List of APICallRecord objects
        start_date: Start date for the report
        end_date: End date for the report
        cloud_logging_service: CloudLoggingService instance for page visit stats
        extension_store_service: ExtensionStoreService instance for store user counts

    Returns:
        Dictionary containing report statistics
    """
    # Initialize counters
    total_calls = len(records)
    # Keyed by IP hash.
    client_breakdown = defaultdict(int)
    client_countries = {}
    country_breakdown = defaultdict(int)
    vegan_breakdown = {"vegan": 0, "non_vegan": 0, "unknown": 0}
    trigger_breakdown = defaultdict(int)

    # How requests ended, and what the whole week spent.
    status_breakdown = defaultdict(int)
    overall_stats = _empty_call_stats()
    # Same counters per provider, so Gemini and the local machine can be
    # compared on volume and on tokens.
    provider_stats = defaultdict(_empty_call_stats)

    # What the model decided each page was. A request that failed classified
    # nothing, so it lands under "unknown".
    page_kind_breakdown = defaultdict(int)
    # The two analysis kinds, counted by the branch that was actually filled in.
    item_stats = _empty_call_stats()
    menu_stats = _empty_call_stats()
    menu_stats["dishes"] = 0
    menu_friendliness = defaultdict(int)

    # Track item domains
    item_domain_breakdown = defaultdict(int)

    feedback = []

    # Process each record
    for record in records:
        if record.ip_hash:
            client_breakdown[record.ip_hash] += 1
            client_countries[record.ip_hash] = record.country
        country_breakdown[record.country or "Unknown"] += 1

        status = record.status or STATUS_OK
        succeeded = status == STATUS_OK
        status_breakdown[status] += 1

        for stats in (overall_stats, provider_stats[record.service or NO_PROVIDER]):
            stats["calls"] += 1
            stats["ok"] += int(succeeded)
            _add_token_usage(stats, record.token_usage)

        page_kind_breakdown[record.page_kind or "unknown"] += 1

        # Vegan breakdown. Only a shopping item has a single vegan verdict; a
        # menu answers per dish, so it counts towards neither.
        if record.shopping_item:
            item_stats["calls"] += 1
            item_stats["ok"] += int(succeeded)
            _add_token_usage(item_stats, record.token_usage)

            is_vegan = record.shopping_item.is_vegan
            if is_vegan is True:
                vegan_breakdown["vegan"] += 1
            elif is_vegan is False:
                vegan_breakdown["non_vegan"] += 1
            else:
                vegan_breakdown["unknown"] += 1

        if record.menu:
            menu_stats["calls"] += 1
            menu_stats["ok"] += int(succeeded)
            _add_token_usage(menu_stats, record.token_usage)
            menu_stats["dishes"] += len(record.menu.items)
            menu_friendliness[record.menu.vegan_friendliness or "unknown"] += 1

        if record.feedback:
            feedback.append(_feedback_entry(record))

        # Trigger type breakdown
        trigger_type = record.trigger_type or "unknown"
        trigger_breakdown[trigger_type] += 1

        # Item domain breakdown
        if record.item_url:
            try:
                domain = urlparse(record.item_url).netloc
                if domain:
                    item_domain_breakdown[domain] += 1
            except Exception:
                pass

    # Format dates for display
    week_start = start_date.strftime("%Y-%m-%d")
    week_end = end_date.strftime("%Y-%m-%d")

    # Fetch page visit stats from Firebase Hosting logs
    page_visit_stats = None
    if cloud_logging_service:
        try:
            logger.info("Fetching page visit statistics from Firebase Hosting logs")
            page_visit_stats = cloud_logging_service.get_page_visits(
                start_date, end_date
            )
            logger.info(
                f"Page visits: {page_visit_stats.get('total_page_views', 0)} views, "
                f"{page_visit_stats.get('unique_visitors', 0)} unique visitors"
            )
        except Exception as e:
            logger.error(f"Failed to fetch page visit stats: {e}")
            page_visit_stats = {"error": str(e)}

    extension_users = None
    if extension_store_service:
        try:
            extension_users = extension_store_service.get_user_stats(start_date)
            logger.info(f"Extension users: {extension_users['stores']}")
        except Exception as e:
            logger.error(f"Failed to fetch extension user counts: {e}")
            extension_users = {"error": str(e)}

    report_data = {
        "week_start": week_start,
        "week_end": week_end,
        "total_calls": total_calls,
        "successful_calls": overall_stats["ok"],
        "failed_calls": total_calls - overall_stats["ok"],
        "success_rate": _pct(overall_stats["ok"], total_calls),
        "status_breakdown": dict(status_breakdown),
        "call_stats": overall_stats,
        "provider_stats": {
            service: stats for service, stats in sorted(provider_stats.items())
        },
        "page_kind_breakdown": dict(page_kind_breakdown),
        "item_stats": item_stats,
        "menu_stats": {**menu_stats, "friendliness": dict(menu_friendliness)},
        "client_breakdown": dict(client_breakdown),
        "client_countries": client_countries,
        "country_breakdown": dict(country_breakdown),
        "vegan_breakdown": vegan_breakdown,
        "trigger_breakdown": dict(trigger_breakdown),
        "item_domain_breakdown": dict(item_domain_breakdown),
        "feedback": sorted(
            feedback, key=lambda entry: entry["given_at"] or "", reverse=True
        ),
        "unique_clients": len(client_breakdown),
        "page_visit_stats": page_visit_stats,
        "extension_users": extension_users,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        f"Generated report data: {total_calls} calls, {len(client_breakdown)} unique clients, "
        f"{_pct(overall_stats['ok'], total_calls):.1f}% success, "
        f"{item_stats['calls']} item and {menu_stats['calls']} menu analyses, "
        f"{overall_stats['total_tokens']} tokens, {len(feedback)} feedback"
    )
    return report_data
