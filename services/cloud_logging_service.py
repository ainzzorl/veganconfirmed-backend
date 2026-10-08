"""
Cloud Logging service for querying Firebase Hosting logs.

This module provides functionality to fetch and analyze page visit statistics
from Firebase Hosting logs stored in Google Cloud Logging.
"""

import os
import re
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Set
from collections import defaultdict

from google.cloud import logging as cloud_logging
from google.api_core.exceptions import ResourceExhausted

logger = logging.getLogger(__name__)

# Asset file extensions to exclude
ASSET_EXTENSIONS = {
    ".js",
    ".css",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
    ".map",
    ".webp",
    ".avif",
    ".mp4",
    ".webm",
    ".mp3",
    ".ogg",
    ".pdf",
    ".zip",
    ".json",
    ".xml",
}

# Bot User-Agent patterns (case-insensitive)
BOT_PATTERNS = [
    r"googlebot",
    r"bingbot",
    r"yandexbot",
    r"duckduckbot",
    r"baiduspider",
    r"facebookexternalhit",
    r"twitterbot",
    r"linkedinbot",
    r"slackbot",
    r"applebot",
    r"ahrefsbot",
    r"semrushbot",
    r"mj12bot",
    r"dotbot",
    r"petalbot",
    r"bytespider",
    r"gptbot",
    r"claudebot",
    r"anthropic",
    r"crawler",
    r"spider",
    r"bot/",
    r"bot;",
    r"headless",
    r"phantom",
    r"selenium",
    r"puppeteer",
    r"playwright",
    r"curl/",
    r"wget/",
    r"python-requests",
    r"python-urllib",
    r"java/",
    r"libwww",
    r"apache-httpclient",
    r"go-http-client",
    r"okhttp",
]

# Compile bot patterns for efficiency
BOT_REGEX = re.compile("|".join(BOT_PATTERNS), re.IGNORECASE)

# Path segments that mark a vulnerability scan rather than a visit. Scanners
# probe the same software under any prefix they can guess -- "/wp/wp-json/...",
# "/blog/wp-json/..." and "/wp/wordpress/wp-json/..." are one scan wearing three
# paths -- so a marker anywhere in the path disqualifies the request, not only
# at its start.
BOT_PATH_SEGMENTS = frozenset(
    {
        "wp",  # WordPress, however deep it is buried
        "wordpress",
        "xmlrpc.php",
        "administrator",  # Joomla
        "admin",
        "phpmyadmin",
        "pma",
        "mysql",
        "cgi-bin",
        "shell",
        "cmd",
        ".env",
        ".git",
        ".svn",
        ".htaccess",
        ".htpasswd",
        "config.php",
        "config.yml",
        "backup",
        "db",
        "database",
        "sql",
        "jsonws",  # Liferay
        "vendor",
        "phpunit",
        "elfinder",
        "filemanager",
        "console",
        "manager",
        "solr",
        "jenkins",
        "actuator",  # Spring Boot
        "boaform",  # Router exploits
        "hnap1",  # D-Link router
        "sdk",
        "remote",
        "evox",
        "stalker_portal",
        "telescope",  # Laravel Telescope
        "debug",
        "uploads",
        "modules",
        "alfa_data",
        ".well-known",
    }
)

# Segment prefixes, for families whose members are not worth listing one by one:
# WordPress hangs everything off wp-something (wp-admin, wp-login.php,
# wp-content, wp-json, wp-includes).
BOT_SEGMENT_PREFIXES = ("wp-", "mini")


def _get_project_id() -> str:
    """
    Try to determine the GCP project ID from various sources.

    Returns:
        The project ID if found, None otherwise.
    """
    # Try common environment variables
    for env_var in [
        "GOOGLE_CLOUD_PROJECT",
        "GCP_PROJECT",
        "GCLOUD_PROJECT",
        "FIREBASE_PROJECT_ID",
    ]:
        project_id = os.environ.get(env_var)
        if project_id:
            return project_id

    # Try to get from firebase-admin if initialized
    try:
        import firebase_admin

        if firebase_admin._apps:
            app = firebase_admin.get_app()
            if app.project_id:
                return app.project_id
    except Exception:
        pass

    return None


class CloudLoggingService:
    """Service for querying Firebase Hosting logs from Cloud Logging."""

    def __init__(self, project_id: str = None):
        """
        Initialize the Cloud Logging service.

        Args:
            project_id: GCP project ID. If not provided, tries to auto-detect.
        """
        self.project_id = project_id or _get_project_id()
        if not self.project_id:
            raise ValueError(
                "Project ID must be provided or set via GOOGLE_CLOUD_PROJECT env var"
            )
        self.client = cloud_logging.Client(project=self.project_id)

    def _is_asset_request(self, path: str) -> bool:
        """
        Check if a request path is for a static asset.

        Args:
            path: The request path

        Returns:
            True if the path is for an asset, False otherwise
        """
        if not path:
            return True

        # Check file extension
        path_lower = path.lower()
        for ext in ASSET_EXTENSIONS:
            if path_lower.endswith(ext):
                return True

        # Check common asset directories and excluded paths
        if any(
            segment in path_lower
            for segment in [
                "/assets/",
                "/static/",
                "/_next/",
                "/__/",
                ".git",
                "/images/",
                "/files/",
                "/media/",
            ]
        ):
            return True

        return False

    def _is_bot_request(self, user_agent: str) -> bool:
        """
        Check if a request is from a bot based on User-Agent.

        Args:
            user_agent: The User-Agent header value

        Returns:
            True if the request appears to be from a bot, False otherwise
        """
        if not user_agent:
            return True  # No user agent is suspicious

        return bool(BOT_REGEX.search(user_agent))

    def _is_bot_path(self, path: str) -> bool:
        """
        Check if a request path is commonly targeted by bots/scanners.

        These paths are typically probed by vulnerability scanners looking
        for WordPress, phpMyAdmin, and other common exploitable software.

        Args:
            path: The request path

        Returns:
            True if the path is commonly targeted by bots, False otherwise
        """
        if not path:
            return False

        return any(
            segment in BOT_PATH_SEGMENTS or segment.startswith(BOT_SEGMENT_PREFIXES)
            for segment in path.lower().split("/")
            if segment
        )

    def _is_page_request(self, path: str) -> bool:
        """
        Check if a request path is for a page (not an asset).

        Args:
            path: The request path

        Returns:
            True if the path is for a page, False otherwise
        """
        if not path:
            return False

        # Exclude asset requests
        if self._is_asset_request(path):
            return False

        path_lower = path.lower()

        # Include paths ending with / (directory index)
        if path_lower.endswith("/"):
            return True

        # Include .html files
        if path_lower.endswith(".html"):
            return True

        # Include paths with no extension (likely clean URLs)
        # Check if the last segment has no dot
        last_segment = path.split("/")[-1]
        if "." not in last_segment:
            return True

        return False

    def _iterate_with_rate_limit_handling(self, entries_iterator, max_retries: int = 5):
        """
        Iterate through log entries with rate limit handling.

        Uses exponential backoff when rate limits are hit.

        Args:
            entries_iterator: The iterator from list_entries()
            max_retries: Maximum number of retries per rate limit error

        Yields:
            Log entries from the iterator
        """
        retry_count = 0
        base_delay = 10  # Start with 10 seconds

        while True:
            try:
                for entry in entries_iterator:
                    yield entry
                    retry_count = 0  # Reset on successful iteration
                # Iterator exhausted normally
                break
            except ResourceExhausted as e:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded for rate limit")
                    raise

                # Exponential backoff: 10s, 20s, 40s, 80s, 160s
                delay = base_delay * (2 ** (retry_count - 1))
                logger.warning(
                    f"Rate limit hit, waiting {delay}s before retry "
                    f"({retry_count}/{max_retries}): {e}"
                )
                time.sleep(delay)
                # Continue iteration - the iterator maintains its position

    def get_page_visits(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """
        Get page visit statistics from Firebase Hosting logs.

        Args:
            start_date: Start of the date range (inclusive)
            end_date: End of the date range (inclusive)

        Returns:
            Dictionary containing:
            - total_page_views: Total number of page views
            - unique_visitors: Number of unique IP addresses
            - page_breakdown: Dict of path -> view count
            - daily_views: Dict of date -> view count
        """
        logger.info(f"Fetching Firebase Hosting logs from {start_date} to {end_date}")

        # Build the filter query
        # Format timestamps for Cloud Logging
        start_ts = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_ts = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        filter_str = (
            f'resource.type="firebase_domain" '
            f'timestamp>="{start_ts}" '
            f'timestamp<="{end_ts}"'
        )

        logger.info(f"Using filter: {filter_str}")

        # Initialize counters
        page_views = 0
        unique_ips: Set[str] = set()
        page_breakdown: Dict[str, int] = defaultdict(int)
        daily_views: Dict[str, int] = defaultdict(int)
        domain_breakdown: Dict[str, int] = defaultdict(int)
        excluded_bots = 0
        excluded_assets = 0
        excluded_bot_paths = 0

        try:
            # Query logs with large page_size to reduce API calls
            # and handle rate limiting with retries
            entries_iterator = self.client.list_entries(
                filter_=filter_str,
                page_size=1000,  # Max allowed, reduces API calls
            )

            # Process entries with rate limit handling
            for entry in self._iterate_with_rate_limit_handling(entries_iterator):
                # Extract HTTP request info from the log entry
                http_request = getattr(entry, "http_request", None)
                if not http_request:
                    # Try payload
                    payload = entry.payload
                    if isinstance(payload, dict):
                        http_request = payload.get("httpRequest", {})
                    else:
                        continue

                # Get request details
                if isinstance(http_request, dict):
                    request_url = http_request.get("requestUrl", "")
                    user_agent = http_request.get("userAgent", "")
                    remote_ip = http_request.get("remoteIp", "")
                    status = http_request.get("status", 0)
                else:
                    request_url = getattr(http_request, "request_url", "")
                    user_agent = getattr(http_request, "user_agent", "")
                    remote_ip = getattr(http_request, "remote_ip", "")
                    status = getattr(http_request, "status", 0)

                # Only count successful responses
                if status and status >= 400:
                    continue

                # Extract path and domain from URL
                if request_url:
                    # Handle full URLs
                    if request_url.startswith("http"):
                        from urllib.parse import urlparse

                        parsed = urlparse(request_url)
                        path = parsed.path or "/"
                        domain = parsed.netloc or "unknown"
                    else:
                        path = request_url.split("?")[0]  # Remove query string
                        domain = "unknown"
                else:
                    continue

                # Check if it's an asset request
                if self._is_asset_request(path):
                    excluded_assets += 1
                    continue

                # Check if it's a bot request
                if self._is_bot_request(user_agent):
                    excluded_bots += 1
                    continue

                # Check if path is commonly targeted by bots
                if self._is_bot_path(path):
                    excluded_bot_paths += 1
                    continue

                # Check if it's a page request
                if not self._is_page_request(path):
                    continue

                # Count the page view
                page_views += 1
                page_breakdown[path] += 1
                domain_breakdown[domain] += 1

                if remote_ip:
                    unique_ips.add(remote_ip)

                # Track daily views
                if hasattr(entry, "timestamp") and entry.timestamp:
                    date_str = entry.timestamp.strftime("%Y-%m-%d")
                    daily_views[date_str] += 1

            logger.info(
                f"Processed logs: {page_views} page views, "
                f"{len(unique_ips)} unique IPs, "
                f"{excluded_bots} bot requests excluded, "
                f"{excluded_bot_paths} bot path requests excluded, "
                f"{excluded_assets} asset requests excluded"
            )

        except Exception as e:
            logger.error(f"Error fetching Cloud Logging data: {e}")
            # Return empty stats on error
            return {
                "total_page_views": 0,
                "unique_visitors": 0,
                "page_breakdown": {},
                "domain_breakdown": {},
                "daily_views": {},
                "error": str(e),
            }

        return {
            "total_page_views": page_views,
            "unique_visitors": len(unique_ips),
            "page_breakdown": dict(page_breakdown),
            "domain_breakdown": dict(domain_breakdown),
            "daily_views": dict(daily_views),
        }
