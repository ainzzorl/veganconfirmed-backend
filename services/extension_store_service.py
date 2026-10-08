"""
User counts for the browser extension, as the Firefox and Chrome stores report them.

Each report run saves what it read, so the next one can show the change since.
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import requests
from google.cloud.firestore import Query

logger = logging.getLogger(__name__)

SNAPSHOT_COLLECTION = "extension_store_snapshots"

FIREFOX_SLUG = "vegan-confirmed"
CHROME_ID = "aojinnpopgkbcamgphdnbonlidflihed"

# AMO's public add-on API. ``average_daily_users`` is what the listing shows.
FIREFOX_URL = f"https://addons.mozilla.org/api/v5/addons/addon/{FIREFOX_SLUG}/"
# The Chrome Web Store has no API for user counts, so the listing page is read
# instead. ``hl=en`` keeps the "N users" text in English.
CHROME_URL = f"https://chromewebstore.google.com/detail/{CHROME_ID}?hl=en"
# The count sits right after the category link, e.g. ">Shopping</a>29 users</div>".
# Google rounds large counts ("21,000,000 users") and may append a "+".
CHROME_USERS_RE = re.compile(r">([\d,]+)\+? users?<")

STORE_LABELS = {"firefox": "Firefox", "chrome": "Chrome"}

# The previous snapshot is the newest one taken by the start of the report's
# period. The slack lets last week's scheduled run, which fires a few seconds
# either side of exactly seven days ago, still count.
PREVIOUS_SNAPSHOT_SLACK = timedelta(days=1)

REQUEST_TIMEOUT = 15


def _fetch_firefox_users() -> int:
    response = requests.get(FIREFOX_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return int(response.json()["average_daily_users"])


def _fetch_chrome_users() -> int:
    response = requests.get(
        CHROME_URL,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US"},
    )
    response.raise_for_status()
    match = CHROME_USERS_RE.search(response.text)
    if not match:
        raise ValueError("user count not found on the Chrome Web Store page")
    return int(match.group(1).replace(",", ""))


FETCHERS = {"firefox": _fetch_firefox_users, "chrome": _fetch_chrome_users}


class ExtensionStoreService:
    def __init__(self, db):
        """``db`` is a Firestore client, e.g. ``FirestoreService().db``."""
        self.db = db

    def fetch_users(self) -> Dict[str, Optional[int]]:
        """Current user count per store; ``None`` for a store that could not be read."""
        users = {}
        for store, fetch in FETCHERS.items():
            try:
                users[store] = fetch()
            except Exception as e:
                logger.warning(f"Could not read {store} user count: {e}")
                users[store] = None
        return users

    def get_user_stats(self, start_date: datetime) -> Dict[str, Any]:
        """Read the current counts, compare them to the snapshot from around
        ``start_date``, and save them as the next run's snapshot."""
        now = datetime.now(timezone.utc)
        users = self.fetch_users()
        previous = self._previous_snapshot(start_date + PREVIOUS_SNAPSHOT_SLACK)

        if any(count is not None for count in users.values()):
            self.db.collection(SNAPSHOT_COLLECTION).add({"taken_at": now, **users})

        previous_at = previous.get("taken_at") if previous else None
        return {
            "stores": {
                store: {
                    "users": count,
                    "previous": previous.get(store) if previous else None,
                }
                for store, count in users.items()
            },
            "previous_at": previous_at.isoformat() if previous_at else None,
        }

    def _previous_snapshot(self, cutoff: datetime) -> Optional[Dict[str, Any]]:
        docs = (
            self.db.collection(SNAPSHOT_COLLECTION)
            .where("taken_at", "<=", cutoff)
            .order_by("taken_at", direction=Query.DESCENDING)
            .limit(1)
            .stream()
        )
        return next((doc.to_dict() for doc in docs), None)
