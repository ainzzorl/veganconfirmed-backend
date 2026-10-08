"""Convert records from before services/privacy.py: IP to hash and country, and
URLs stripped, in api_calls and in the prompts stored in lms_jobs.

    uv run python scripts/backfill_privacy.py [--apply]   # dry run without it
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from google.cloud.firestore import DELETE_FIELD  # noqa: E402

from services.desktop_service import jobs_collection_name  # noqa: E402
from services.firestore_service import FirestoreService  # noqa: E402
from services.privacy import IpPrivacy, strip_url  # noqa: E402

PROMPT_URL_LINE = re.compile(r"^(\s*(?:Webpage|Page) URL: )(\S+)", re.MULTILINE)

BATCH_SIZE = 400


def api_call_update(data: dict, ip_privacy: IpPrivacy) -> dict:
    update = {}
    if "ip_address" in data:
        client = ip_privacy.describe(str(data["ip_address"] or ""))
        update.update(
            ip_hash=client.ip_hash, country=client.country, ip_address=DELETE_FIELD
        )
    for field in ("item_url", "origin_url"):
        url = data.get(field)
        if isinstance(url, str) and strip_url(url) != url:
            update[field] = strip_url(url)
    return update


def job_update(data: dict) -> dict:
    request = data.get("request")
    if not isinstance(request, dict):
        return {}
    messages = request.get("messages") or []
    changed = False
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            stripped = PROMPT_URL_LINE.sub(
                lambda m: m.group(1) + strip_url(m.group(2)), content
            )
            if stripped != content:
                message["content"] = stripped
                changed = True
    return {"request.messages": messages} if changed else {}


def backfill(db, collection: str, make_update, apply: bool) -> None:
    batch, pending, updated, scanned = db.batch(), 0, 0, 0
    for doc in db.collection(collection).stream():
        scanned += 1
        update = make_update(doc.to_dict())
        if not update:
            continue
        updated += 1
        if apply:
            batch.update(doc.reference, update)
            pending += 1
            if pending == BATCH_SIZE:
                batch.commit()
                batch, pending = db.batch(), 0
    if apply and pending:
        batch.commit()
    verb = "Updated" if apply else "Would update"
    print(f"{collection}: {verb} {updated} of {scanned} documents")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="write the changes")
    args = parser.parse_args()

    ip_privacy = IpPrivacy()
    # The addresses are deleted, so anything not derived now is lost.
    if not ip_privacy.secret or ip_privacy.reader is None:
        sys.exit("IP_HASH_SECRET and the GeoIP database are both required")

    firestore_service = FirestoreService()
    db = firestore_service.db
    backfill(
        db,
        firestore_service.collection_name,
        lambda data: api_call_update(data, ip_privacy),
        args.apply,
    )
    backfill(db, jobs_collection_name(), job_update, args.apply)
    if not args.apply:
        print("Dry run; pass --apply to write.")


if __name__ == "__main__":
    main()
