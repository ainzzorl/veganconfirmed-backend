"""What records keep instead of the client IP and the full URL.

The IP becomes a country, from DB-IP's local database (CC BY 4.0,
https://db-ip.com; see scripts/fetch_geoip.sh), and an HMAC under
``IP_HASH_SECRET``. The hash is a pseudonym, not anonymity: whoever holds the
secret can brute-force IPv4 back.
"""

import hashlib
import hmac
import ipaddress
import logging
import os
import re
from pathlib import Path
from typing import NamedTuple, Optional
from urllib.parse import urlsplit, urlunsplit

import maxminddb

logger = logging.getLogger(__name__)

GEOIP_DB_DEFAULT = (
    Path(__file__).resolve().parent.parent / "data" / "dbip-country-lite.mmdb"
)

IP_HASH_LENGTH = 16  # hex digits


def strip_url(url: str) -> str:
    """``url`` without query string, fragment or credentials, which hold
    session tokens and tracking IDs but rarely anything about the page."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return re.split(r"[?#]", url, maxsplit=1)[0]
    netloc = parts.netloc.rpartition("@")[2]
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


class ClientIp(NamedTuple):
    ip_hash: Optional[str]
    country: Optional[str]  # ISO 3166-1 alpha-2


def _parse(ip: str) -> Optional[ipaddress._BaseAddress]:
    try:
        return ipaddress.ip_address(ip.strip())
    except (AttributeError, ValueError):
        return None


class IpPrivacy:
    """Without the secret or the database, the matching field is ``None``."""

    def __init__(self, secret: Optional[str] = None, geoip_db: Optional[str] = None):
        self.secret = secret if secret is not None else os.getenv("IP_HASH_SECRET")
        if not self.secret:
            logger.warning("IP_HASH_SECRET is not set; records will carry no ip_hash")

        path = geoip_db or os.getenv("GEOIP_DB_PATH") or GEOIP_DB_DEFAULT
        try:
            self.reader: Optional[maxminddb.Reader] = maxminddb.open_database(str(path))
        except (OSError, maxminddb.InvalidDatabaseError) as e:
            logger.warning(
                f"No GeoIP database at {path} ({e}); records will carry no "
                f"country. Run scripts/fetch_geoip.sh."
            )
            self.reader = None

    def describe(self, ip: str) -> ClientIp:
        address = _parse(ip)
        if address is None:
            return ClientIp(None, None)
        return ClientIp(self._hash(address), self._country(address))

    def _hash(self, address: ipaddress._BaseAddress) -> Optional[str]:
        if not self.secret:
            return None
        digest = hmac.new(
            self.secret.encode(), str(address).encode(), hashlib.sha256
        ).hexdigest()
        return digest[:IP_HASH_LENGTH]

    def _country(self, address: ipaddress._BaseAddress) -> Optional[str]:
        if self.reader is None or not address.is_global:
            return None
        try:
            found = self.reader.get(str(address))
        except ValueError:
            return None
        if not isinstance(found, dict):
            return None
        return (found.get("country") or {}).get("iso_code")
