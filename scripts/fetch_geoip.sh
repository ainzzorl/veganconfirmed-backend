#!/bin/bash
#
# Download DB-IP's IP-to-Country Lite database (CC BY 4.0, https://db-ip.com)
# for services/privacy.py. Falls back to last month's edition early in a month,
# before the new one is out.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEST=data/dbip-country-lite.mmdb
mkdir -p "$(dirname "$DEST")"

this_month="$(date -u +%Y-%m)"
# BSD date, then GNU date.
last_month="$(date -u -v1d -v-1m +%Y-%m 2>/dev/null || date -u -d "$(date -u +%Y-%m-01) -1 month" +%Y-%m)"

for month in "$this_month" "$last_month"; do
  if curl -fsSL "https://download.db-ip.com/free/dbip-country-lite-$month.mmdb.gz" -o "$DEST.gz.tmp"; then
    gunzip -c "$DEST.gz.tmp" > "$DEST.tmp"
    mv "$DEST.tmp" "$DEST"
    rm -f "$DEST.gz.tmp"
    echo "GeoIP database: DB-IP $month edition at $DEST"
    exit 0
  fi
done
rm -f "$DEST.gz.tmp"
echo "Error: could not download the DB-IP country database" >&2
exit 1
