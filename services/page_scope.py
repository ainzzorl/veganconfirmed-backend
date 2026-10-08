"""Which page kinds a request could plausibly be, decided before the model runs.

``page_prompt`` asks one model call to do two things: classify the page, then
analyze it as whatever it decided. That is a multi-intent prompt, and the
prompt pays for it twice — in tokens (both branches are described whether or
not they are used) and in accuracy, since the small models this runs on lose
track of "fill in one branch and leave the other null".

Some requests do not need the question asked. Not because we can classify the
page — we cannot, and this module never tries — but because *the branch is
impossible or irrelevant for that request*. Excluding a branch is a far weaker
claim than choosing one, and it is the only claim made here:

- The **automatic** trigger fires from the extension's add-to-cart handler, and
  the flow behind it exists to warn someone before they buy a non-vegan
  product. It is the only interruptive path (see ``applyAnalysisOutcome`` in the
  extension's background.js); a menu verdict on it reaches nobody. So the menu
  branch is dropped there regardless of what the page turns out to be — a
  DoorDash menu under this scope is simply "other", which is what that flow
  wants.
- The **Google Maps** sources come from the extractor that reads a place
  panel's menu tab. A product cannot arrive that way.
- A handful of **retail domains** never serve a restaurant menu. Amazon alone is
  half of all production traffic, and no menu has ever been recorded on any of
  these hosts.
- Some pages **declare their own kind**, in metadata the extension forwards as
  ``page_signals``: the ``og:type`` they name, and the schema.org ``@type``
  values of their JSON-LD and microdata. A page carrying a
  ``Menu``/``MenuSection``/``MenuItem`` graph is describing a menu — those types
  exist for nothing else — and a page carrying ``Product`` is describing
  something for sale. The claim is precise because the site is making it about
  itself, not because we read its text.
- Some sites **declare their platform**, through the hosts they load scripts,
  stylesheets and frames from — also forwarded in ``page_signals``. A site built
  on SpotHopper or BentoBox, ordering through Toast or DoorDash, taking bookings
  through Resy or OpenTable, is a restaurant's site: those platforms serve
  nothing else. Most restaurants declare no schema at all, so this is the only
  signal a page like ``agave-mexican-bistro`` in the eval set carries. It is a
  claim about the site rather than the page, which is why a page declaring
  itself a product is exempt from it: the merch page on a restaurant platform
  still gets the item branch.

Every scope keeps ``"other"`` reachable. That is what makes pruning safe: a
request whose exclusion was wrong for the page in front of it falls back to "I
don't know what this is" rather than to a confident wrong verdict. The
scope-picking rules must never remove the branch the page actually needs *and*
"other" at the same time, so a scope is always two kinds, never one.
"""

import logging
from typing import Any, NamedTuple, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# Every kind the model may be asked to choose from. ``page_prompt.PAGE_KINDS``
# is derived from this so the two cannot drift.
ALL_PAGE_KINDS = ("shopping_item", "restaurant_menu", "other")

# The pruned scopes. "other" is in both, deliberately — see the module docstring.
ITEM_OR_OTHER = ("shopping_item", "other")
MENU_OR_OTHER = ("restaurant_menu", "other")

# Extractors that only ever produce restaurant menus. ``source`` values come
# from the extension (``google_maps_menu_tab``) and its Maps panel path; "page"
# and "pdf" say nothing about the kind and are absent here.
MENU_ONLY_SOURCES = frozenset({"google_maps_menu_tab", "google_maps_panel"})

# The trigger the extension's add-to-cart handler sets.
AUTOMATIC_TRIGGER = "automatic"

# Retail hosts that sell products and never publish a restaurant menu.
#
# Chosen from production traffic rather than from imagination: these are the
# domains that actually appear, ordered by volume, and across every URL on
# record no menu has been seen on any of them. The list is worth keeping short
# — it exists to cover the *manual* requests on big retailers that the
# automatic-trigger rule already covers, which is why it is sorted by how much
# non-automatic traffic each one carries.
#
# Note on IKEA: it is the one entry that genuinely operates restaurants and
# publishes their menus. Such a page lands on "other" here rather than getting
# a wrong verdict, which is an acceptable trade at its volume.
SHOPPING_ONLY_DOMAINS = frozenset(
    {
        "kohls.com",
        "walmart.com",
        "asda.com",
        "ozon.ru",
        "tesco.com",
        "sainsburys.co.uk",
        "dm.de",
        "flipkart.com",
        "matspar.se",
        "target.com",
        "homedepot.com",
        "lowes.com",
        "acehardware.com",
        "advanceautoparts.com",
        "aliexpress.com",
        "uniqlo.com",
    }
)

# Retailers that operate a domain per country, matched on the leading label of
# the registrable domain so amazon.com, amazon.co.uk and amazon.com.au are one
# entry rather than a list that goes stale every time a new market appears.
SHOPPING_ONLY_BRANDS = frozenset({"amazon", "ikea", "ebay", "decathlon"})

# Two-part public suffixes we actually see, so "amazon.co.uk" reduces to itself
# rather than to "co.uk". Not exhaustive, and does not need to be: an unlisted
# suffix yields a longer domain that simply fails to match, which costs a
# pruning opportunity and never causes a wrong one.
_MULTI_PART_SUFFIXES = frozenset(
    {
        "co.uk",
        "co.jp",
        "com.au",
        "co.nz",
        "com.br",
        "co.il",
        "com.mx",
        "co.kr",
        "com.tr",
        "com.sg",
        "co.za",
        "com.hk",
        "co.in",
        "com.ar",
    }
)

# schema.org types that exist only to describe a menu. Kept deliberately narrow:
# "Restaurant", "FoodEstablishment" and the "hasMenu" property are *not* here,
# because a restaurant's site-wide schema graph is routinely injected into every
# page it serves — including the one selling gift cards and merch.
MENU_SCHEMA_TYPES = frozenset({"menu", "menusection", "menuitem"})

# schema.org types that describe something for sale. "Offer" is deliberately
# absent: a menu prices its dishes with it.
ITEM_SCHEMA_TYPES = frozenset(
    {
        "product",
        "productgroup",
        "productmodel",
        "individualproduct",
        "someproducts",
    }
)

# Hosts that only restaurants load from: website builders, ordering platforms
# and booking widgets sold to restaurants and to nobody else. Matched as a
# domain suffix, so "static.spotapps.co" and "media-cdn.getbento.com" match
# their platform. Generic CDNs and CMSes are deliberately absent — a restaurant
# on Prismic or Cloudflare says nothing about what it is.
#
# The trade this makes is the one the IKEA entry above makes in reverse: a
# restaurant's merch or gift-card page, on a platform host and declaring no
# Product of its own, lands on "other" instead of an item verdict.
MENU_PLATFORM_HOSTS = frozenset(
    {
        # Restaurant website builders.
        "spotapps.co",
        "spothopperapp.com",
        "getbento.com",
        "popmenu.com",
        "owner.com",
        # Ordering platforms.
        "toasttab.com",
        "cdn4dd.com",
        "olo.com",
        "olocdn.net",
        "chownow.com",
        "menufy.com",
        "slicelife.com",
        # Reservations and guest management.
        "opentable.com",
        "resy.com",
        "exploretock.com",
        "sevenrooms.com",
        "wisely.io",
    }
)

# The most hosts worth reading off one page, mirroring MAX_SCHEMA_TYPES: the
# payload is client-supplied, and the rule needs one match, not an inventory.
MAX_ASSET_HOSTS = 40

# Open Graph object types, matched on the part before the subtype so that
# "restaurant.menu" and "product.item" are covered by their family.
MENU_OG_TYPE = "restaurant"
ITEM_OG_TYPE = "product"

# The most schema types worth reading off one page. A page declaring more than
# this has a graph describing its whole site, not itself.
MAX_SCHEMA_TYPES = 40


class ScopeDecision(NamedTuple):
    """The scope a request got, and the name of the rule that chose it."""

    page_kinds: tuple[str, ...]
    rule: str


def registrable_domain(url: str) -> str:
    """Reduce a URL to the domain a rule can be written against.

    ``https://www.amazon.co.uk/dp/B0…`` becomes ``amazon.co.uk`` and
    ``https://smile.amazon.com/…`` becomes ``amazon.com``. Returns an empty
    string for anything unparseable — callers treat that as "no rule matched".
    """
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    if not host:
        return ""

    host = host.removeprefix("www.")
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if ".".join(labels[-2:]) in _MULTI_PART_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def is_shopping_only_domain(url: str) -> bool:
    """Whether ``url`` is on a host known never to serve a restaurant menu."""
    domain = registrable_domain(url)
    if not domain:
        return False
    return domain in SHOPPING_ONLY_DOMAINS or domain.split(".")[0] in (
        SHOPPING_ONLY_BRANDS
    )


def _normalized_signals(
    page_signals: Optional[dict[str, Any]],
) -> tuple[Optional[str], frozenset[str], frozenset[str]]:
    """The ``(og_type, schema_types, asset_hosts)`` a request declared.

    ``page_signals`` is client-supplied and advisory, so anything unexpected in
    it has to mean "declared nothing" rather than an error: a hint must never be
    able to fail an analysis. Types are lowercased and reduced to their bare
    name, so ``http://schema.org/Product`` and ``Product`` are one thing.
    """
    if not isinstance(page_signals, dict):
        return None, frozenset(), frozenset()

    raw_og_type = page_signals.get("og_type")
    og_type = (
        raw_og_type.strip().lower() if isinstance(raw_og_type, str) else ""
    ) or None

    schema_types = set()
    raw_types = page_signals.get("schema_types")
    if isinstance(raw_types, (list, tuple)):
        for value in raw_types[:MAX_SCHEMA_TYPES]:
            if isinstance(value, str) and value.strip():
                schema_types.add(value.strip().lower().rstrip("/").rsplit("/", 1)[-1])

    asset_hosts = set()
    raw_hosts = page_signals.get("asset_hosts")
    if isinstance(raw_hosts, (list, tuple)):
        for value in raw_hosts[:MAX_ASSET_HOSTS]:
            if isinstance(value, str) and value.strip():
                asset_hosts.add(value.strip().lower().strip("."))

    return og_type, frozenset(schema_types), frozenset(asset_hosts)


def _og_family(og_type: Optional[str]) -> str:
    """The family of an Open Graph type: "restaurant.menu" is a restaurant."""
    return og_type.split(".")[0] if og_type else ""


def _declares_item(og_type: Optional[str], schema_types: frozenset[str]) -> bool:
    """Whether either half of the item declaration is present."""
    return bool(schema_types & ITEM_SCHEMA_TYPES) or (
        _og_family(og_type) == ITEM_OG_TYPE
    )


def declares_menu_only(page_signals: Optional[dict[str, Any]]) -> bool:
    """Whether the page declares a menu and declares nothing for sale.

    The menu declaration stands on its own metadata, never on the page's prose
    or its title: ``clara-junction-menu`` in the eval set is a real menu with a
    full ``Menu``/``MenuItem`` graph that nonetheless calls itself
    ``og:type=article``, which is why a menu is never ruled *out* by og:type.
    """
    og_type, schema_types, _ = _normalized_signals(page_signals)
    declares_menu = bool(schema_types & MENU_SCHEMA_TYPES) or (
        _og_family(og_type) == MENU_OG_TYPE
    )
    return declares_menu and not _declares_item(og_type, schema_types)


def declares_shopping_only(page_signals: Optional[dict[str, Any]]) -> bool:
    """Whether the page declares something for sale and declares no menu.

    Both halves of the item declaration are needed, not either one: in the eval
    set ``thredup-wool-skirt`` carries ``Product`` JSON-LD with no ``og:type``,
    and ``polyester_pants`` declares ``og:type=product`` with no ``Product``.
    """
    og_type, schema_types, _ = _normalized_signals(page_signals)
    return _declares_item(og_type, schema_types) and not (
        schema_types & MENU_SCHEMA_TYPES or _og_family(og_type) == MENU_OG_TYPE
    )


def _on_menu_platform(host: str) -> bool:
    """Whether ``host`` is a menu platform or a subdomain of one."""
    labels = host.split(".")
    return any(
        ".".join(labels[i:]) in MENU_PLATFORM_HOSTS for i in range(len(labels) - 1)
    )


def declares_menu_platform(page_signals: Optional[dict[str, Any]]) -> bool:
    """Whether the site runs on a restaurant platform and sells nothing here.

    The exemption is what keeps this a claim about the page: a restaurant's shop
    page carries the same platform hosts as its menu, and the ``Product`` it
    declares is the half that is about the page in front of us.
    """
    og_type, schema_types, asset_hosts = _normalized_signals(page_signals)
    if _declares_item(og_type, schema_types):
        return False
    return any(_on_menu_platform(host) for host in asset_hosts)


def decide_scope(
    url: str,
    trigger_type: Optional[str] = None,
    source: Optional[str] = None,
    page_signals: Optional[dict[str, Any]] = None,
) -> ScopeDecision:
    """The page kinds this request could be, and which rule said so.

    The rules are ordered most-specific first; each is an exclusion, never a
    classification, and the reasoning behind each lives in the module docstring.

    The order is load-bearing in one place: what a page declares is read *after*
    the automatic trigger, never before it. An ordering platform serves a
    genuine ``Menu`` graph and an add-to-cart button on the same page, and on
    that click the interruptive flow still wants the item branch. The declared
    rules also sit below the domain list, which keeps behaviour on those hosts
    exactly as it was before pages were allowed to speak for themselves. The
    platform rule comes last of all, being the one claim about the site rather
    than about the page.
    """
    if source in MENU_ONLY_SOURCES:
        return ScopeDecision(MENU_OR_OTHER, "menu_source")

    if trigger_type == AUTOMATIC_TRIGGER:
        return ScopeDecision(ITEM_OR_OTHER, "automatic_trigger")

    if is_shopping_only_domain(url):
        return ScopeDecision(ITEM_OR_OTHER, "shopping_domain")

    if declares_menu_only(page_signals):
        return ScopeDecision(MENU_OR_OTHER, "declared_menu")

    if declares_shopping_only(page_signals):
        return ScopeDecision(ITEM_OR_OTHER, "declared_item")

    if declares_menu_platform(page_signals):
        return ScopeDecision(MENU_OR_OTHER, "menu_platform")

    return ScopeDecision(ALL_PAGE_KINDS, "none")


def scope_for_request(
    url: str,
    trigger_type: Optional[str] = None,
    source: Optional[str] = None,
    page_signals: Optional[dict[str, Any]] = None,
) -> tuple[str, ...]:
    """The page kinds this request could be — ``decide_scope`` without the rule.

    Returns one of ``ALL_PAGE_KINDS``, ``ITEM_OR_OTHER`` or ``MENU_OR_OTHER``.
    """
    return decide_scope(url, trigger_type, source, page_signals).page_kinds


def describe_scope(page_kinds: tuple[str, ...]) -> str:
    """A short label for logs, so the rule's reach can be read off production."""
    if page_kinds == ITEM_OR_OTHER:
        return "item_or_other"
    if page_kinds == MENU_OR_OTHER:
        return "menu_or_other"
    return "all"
