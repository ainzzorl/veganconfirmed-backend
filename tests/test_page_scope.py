"""Offline tests for the page-kind scope: which branch a request is asked about.

Pure functions over strings and dicts, so they run anywhere:

    uv run pytest tests/test_page_scope.py

What they protect, in two halves.

The rules (``page_scope``) may only ever *exclude* a branch, and may never
exclude the "other" escape along with it — a request whose exclusion turns out
to be wrong for the page in front of it has to be able to answer "I don't know
what this is" rather than a confident wrong verdict.

The pruning (``page_prompt``) has to be total and invisible: total, in that a
branch dropped from the instructions is dropped from the schema, the JSON field
list and the accepted ``page_kind`` values too; invisible, in that the response
handed back still carries everything the released extension reads, whichever
branch was pruned.

One test reaches past the pure functions, to ``AnalysisCore``: the rules that
read what a page declares about itself are only as good as the switch that lets
them run, and nothing else pins its default.
"""

from __future__ import annotations

import pytest

import analysis_core
from analysis_core import AnalysisCore
from models.analysis_model import PageAnalysis
from models.content_model import PageAnalysisRequest
from services.page_prompt import (
    build_gemini_page_schema,
    build_openai_page_schema,
    build_page_instructions,
    build_page_json_output_instruction,
    normalize_page_analysis,
)
from services.page_scope import (
    ALL_PAGE_KINDS,
    ITEM_OR_OTHER,
    MENU_OR_OTHER,
    decide_scope,
    describe_scope,
    registrable_domain,
    scope_for_request,
)

_SCOPES = [ALL_PAGE_KINDS, ITEM_OR_OTHER, MENU_OR_OTHER]


# --- the rules -------------------------------------------------------------


def test_automatic_trigger_drops_the_menu_branch():
    """The add-to-cart flow only ever interrupts for a product."""
    assert (
        scope_for_request("https://example-shop.test/p/1", trigger_type="automatic")
        == ITEM_OR_OTHER
    )


def test_automatic_trigger_on_a_delivery_site_still_drops_the_menu_branch():
    """Not a claim about the page — a claim about what that flow can act on.

    An add-to-cart click on a delivery marketplace happens over a real menu.
    The scope drops it anyway: the menu verdict would reach nobody, and "other"
    is the answer that flow wants.
    """
    assert (
        scope_for_request(
            "https://www.doordash.com/store/some-restaurant-123/",
            trigger_type="automatic",
        )
        == ITEM_OR_OTHER
    )


@pytest.mark.parametrize("source", ["google_maps_menu_tab", "google_maps_panel"])
def test_maps_extractors_drop_the_item_branch(source):
    assert (
        scope_for_request("https://www.google.com/maps/place/Somewhere", source=source)
        == MENU_OR_OTHER
    )


def test_the_extractor_outranks_the_trigger():
    """Both rules could fire; the one about the payload's origin is the surer."""
    assert (
        scope_for_request(
            "https://www.google.com/maps/place/Somewhere",
            trigger_type="automatic",
            source="google_maps_menu_tab",
        )
        == MENU_OR_OTHER
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://www.amazon.com/dp/B0BZT4MHDJ",
        "https://www.amazon.co.uk/checkout/p/p-220",
        "https://smile.amazon.com/gp/help",
        "https://www.kohls.com/product/prd-7838320/womens.jsp",
        "https://www.ikea.com/us/en/p/billy-bookcase-00263850/",
    ],
)
def test_retail_domains_drop_the_menu_branch_without_a_trigger(url):
    """Covers the manual requests on big retailers the trigger rule misses."""
    assert scope_for_request(url, trigger_type="manual") == ITEM_OR_OTHER


@pytest.mark.parametrize(
    "url",
    [
        "https://theolivebranch.example.com/menu",
        "https://order.online/store/udon-mugizo-1228295",
        "https://news.example.com/article",
        "not a url at all",
        "",
    ],
)
def test_an_unrecognized_request_is_asked_the_whole_question(url):
    assert scope_for_request(url, trigger_type="manual") == ALL_PAGE_KINDS


def test_pdf_and_page_sources_decide_nothing_on_their_own():
    for source in ("page", "pdf"):
        assert scope_for_request("https://a-restaurant.test/menu", source=source) == (
            ALL_PAGE_KINDS
        )


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.amazon.co.uk/dp/B0", "amazon.co.uk"),
        ("https://smile.amazon.com/x", "amazon.com"),
        ("https://kohls.com/", "kohls.com"),
        ("https://shop.advanceautoparts.com/p/x", "advanceautoparts.com"),
        ("https://sub.domain.example.org/x", "example.org"),
        ("", ""),
    ],
)
def test_registrable_domain(url, expected):
    assert registrable_domain(url) == expected


def test_a_lookalike_host_is_not_the_brand():
    """`amazon` matches as a domain label, not as a substring of one."""
    assert scope_for_request("https://amazon-deals-today.example/p/1") == ALL_PAGE_KINDS
    assert scope_for_request("https://notamazon.com/p/1") == ALL_PAGE_KINDS


@pytest.mark.parametrize("scope", _SCOPES)
def test_every_scope_keeps_the_other_escape(scope):
    """The invariant the whole design rests on."""
    assert "other" in scope
    assert len(scope) >= 2


# --- what the page declares about itself -----------------------------------


def signals(og_type=None, schema_types=(), asset_hosts=()):
    return {
        "og_type": og_type,
        "schema_types": list(schema_types),
        "asset_hosts": list(asset_hosts),
    }


def test_a_declared_menu_drops_the_item_branch():
    """The types that only a menu has, on a page calling itself an article.

    Which is not a contrived pairing: ``clara-junction-menu`` in the eval set is
    a real menu that declares ``og:type=article`` over a full Menu graph. A menu
    is therefore never ruled out by og:type — only ruled in by its own types.
    """
    declared = signals(
        og_type="article", schema_types=["Restaurant", "Menu", "MenuSection", "Offer"]
    )
    assert (
        scope_for_request("https://a-restaurant.test/food", page_signals=declared)
        == MENU_OR_OTHER
    )


def test_an_open_graph_restaurant_drops_the_item_branch():
    for og_type in ("restaurant.menu", "restaurant.restaurant"):
        assert (
            scope_for_request(
                "https://a-restaurant.test/", page_signals=signals(og_type)
            )
            == MENU_OR_OTHER
        )


@pytest.mark.parametrize(
    "declared",
    [
        signals(og_type="product", schema_types=[]),
        signals(og_type="product.item", schema_types=[]),
        signals(schema_types=["Product", "Offer", "BreadcrumbList"]),
        signals(schema_types=["http://schema.org/Product"]),
    ],
)
def test_a_declared_product_drops_the_menu_branch(declared):
    """Either half of the declaration is enough, which is why both are read.

    In the eval set ``thredup-wool-skirt`` carries Product JSON-LD and no
    og:type, and ``polyester_pants`` declares og:type=product and no Product.
    """
    assert (
        scope_for_request("https://a-shop.test/p/1", page_signals=declared)
        == ITEM_OR_OTHER
    )


def test_a_page_declaring_both_is_asked_the_whole_question():
    """A restaurant's own shop: a Menu graph in the footer, a Product on sale."""
    declared = signals(schema_types=["Restaurant", "Menu", "Product"])
    assert (
        scope_for_request("https://a-restaurant.test/gift-cards", page_signals=declared)
        == ALL_PAGE_KINDS
    )


@pytest.mark.parametrize(
    "schema_types",
    [
        # Injected site-wide, onto the shop page as readily as the menu page.
        ["Restaurant", "FoodEstablishment", "PostalAddress"],
        # A menu prices its dishes with these; they are not a shopping signal.
        ["Offer", "AggregateOffer"],
    ],
)
def test_types_that_are_not_evidence_decide_nothing(schema_types):
    assert (
        scope_for_request(
            "https://a-restaurant.test/",
            page_signals=signals(schema_types=schema_types),
        )
        == ALL_PAGE_KINDS
    )


# --- what platform the site runs on ----------------------------------------


@pytest.mark.parametrize(
    "hosts",
    [
        # agave-mexican-bistro: SpotHopper, and no schema of any kind.
        ["www.googletagmanager.com", "static.spotapps.co"],
        # clara-junction-menu, whose Resy widget says the same thing.
        ["fonts.gstatic.com", "widgets.resy.com", "images.getbento.com"],
        # udon-mugizo, ordering through DoorDash's white-label front.
        ["web-assets.cdn4dd.com", "beacon.riskified.com"],
    ],
)
def test_a_restaurant_platform_drops_the_item_branch(hosts):
    """The only signal most restaurants carry: what their site is built on."""
    assert (
        scope_for_request(
            "https://a-restaurant.test/food-menu",
            page_signals=signals(asset_hosts=hosts),
        )
        == MENU_OR_OTHER
    )


def test_a_product_on_a_restaurant_platform_keeps_the_item_branch():
    """The merch page carries the same hosts; only it declares a Product."""
    declared = signals(
        schema_types=["Product", "Offer"], asset_hosts=["cdn.getbento.com"]
    )
    assert (
        scope_for_request(
            "https://a-restaurant.test/shop/hot-sauce", page_signals=declared
        )
        == ITEM_OR_OTHER
    )


@pytest.mark.parametrize(
    "hosts",
    [
        # A generic CMS and CDN: eureka-restaurant runs on these, so do shops.
        ["images.prismic.io", "cdnjs.cloudflare.com", "connect.facebook.net"],
        # Shopify's, which say "a shop" — a claim this rule does not make.
        ["cdn.shopify.com", "monorail-edge.shopifysvc.com"],
        # Suffix matching is on labels, not on substrings of one.
        ["cdn.notgetbento.com", "resy.com.tracker.test"],
    ],
)
def test_hosts_that_are_not_a_restaurant_platform_decide_nothing(hosts):
    assert (
        scope_for_request(
            "https://a-page.test/x", page_signals=signals(asset_hosts=hosts)
        )
        == ALL_PAGE_KINDS
    )


def test_the_trigger_is_read_before_the_platform():
    """DoorDash is a menu platform and an add-to-cart button at once."""
    assert (
        scope_for_request(
            "https://www.doordash.com/store/1",
            trigger_type="automatic",
            page_signals=signals(asset_hosts=["web-assets.cdn4dd.com"]),
        )
        == ITEM_OR_OTHER
    )


@pytest.mark.parametrize(
    "declared",
    [
        None,
        "Menu",
        [],
        {},
        {"og_type": 7, "schema_types": "Menu", "asset_hosts": "getbento.com"},
        {"schema_types": [None, 3, "", "   "]},
        {"og_type": None, "schema_types": None},
    ],
)
def test_a_hint_that_makes_no_sense_changes_nothing(declared):
    """It is client-supplied and advisory, so it must not be able to fail a call."""
    assert (
        scope_for_request("https://a-restaurant.test/menu", page_signals=declared)
        == ALL_PAGE_KINDS
    )


def test_the_trigger_and_the_domain_are_read_before_the_declaration():
    """Both orderings matter, and for different reasons.

    An ordering platform serves a real Menu graph behind a real add-to-cart
    button; on that click the interruptive flow still wants the item branch. And
    a declaration cannot re-open a branch on a host the domain rule closed.
    """
    menu = signals(schema_types=["Menu", "MenuItem"])
    assert (
        scope_for_request(
            "https://order.example/store/1", trigger_type="automatic", page_signals=menu
        )
        == ITEM_OR_OTHER
    )
    assert (
        scope_for_request("https://www.ikea.com/restaurant/", page_signals=menu)
        == ITEM_OR_OTHER
    )


@pytest.mark.parametrize(
    "kwargs,rule",
    [
        ({"source": "google_maps_menu_tab"}, "menu_source"),
        ({"trigger_type": "automatic"}, "automatic_trigger"),
        ({"url": "https://www.amazon.com/dp/B0"}, "shopping_domain"),
        ({"page_signals": signals(schema_types=["MenuItem"])}, "declared_menu"),
        ({"page_signals": signals(og_type="product")}, "declared_item"),
        (
            {"page_signals": signals(asset_hosts=["static.spotapps.co"])},
            "menu_platform",
        ),
        ({}, "none"),
    ],
)
def test_a_decision_says_which_rule_made_it(kwargs, rule):
    """Production has to be able to show what each rule actually reached."""
    kwargs.setdefault("url", "https://a-page.test/x")
    assert decide_scope(**kwargs).rule == rule


# --- what a scope does to the prompt ---------------------------------------


def test_the_unused_branch_is_gone_from_the_instructions():
    item, _ = build_page_instructions(page_kinds=ITEM_OR_OTHER)
    menu, _ = build_page_instructions(page_kinds=MENU_OR_OTHER)

    assert "restaurant_menu" not in item
    assert "vegan_friendliness" not in item
    assert "shopping_item" not in menu
    assert "is_cruelty_free" not in menu

    # And the cross-branch bookkeeping goes with it.
    assert "leave the other branch's fields null" not in item
    assert "leave the other branch's fields null" not in menu


def test_the_item_scope_tells_the_model_where_menus_go():
    """Pruning removes the branch, so "other" has to be named as its home."""
    instructions, _ = build_page_instructions(page_kinds=ITEM_OR_OTHER)
    assert "restaurant menu" in instructions
    assert 'is "other" here' in instructions


@pytest.mark.parametrize(
    "builder", [build_openai_page_schema, build_gemini_page_schema]
)
def test_the_schema_carries_only_the_branch_in_scope(builder):
    item = builder(False, ITEM_OR_OTHER)
    menu = builder(False, MENU_OR_OTHER)

    assert item["properties"]["page_kind"]["enum"] == ["shopping_item", "other"]
    assert menu["properties"]["page_kind"]["enum"] == ["restaurant_menu", "other"]

    for field in ("items", "restaurant_name", "vegan_friendliness"):
        assert field not in item["properties"]
        assert field not in item["required"]
    for field in (
        "is_vegan",
        "materials_found",
        "is_cruelty_free",
        "confidence_level",
    ):
        assert field not in menu["properties"]
        assert field not in menu["required"]


def test_the_page_kind_description_names_only_what_the_enum_offers():
    schema = build_openai_page_schema(False, ITEM_OR_OTHER)
    assert "restaurant_menu" not in schema["properties"]["page_kind"]["description"]


def test_the_local_json_instruction_follows_the_schema():
    """The local path has no constrained decoding; this list is all it gets."""
    instruction = build_page_json_output_instruction(False, ITEM_OR_OTHER)
    schema = build_openai_page_schema(False, ITEM_OR_OTHER)

    for field in schema["required"]:
        assert f"- {field} (" in instruction
    assert "restaurant_name" not in instruction
    assert "inferred_ingredients" not in instruction


def test_a_narrowed_scope_shortens_the_prompt():
    """The point of the exercise: the freed tokens go to the page itself."""
    full, _ = build_page_instructions()
    item, _ = build_page_instructions(page_kinds=ITEM_OR_OTHER)
    menu, _ = build_page_instructions(page_kinds=MENU_OR_OTHER)

    assert len(item) < len(full)
    assert len(menu) < len(full)


def test_an_unknown_scope_is_refused():
    with pytest.raises(ValueError):
        build_page_instructions(page_kinds=("shopping_item",))


# --- what a scope does to the response -------------------------------------


def test_a_scoped_out_kind_is_not_honored():
    """A menu verdict from a schema with no menu fields would be an empty menu."""
    analysis = normalize_page_analysis(
        {"page_kind": "restaurant_menu", "summary": "s"},
        include_user_avoided=False,
        page_kinds=ITEM_OR_OTHER,
    )

    assert analysis["page_kind"] == "other"
    assert analysis["items"] == []
    assert analysis["restaurant_name"] is None


# Exactly what the released extension (manifest 1.0.4, which predates menus)
# reads off the top level of `analysis`. It knows nothing of `page_kind`, of
# menus, or of the nesting, so every one of these has to keep arriving flat
# whichever branch was pruned — a missing key here breaks every installed
# extension. See models.analysis_model.LEGACY_ITEM_ALIASES and
# LEGACY_PROSE_ALIAS (`explanation`, which is no longer asked for or stored and
# comes back as a copy of `summary`).
_LEGACY_1_0_4_KEYS = (
    "is_shopping_item",
    "is_vegan",
    "is_cruelty_free",
    "cruelty_free_explanation",
    "confidence_level",
    "summary",
    "explanation",
    "user_avoided_ingredients",
)


@pytest.mark.parametrize("scope", _SCOPES)
def test_the_released_extension_still_finds_what_it_reads(scope):
    """The compat guarantee: 1.0.4 reads these flat, whatever was pruned."""
    analysis = normalize_page_analysis(
        {"page_kind": "other", "summary": "s"},
        include_user_avoided=False,
        page_kinds=scope,
    )
    response = PageAnalysis.from_flat(analysis).to_response_dict()

    for field in _LEGACY_1_0_4_KEYS:
        assert field in response

    # `explanation` is gone from the model and the database; the response still
    # carries it, as a copy of the one field that survived.
    assert response["explanation"] == response["summary"] == "s"

    # And the nested shape newer code reads, with neither branch filled.
    assert response["page_kind"] == "other"
    assert response["shopping_item"] is None
    assert response["menu"] is None


def test_an_item_verdict_survives_the_item_scope():
    analysis = normalize_page_analysis(
        {
            "page_kind": "shopping_item",
            "is_vegan": False,
            "summary": "Leather wallet. Made of leather.",
            "confidence_level": "high",
        },
        include_user_avoided=False,
        page_kinds=ITEM_OR_OTHER,
    )

    assert analysis["page_kind"] == "shopping_item"
    assert analysis["is_vegan"] is False
    assert analysis["items"] == []

    # ...and reaches the extension under `shopping_item`, mirrored flat for 1.0.4.
    response = PageAnalysis.from_flat(analysis).to_response_dict()
    assert response["shopping_item"]["is_vegan"] is False
    assert response["shopping_item"]["confidence_level"] == "high"
    assert response["menu"] is None
    assert response["is_shopping_item"] is True
    assert response["is_vegan"] is False


def test_describe_scope_labels_each_scope_distinctly():
    assert len({describe_scope(scope) for scope in _SCOPES}) == len(_SCOPES)


def test_a_page_declaring_itself_narrows_the_scope_by_default(monkeypatch):
    """``PAGE_SIGNAL_SCOPE`` is on unless a deployment turns it off.

    Built through the real constructor, because the default is the thing under
    test; the providers are stubbed so nothing is configured or dialled.
    """
    monkeypatch.delenv("PAGE_SIGNAL_SCOPE", raising=False)
    monkeypatch.setattr(analysis_core, "DesktopService", lambda: object())
    monkeypatch.setattr(analysis_core, "GeminiService", lambda: object())

    core = AnalysisCore(enable_database=False)
    decision = core.scope_for(
        PageAnalysisRequest(
            url="https://nutsforcheese.com/products/dairy-free-black-garlic-cheese",
            content="Black Garlic — fermented cashew wedge.",
            timestamp="2026-08-22T21:00:00Z",
            page_signals={"og_type": "product", "schema_types": ["Product", "Offer"]},
        )
    )

    assert decision == (ITEM_OR_OTHER, "declared_item")
