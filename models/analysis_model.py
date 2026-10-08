"""The analysis payload as the extension and the database see it.

``page_prompt`` hands back one flat dict — the shape the *model* answers, whose
field order is load-bearing (see ``page_prompt._required_fields``). This module
turns that into the shape everything downstream reads: the fields that belong to
one page_kind grouped under ``shopping_item`` or ``menu``, so a menu record no
longer carries three null product columns and a product no longer carries an
empty dish list.

``PageAnalysis.from_flat`` is the only place that knows the flat -> nested
mapping. Everything else — the API response, ``APICallRecord``, the db_viewer —
works from the nested form.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

from services.analysis_prompt import CONFIDENCE_LEVELS
from services.menu_prompt import FRIENDLINESS_LEVELS, ITEM_VERDICTS
from services.page_scope import ALL_PAGE_KINDS

PageKind = Literal[ALL_PAGE_KINDS]  # type: ignore[valid-type]
ConfidenceLevel = Literal[tuple(CONFIDENCE_LEVELS)]  # type: ignore[valid-type]
ItemVerdict = Literal[tuple(ITEM_VERDICTS)]  # type: ignore[valid-type]
Friendliness = Literal[tuple(FRIENDLINESS_LEVELS)]  # type: ignore[valid-type]

# The fields the released extension (manifest 1.0.4, predating menus) reads off
# the top level of ``analysis``. Four of them are shared fields that live there
# anyway; the rest are mirrored out of ``shopping_item`` by ``to_response_dict``.
# When the manifest is bumped past 1.0.4 and that build is gone from the field,
# the mirrors below can go with it.
LEGACY_ITEM_ALIASES = (
    "is_vegan",
    "is_cruelty_free",
    "cruelty_free_explanation",
    "confidence_level",
)

# The prose field the analysis used to carry alongside ``summary``. The model is
# no longer asked for it and nothing stores it, but released extension builds
# still read it, so ``to_response_dict`` mirrors ``summary`` into it. It can go
# once no build in the field reads it.
LEGACY_PROSE_ALIAS = "explanation"


class MenuItem(BaseModel):
    """One dish, as the menu branch reported it."""

    name: str
    section: Optional[str] = None
    explicit_ingredients: list[str] = []
    inferred_ingredients: list[str] = []
    reason: str = ""
    verdict: ItemVerdict = "unclear"
    # Present only when the request carried an avoid-list.
    user_avoided_ingredients: Optional[list[str]] = None


class ShoppingItemAnalysis(BaseModel):
    """The product question: is *this item* vegan?"""

    # Named as on ``MenuItem``; the prompt calls them materials_found and
    # inferred_materials, since for clothing they are materials.
    explicit_ingredients: list[str] = []
    inferred_ingredients: list[str] = []
    # The entries of the two lists above that come from an animal.
    animal_derived_ingredients: list[str] = []
    is_vegan: Optional[bool] = None
    is_cruelty_free: Optional[bool] = None
    cruelty_free_explanation: Optional[str] = None
    confidence_level: ConfidenceLevel = "low"


class MenuAnalysis(BaseModel):
    """The menu question: which dishes on *this menu* are vegan?"""

    restaurant_name: Optional[str] = None
    vegan_friendliness: Friendliness = "none"
    # Already capped at menu_prompt.MAX_MENU_ITEMS upstream.
    items: list[MenuItem] = []


class PageAnalysis(BaseModel):
    """One page's analysis: what kind of page it is, and what that kind answers.

    Exactly one of ``shopping_item`` / ``menu`` is set, chosen by ``page_kind``;
    both are ``None`` for "other". ``extra="allow"`` is what lets
    ``to_response_dict``'s legacy aliases survive a round-trip through this
    model without being declared as fields of it.
    """

    model_config = ConfigDict(extra="allow")

    page_kind: PageKind = "other"
    summary: str = ""
    # The product answer; a menu reports its avoid-list hits per dish instead.
    user_avoided_ingredients: list[str] = []
    shopping_item: Optional[ShoppingItemAnalysis] = None
    menu: Optional[MenuAnalysis] = None

    @classmethod
    def from_flat(cls, analysis: dict[str, Any]) -> "PageAnalysis":
        """Group a normalized flat analysis by the page kind it answered.

        The branch that did not apply has already been forced null/empty by
        ``normalize_page_analysis``, so the kind alone decides which container
        is built.
        """
        page_kind = analysis.get("page_kind", "other")

        shopping_item = None
        if page_kind == "shopping_item":
            shopping_item = ShoppingItemAnalysis(
                explicit_ingredients=analysis.get("materials_found") or [],
                inferred_ingredients=analysis.get("inferred_materials") or [],
                animal_derived_ingredients=(
                    analysis.get("animal_derived_materials") or []
                ),
                is_vegan=analysis.get("is_vegan"),
                is_cruelty_free=analysis.get("is_cruelty_free"),
                cruelty_free_explanation=analysis.get("cruelty_free_explanation"),
                confidence_level=analysis.get("confidence_level") or "low",
            )

        menu = None
        if page_kind == "restaurant_menu":
            menu = MenuAnalysis(
                restaurant_name=analysis.get("restaurant_name"),
                vegan_friendliness=analysis.get("vegan_friendliness") or "none",
                items=[MenuItem(**item) for item in analysis.get("items") or []],
            )

        return cls(
            page_kind=page_kind,
            summary=analysis.get("summary", ""),
            user_avoided_ingredients=analysis.get("user_avoided_ingredients") or [],
            shopping_item=shopping_item,
            menu=menu,
        )

    def to_response_dict(self) -> dict[str, Any]:
        """The analysis as the API returns it: nested, plus the legacy aliases.

        ``is_shopping_item`` and the four ``LEGACY_ITEM_ALIASES`` are what
        extension 1.0.4 reads; it knows nothing of ``page_kind`` or of menus, so
        for anything but a product page they come back false/null, which is the
        answer that build already renders as "not a shopping page".

        ``explanation`` is a second copy of ``summary`` — see
        ``LEGACY_PROSE_ALIAS``.
        """
        payload = self.model_dump()
        payload["is_shopping_item"] = self.page_kind == "shopping_item"
        for alias in LEGACY_ITEM_ALIASES:
            payload[alias] = (
                getattr(self.shopping_item, alias) if self.shopping_item else None
            )
        payload[LEGACY_PROSE_ALIAS] = self.summary
        return payload
