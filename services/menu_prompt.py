"""Restaurant-menu branch of the page-analysis prompt.

The sibling ``analysis_prompt`` owns the product branch ("is *this item*
vegan?"); this one owns the menu branch ("which items on *this menu* are
vegan?") — one verdict per item plus a restaurant-level rating. ``page_prompt``
composes the two into the single prompt that actually runs.

As in ``analysis_prompt``, nothing here is a complete prompt or schema; the
module exports fragments (``build_menu_branch``, ``menu_properties``,
``normalize_menu_fields``) for ``page_prompt`` to assemble.

Menu text is parsed by the model rather than by the caller: menu markup varies
wildly across sources (a Google Maps place panel, a restaurant's own page), so
the caller passes bounded page text and the schema imposes the structure.

Drinks are kept out of the item list entirely. A verdict earns its row only
when the diner has a decision to make, and drinks rarely carry one: a soft
drink is vegan and saying so is noise, while a latte is a plant-milk swap that
nearly every counter will make but almost no menu prints — reported as
"not_vegan" it reads as a false alarm. The exclusion is deliberately blunt, so
it also drops the drink that would have been worth a warning (a milkshake, a
sour made with egg white) and leaves a drinks-only page — a cafe, a bar — with
no items at all.
"""


# The model is asked to list every item it finds, up to this many. The cap is
# what keeps a long menu with a per-item reason for every entry inside the
# output-token budget of the local model (``LMS_MAX_TOKENS``, 8192 by default).
MAX_MENU_ITEMS = 60

# Per-item verdicts. "unclear" has to be a first-class outcome, otherwise an
# item named "Chef's Special" gets a guessed verdict; "veganizable" likewise,
# for the item the menu itself offers in a vegan version (choice of beef,
# chicken or tofu) — neither vegan nor simply not.
ITEM_VERDICTS = ["vegan", "likely_vegan", "veganizable", "not_vegan", "unclear"]

# Restaurant-level rating of how well a vegan can eat here.
FRIENDLINESS_LEVELS = ["high", "medium", "low", "none"]


_ITEM_FIELD_DESCRIPTIONS = {
    "name": "The item name exactly as it appears on the menu",
    "section": "The menu section the item appears under (e.g. Starters, Mains), or null if the menu is not divided into sections",
    "explicit_ingredients": "Ingredients the menu itself names for this item, in its title or its description (empty if it names none)",
    "inferred_ingredients": "Ingredients the item contains by convention but the menu does not name, e.g. the beef patty in a burger (empty if the item is too vaguely described to infer any)",
    "reason": "One short sentence stating the item's vegan status and why. Name the specific animal-derived ingredient when there is one. Do not restate the item name.",
    "verdict": "Vegan status of the item as listed, before any modification. Must match the status already given in the reason.",
    "user_avoided_ingredients": "Ingredients from the user's avoid-list found in this item",
}

# Per-item field order, and it matters for the same reason the top-level order
# does (see ``page_prompt._REQUIRED_FIELDS``): each field is answered in the
# order listed, so a verdict emitted before its justification is a guess. The
# item is identified, its ingredients are named — what the menu states, then
# what the dish carries by convention — the reason is written, and only then is
# the verdict committed.
_ITEM_REQUIRED_FIELDS = [
    "name",
    "section",
    "explicit_ingredients",
    "inferred_ingredients",
    "reason",
    "verdict",
]

FIELD_DESCRIPTIONS = {
    "restaurant_name": "Name of the restaurant, or null unless page_kind is restaurant_menu",
    "items": "One entry per item found on the menu (empty unless page_kind is restaurant_menu)",
    "vegan_friendliness": 'How well a vegan diner can eat at this restaurant, based on the menu as a whole. Use "none" unless page_kind is restaurant_menu.',
}

# The branch's top-level fields, in order.
MENU_FIELDS = [
    "restaurant_name",
    "items",
    "vegan_friendliness",
]


def includes_user_avoided(user_avoided_ingredients: list[str] | None) -> bool:
    """Whether the per-item user-avoided-ingredients field is part of the schema.

    The rule lives here so callers that build the schema and callers that
    normalize a response agree on it.
    """
    return bool(user_avoided_ingredients)


def build_menu_branch(user_avoided_ingredients: list[str] | None = None) -> str:
    """Build the instruction text for the restaurant-menu branch.

    The caller (``page_prompt.build_page_instructions``) has already told the
    model how to choose a ``page_kind``; this text applies once it has chosen
    ``restaurant_menu``.
    """
    user_avoided_ingredients_instruction = ""
    if includes_user_avoided(user_avoided_ingredients):
        user_avoided_ingredients_instruction = (
            "\n\nUSER AVOID-LIST:\n"
            "- The user also wants to avoid these ingredients: "
            + ", ".join(user_avoided_ingredients)
            + "\n- For each item, list any of them you find in that item's "
            "user_avoided_ingredients field\n"
            "- These are in addition to, not instead of, the animal-derived "
            "ingredients above. An item containing only avoid-list ingredients "
            "is still vegan; note the ingredient without changing the verdict."
        )

    return f"""IF page_kind IS "restaurant_menu" — extract every item and classify each one:
- List every food item the menu holds, from the first section to the last. Work through the whole menu: never stop part-way, and never shorten the list to save effort or time. A menu reported in part is wrong, not brief. Drinks are the one exception: see the DRINKS rule below.
- {MAX_MENU_ITEMS} items is a ceiling, not a target. Fewer than that is correct only when the menu itself holds fewer. When the menu holds more, keep the {MAX_MENU_ITEMS} most useful to a vegan diner: plant-based and ambiguous items first, obviously meat-based items last.
- Use the item name exactly as written; do not translate, rename, or tidy it
- Record the section heading each item appears under, or null if the menu has no sections
- Include an item once. Skip duplicate listings of the same item.
- Do not include non-food entries such as section headings on their own, opening hours, or delivery information
- Set restaurant_name from the page, or null if it cannot be determined

DRINKS — never list a drink as an item:
- Skip everything drinkable, whatever its ingredients: soft drinks, water, juice, coffee, tea, beer, wine, spirits, cocktails, smoothies, milkshakes and bubble tea. Skip them whether they are vegan, are not, or could be made vegan.
- This holds even when a whole section of the menu is drinks, and even when the menu advertises a vegan version of a drink.
- The food items are the analysis. Judge vegan_friendliness on them alone.

FULLY VEGAN RESTAURANTS — settle this before you classify any item:
- Does the page establish that this restaurant serves only vegan food? Its name, its description, or vegan labelling running through the whole menu are each enough on their own.
- "Vegetarian" is not "vegan". At a vegetarian restaurant the cheese, egg, butter and yoghurt are the real thing, and the rules below apply unchanged.
- At a fully vegan restaurant the meat, fish and dairy words on the menu name the imitation, not the ingredient: "Fish & Chips" is battered plant protein, "Drumsticks" are seitan, "cream cheese" is the plant-based version. Quotation marks around such a word — Fried "Ocean" Fillet — are the menu saying exactly that.
- So on such a menu, never infer an animal-derived ingredient into an item. Record the imitation ingredient as the menu writes it, give the item the verdict "vegan", and say in the reason that the restaurant is fully vegan.

For each item, answer in this order — the ingredients you can point to, then the reason, then the verdict:

EXPLICIT_INGREDIENTS first: list the ingredients the menu itself names for this item, whether in its title or in its description ("Halloumi fries with mint yoghurt" → halloumi, mint, yoghurt). Leave it empty when the menu gives nothing but a name.

INFERRED_INGREDIENTS next: list what the item contains by convention, which the menu leaves unsaid — a burger has a beef patty and a wheat bun unless it says otherwise, a carbonara has egg and cured pork, a croissant has butter. Do not repeat anything already in explicit_ingredients. Leave it empty when the name and description give no reliable basis to infer anything.
- Infer what the item normally is, not everything it could be. Never invent an ingredient that is neither stated nor conventional.
- Watch for the animal-derived ingredients that read as vegetarian: cheese, cream, butter, yoghurt, egg, mayonnaise and honey are all animal-derived
- Fish and other sea animals are meat: fish, shrimp, prawns, squid, calamari, octopus, scallops, clams, mussels, oysters, crab, lobster, anchovies, roe, and the sauces and broths made from them are all animal-derived, and an item holding any of them is not vegan
- Judge the item exactly as listed. An animal-derived ingredient still counts when it is served on the side, listed as a topping or garnish, or could obviously be left out — record it, and say in the reason that it could be left out

REASON next: one short sentence naming the item's vegan status and why. When either list holds an animal-derived ingredient, name the one responsible. When neither does, say whether the item is plant-based as described or simply too vague to tell. Do not restate the item name.

VERDICT last (based on the item as listed, before any modification), and it must match the status you just gave in the reason:
- "vegan": neither ingredient list holds an animal-derived ingredient, and the item is either explicitly labelled vegan or composed only of plant ingredients
- "likely_vegan": neither ingredient list holds an animal-derived ingredient and the item is plant-based as described, but a common preparation could include an animal product that is not stated (e.g. pasta that may contain egg, bread that may be brushed with butter, soup that may use meat stock)
- "veganizable": the menu offers this item in several versions and at least one of them is vegan — a bowl with a choice of beef, chicken or tofu, a burger the menu offers with a plant-based patty. Only for a choice the menu itself offers; an item that would merely survive having its cheese left off is "not_vegan".
- "not_vegan": an animal-derived ingredient is in one of the lists and the menu offers no vegan version. If you named an animal-derived ingredient, the verdict is "not_vegan" or "veganizable" — never "likely_vegan" or "unclear".
- "unclear": the ingredient lists are empty only because the name and description give no reliable basis to judge (e.g. "Chef's Special", "Soup of the Day")
- Respect explicit menu labelling (V, VG, vegan, plant-based) over your own inference, but note in the reason when a label is ambiguous — "V" often means vegetarian rather than vegan

RESTAURANT-LEVEL FIELDS:
- vegan_friendliness: "high" if there are several clearly vegan items across the menu, "medium" if there are a few vegan or veganizable items, "low" if only one or two marginal options exist, "none" if a vegan could not eat a meal here
- summary: one to two sentences a diner can act on — roughly how many vegan options there are and where on the menu they are. The per-item detail belongs in each item's reason, not here. Never leave it empty or use placeholder text.{user_avoided_ingredients_instruction}"""


def menu_item_properties(include_user_avoided: bool, nullable_flavor: bool) -> dict:
    """Build the per-item object properties in one of the two schema flavors.

    ``nullable_flavor`` selects Gemini's ``nullable: True`` over the OpenAI
    ``type``-union spelling for the optional fields.
    """

    def optional(type_name: str, description: str) -> dict:
        if nullable_flavor:
            return {"type": type_name, "nullable": True, "description": description}
        return {"type": [type_name, "null"], "description": description}

    def string_list(field: str) -> dict:
        return {
            "type": "array",
            "items": {"type": "string"},
            "description": _ITEM_FIELD_DESCRIPTIONS[field],
        }

    properties = {
        "name": {
            "type": "string",
            "description": _ITEM_FIELD_DESCRIPTIONS["name"],
        },
        "section": optional("string", _ITEM_FIELD_DESCRIPTIONS["section"]),
        "verdict": {
            "type": "string",
            "enum": list(ITEM_VERDICTS),
            "description": _ITEM_FIELD_DESCRIPTIONS["verdict"],
        },
        "reason": {
            "type": "string",
            "description": _ITEM_FIELD_DESCRIPTIONS["reason"],
        },
        "explicit_ingredients": string_list("explicit_ingredients"),
        "inferred_ingredients": string_list("inferred_ingredients"),
    }

    if include_user_avoided:
        properties["user_avoided_ingredients"] = string_list(
            "user_avoided_ingredients"
        )

    return properties


def menu_item_required(include_user_avoided: bool) -> list[str]:
    required = list(_ITEM_REQUIRED_FIELDS)
    if include_user_avoided:
        required.append("user_avoided_ingredients")
    return required


def menu_properties(include_user_avoided: bool, nullable_flavor: bool) -> dict:
    """Build the restaurant-menu schema properties in one of the two flavors."""

    def optional(type_name: str, description: str) -> dict:
        if nullable_flavor:
            return {"type": type_name, "nullable": True, "description": description}
        return {"type": [type_name, "null"], "description": description}

    item_schema = {
        "type": "object",
        "properties": menu_item_properties(include_user_avoided, nullable_flavor),
        "required": menu_item_required(include_user_avoided),
    }
    if not nullable_flavor:
        item_schema["additionalProperties"] = False

    return {
        "restaurant_name": optional("string", FIELD_DESCRIPTIONS["restaurant_name"]),
        "items": {
            "type": "array",
            "description": FIELD_DESCRIPTIONS["items"],
            "items": item_schema,
        },
        # Not nullable, unlike the other cross-branch fields: an enum combined
        # with `nullable` is the shakiest corner of Gemini's schema dialect, so
        # the model always picks a level ("none" when there is nothing to eat)
        # and `normalize_menu_fields` nulls it for non-menu pages instead.
        "vegan_friendliness": {
            "type": "string",
            "enum": list(FRIENDLINESS_LEVELS),
            "description": FIELD_DESCRIPTIONS["vegan_friendliness"],
        },
    }


def normalize_menu_fields(
    analysis: dict, applies: bool, include_user_avoided: bool
) -> None:
    """Default the menu fields in place, so both providers return one shape.

    A model that omits ``section`` or an ingredient list on an item, or drops
    ``items`` entirely, would otherwise reach the extension as a
    differently-shaped object depending on which provider served the request.

    ``applies`` is False for every page_kind other than ``restaurant_menu``, in
    which case the fields are forced empty rather than merely defaulted: a
    product page that came back carrying invented items would otherwise render
    as a menu.
    """
    if not applies:
        analysis["restaurant_name"] = None
        analysis["items"] = []
        analysis["vegan_friendliness"] = None
        return

    analysis.setdefault("restaurant_name", None)
    analysis.setdefault("vegan_friendliness", "none")

    items = analysis.get("items")
    if not isinstance(items, list):
        items = []

    normalized_items = []
    for item in items[:MAX_MENU_ITEMS]:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        item.setdefault("section", None)
        item.setdefault("verdict", "unclear")
        item.setdefault("reason", "")
        item.setdefault("explicit_ingredients", [])
        item.setdefault("inferred_ingredients", [])
        if include_user_avoided:
            item.setdefault("user_avoided_ingredients", [])
        normalized_items.append(item)

    analysis["items"] = normalized_items
