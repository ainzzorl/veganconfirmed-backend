"""Shopping-item branch of the page-analysis prompt.

A single analysis answers "what kind of page is this, and what should a vegan
know about it?" — see ``page_prompt``, which composes the whole prompt. This
module owns one branch of it: the product question ("is *this item* vegan?").
The sibling ``menu_prompt`` owns the other ("which dishes on *this menu* are
vegan?").

Nothing here is a complete prompt or a complete schema. The module exports
*fragments* that ``page_prompt`` assembles:

- ``build_item_branch`` — the instruction text for classifying a product
- ``item_properties`` — the branch's response-schema properties, in either of
  the two flavors we have to speak (Gemini expresses optional fields with
  ``nullable``, the OpenAI-compatible endpoint used by LM Studio with ``type``
  unions)
- ``ITEM_EVIDENCE_FIELDS`` / ``ITEM_VERDICT_FIELDS`` — the branch's fields, split
  by where they belong in the response: evidence is gathered before the shared
  prose, verdicts are committed after it (see ``page_prompt._REQUIRED_FIELDS``)
- ``ITEM_FIELDS`` / ``normalize_item_fields`` — the full field list and the
  defaults

Keeping the branches in their own modules keeps each one readable; splicing
them together in one place keeps the model from being asked two questions in
two voices.
"""


# Field descriptions shared by both schema flavors.
FIELD_DESCRIPTIONS = {
    "materials_found": "The materials and ingredients the page states this product is made of, in the page's own words (e.g. \"leather\", \"merino wool\", \"organic cotton\", \"rubber\"). Empty unless page_kind is shopping_item, or when the page states none at all.",
    "inferred_materials": "Ingredients and materials the product contains by convention but the page does not state, e.g. the pork in a sausage with no ingredient list. Never repeats an entry of materials_found. Empty when the page states the full composition, or when the product is too vaguely described to infer any.",
    "animal_derived_materials": "The entries of materials_found and inferred_materials that come from an animal (leather, suede, wool, silk, down, beeswax, gelatin, dairy, egg, meat, fish, and the like). Empty when none of them does.",
    "is_vegan": "Whether the product is vegan: false when animal_derived_materials is non-empty, true when it is empty (null unless page_kind is shopping_item). Must agree with the verdict already given in the summary.",
    "is_cruelty_free": "Whether the product is cruelty-free/not tested on animals (null if not applicable, e.g. food, clothing, electronics)",
    "cruelty_free_explanation": "Brief explanation of cruelty-free findings. Null unless the product is a cosmetic, personal-care or cleaning product — food, clothing, footwear and electronics get null here, not a sentence.",
    "user_avoided_ingredients": "List of user-specified ingredients found in the product",
    "confidence_level": "Confidence in the vegan and cruelty-free verdicts above (null unless page_kind is shopping_item)",
}

# How sure the product verdict is. Only the shopping-item branch has one: a menu
# says what it is unsure of per dish, through the "unclear" verdict and
# vegan_friendliness, so a single level judging the whole page would say nothing
# there.
CONFIDENCE_LEVELS = ["high", "medium", "low"]

# Evidence the model gathers *before* it commits to anything, answered first in
# the response so the verdicts below are written after the page has been read
# rather than in place of reading it.
#
# Two steps, not one, because gathering alone was not enough: given a
# materials_found listing "Leather" four times, flash-lite still wrote "vegan as
# it is made from leather and synthetic materials". Filtering the list into
# animal_derived_materials turns the verdict from a judgement into a lookup —
# is_vegan is false exactly when that list is non-empty. It mirrors the menu
# branch, where naming an item's ingredients ahead of its verdict took verdict
# accuracy from 83% to 100%.
#
# ``inferred_materials`` sits between the two for the same reason the menu
# branch's inferred_ingredients does: a product page that names no ingredients
# still has some, and a sausage with no ingredient list is not vegan by default.
#
# All three are returned to the extension (see ``ShoppingItemAnalysis``), which
# lists the first two and highlights the entries of the third.
ITEM_EVIDENCE_FIELDS = [
    "materials_found",
    "inferred_materials",
    "animal_derived_materials",
]

# The verdicts, answered last. Within the pair, the cruelty-free reasoning
# precedes its boolean for the same reason ``summary`` precedes ``is_vegan``:
# a verdict token emitted before its justification is a guess.
# ``confidence_level`` closes the list because it judges everything above it.
ITEM_VERDICT_FIELDS = [
    "is_vegan",
    "cruelty_free_explanation",
    "is_cruelty_free",
    "confidence_level",
]

# The branch's fields. ``summary`` is deliberately *not* here: it is filled for
# every page_kind (a menu describes itself through per-dish reasons, but an
# "other" page still owes the user a sentence), so page_prompt owns it.
ITEM_FIELDS = ITEM_EVIDENCE_FIELDS + ITEM_VERDICT_FIELDS


def build_item_branch(user_avoided_ingredients: list[str] | None = None) -> str:
    """Build the instruction text for the shopping-item branch.

    The caller (``page_prompt.build_page_instructions``) has already told the
    model how to choose a ``page_kind``; this text applies once it has chosen
    ``shopping_item``.
    """
    user_avoided_ingredients_instruction = ""
    if user_avoided_ingredients:
        user_avoided_ingredients_instruction = (
            "\n- Also check whether the product contains any of these "
            "ingredients the user wants to avoid: "
            + ", ".join(user_avoided_ingredients)
            + "\n  Include any you find in the top-level "
            "user_avoided_ingredients field."
        )

    return f"""IF page_kind IS "shopping_item" — analyze the product for vegan status:

FIRST, fill in materials_found — the materials and ingredients the page states the product is made of, in the page's own words:
- List them all, plant and synthetic ones (cotton, polyester, rubber, plastic) as well as animal-derived ones. Do not filter or judge yet.
- Where to look, by product type:
  - Food/beverage: the ingredient list, product description, dietary claims and certifications
  - Clothing/textiles: the fabric or material composition — watch for leather, wool, merino, cashmere, silk, fur, down, feathers, suede, shearling, animal-based dyes
  - Cosmetics/beauty: the ingredient list — watch for beeswax, lanolin, carmine, collagen, keratin, and other animal-derived ingredients
  - Household goods: the material composition — watch for leather, wool, down, animal-based glues
- Leave materials_found empty only when the page states no materials or ingredients at all.

NEXT, fill in inferred_materials — what the product contains by convention that the page leaves unsaid:
- A chicken sausage with no ingredient list has chicken and usually a casing; a milk chocolate bar has milk; a croissant has butter.
- Do not repeat anything already in materials_found. Leave it empty when the page states the product's full composition.
- Infer what the product normally is, not everything it could be. Never invent an ingredient or material that is neither stated nor conventional, and never infer one that contradicts what the page states (a "vegan" or "plant-based" product does not conventionally contain meat or dairy).

THEN, go through materials_found and inferred_materials entry by entry and copy the animal-derived ones into animal_derived_materials:
- These come from an animal wherever they appear, in a material list as much as in an ingredient list: leather, suede, nubuck, shearling, fur, wool, merino, cashmere, alpaca, mohair, angora, silk, down, feathers, horn, bone, pearl, mother-of-pearl, beeswax, lanolin, carmine, collagen, keratin, gelatin, honey, milk, dairy, cheese, egg, and any meat or fish
- These do not: "vegan leather", "faux leather", "faux fur", "synthetic suede", "plant-based leather", and any material the page names as synthetic or plant-derived
- Leave animal_derived_materials empty when none of the materials came from an animal

THEN read is_vegan off that list — there is no judgement left at this step:
- is_vegan is false when animal_derived_materials is non-empty, and true when it is empty
- The summary must say the same thing. If animal_derived_materials contains leather, the summary says the product is not
  vegan because it is made with leather; it may not call a product vegan while naming an animal-derived material it is made of.
- Only when materials_found and inferred_materials both came out empty — nothing is stated or conventionally known about what the
  product is made of — default to is_vegan=true, set confidence_level to medium or low, and note in the summary that the composition
  was not stated. A material stated on the page always outweighs this default: never apply it to a product whose materials are listed.
- The summary should be a complete one-to-two sentence plain-language description of the product and its vegan status, concise and
  to the point. Open directly with the vegan (and cruelty-free, if applicable) verdict and the reason for it. Do not begin with
  preamble about the page itself (e.g. "The page describes...", "The page lists a product..."); go straight to the reasoning behind
  the status. Never leave it empty or use placeholder text like "..." or "analysis".{user_avoided_ingredients_instruction}

CRUELTY-FREE ANALYSIS — check first whether it applies at all:
- It applies to cosmetics, skincare, makeup, personal care (soap, body wash, shampoo, deodorant, toothpaste and the like), hair care, body care, household cleaners, cleaning products and laundry products, and to nothing else.
- For every other product — food, beverages, clothing, textiles, footwear, electronics, furniture, accessories — set cruelty_free_explanation to null and is_cruelty_free to null, and write nothing about animal testing anywhere in the response. Being vegan does not make a product a cruelty-free question: a plant-based sausage still gets null.
- Base an applicable verdict only on what the page says. Never infer a brand's animal-testing policy from what you know of the brand in general.
- For applicable products, search the whole page, not just the ingredient list — the description, feature bullets, product attributes and brand text are where these statements usually are. Look for:
  - Cruelty-free certifications: Leaping Bunny, PETA cruelty-free, Choose Cruelty Free (CCF), Cruelty Free International
  - Brand statements about animal testing policies
  - "Not tested on animals" claims
  - Parent company animal testing policies (some cruelty-free brands are owned by companies that test on animals)
- Set is_cruelty_free to true if there is evidence the product/brand is cruelty-free
- Set is_cruelty_free to false if there is evidence the product/brand tests on animals or sells in markets requiring animal testing (e.g. mainland China for cosmetics)
- If no cruelty-free information is available for an applicable product, set is_cruelty_free to null and explain in cruelty_free_explanation
- Provide a brief cruelty_free_explanation for applicable products

FINALLY, set confidence_level — it judges everything you have written above:
- "high" when the page states the ingredients, materials or a vegan label
- "medium" when the product is recognizable but its composition is not stated, so the verdict rests on inferred_materials
- "low" when the content is partial, poorly described, or mostly unclear"""


def item_properties(nullable_flavor: bool) -> dict:
    """Build the shopping-item schema properties in one of the two flavors.

    ``nullable_flavor`` selects Gemini's ``nullable: True`` over the OpenAI
    ``type``-union spelling for the optional fields.
    """

    def optional(type_name: str, description: str) -> dict:
        if nullable_flavor:
            return {"type": type_name, "nullable": True, "description": description}
        return {"type": [type_name, "null"], "description": description}

    return {
        # Not nullable: an empty list is the "nothing stated" answer, and it
        # has to be cheap for the model to write, since every non-product page
        # writes it.
        "materials_found": {
            "type": "array",
            "items": {"type": "string"},
            "description": FIELD_DESCRIPTIONS["materials_found"],
        },
        "inferred_materials": {
            "type": "array",
            "items": {"type": "string"},
            "description": FIELD_DESCRIPTIONS["inferred_materials"],
        },
        "animal_derived_materials": {
            "type": "array",
            "items": {"type": "string"},
            "description": FIELD_DESCRIPTIONS["animal_derived_materials"],
        },
        "is_vegan": optional("boolean", FIELD_DESCRIPTIONS["is_vegan"]),
        "is_cruelty_free": optional("boolean", FIELD_DESCRIPTIONS["is_cruelty_free"]),
        "cruelty_free_explanation": optional(
            "string", FIELD_DESCRIPTIONS["cruelty_free_explanation"]
        ),
        # Not nullable, unlike the other optional fields: an enum combined with
        # `nullable` is the shakiest corner of Gemini's schema dialect (see
        # menu_prompt.vegan_friendliness), so the model always picks a level and
        # ``normalize_item_fields`` nulls it for non-product pages instead.
        "confidence_level": {
            "type": "string",
            "enum": list(CONFIDENCE_LEVELS),
            "description": FIELD_DESCRIPTIONS["confidence_level"],
        },
    }


def normalize_item_fields(analysis: dict, applies: bool) -> None:
    """Default the item fields in place.

    ``applies`` is False for every page_kind other than ``shopping_item``, in
    which case the fields are forced to null rather than merely defaulted: a
    menu that came back carrying ``is_vegan: true`` would otherwise reach the
    extension looking like a product verdict.

    """
    if not applies:
        for field in ITEM_EVIDENCE_FIELDS:
            analysis[field] = []
        for field in ITEM_VERDICT_FIELDS:
            analysis[field] = None
        return

    for field in ITEM_EVIDENCE_FIELDS:
        if not isinstance(analysis.get(field), list):
            analysis[field] = []
    for field in ITEM_VERDICT_FIELDS:
        analysis.setdefault(field, None)

    # Unlike the other verdicts, an absent or unrecognized confidence is not
    # "unknown": the schema offers three levels, and the least of them is the
    # honest answer for an analysis that declined to say.
    if analysis["confidence_level"] not in CONFIDENCE_LEVELS:
        analysis["confidence_level"] = "low"
