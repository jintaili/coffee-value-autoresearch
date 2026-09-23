"""One A+B catalog for rating and price. Question IDs are stable feature IDs."""

from __future__ import annotations

import hashlib
import json

from coffee_value.features import COUNTRY_ALIASES

CONTRACT_VERSION = "coffee-ab-1"
MODEL = "jev-1.13.0"

COMMON = (
    "Judge this lot only. Require direct evidence or a clear synonym. Ignore other products, stereotypes, reputation, and price. "
    "Conflicting means incompatible source claims, not model uncertainty. "
)
EVIDENCE = {
    "supported": "This exact claim is directly stated or clearly paraphrased for this lot.",
    "explicitly_negated": "This exact claim is directly denied with not, without, or excludes. Another process or variety is not denial.",
    "not_stated": "The claim is neither stated nor directly denied, even if another method is stated.",
    "conflicting": "Incompatible direct claims about this lot.",
}
DEGREES = {
    "acidity": ("soft, muted, or low acidity", "moderate acidity", "pronounced, intense, or high acidity"),
    "sweetness": ("low perceived sweetness", "moderate perceived sweetness", "pronounced or intense sweetness"),
    "body": ("light, thin, or delicate body", "medium body", "full, heavy, thick, viscous, or syrupy body"),
}


def evidence(claim: str, boundary: str = "") -> dict:
    return {"type": "choice", "instructions": COMMON + f"Does `state` directly establish {claim}? Another method or label is not a denial. " + boundary, "criteria": EVIDENCE}


def category(question: str, options: dict[str, str | None]) -> dict:
    return {"type": "choice", "instructions": COMMON + question,
            "criteria": {**options, "other": "A stated answer outside these named categories.",
                         "not_stated": "No answer stated for this coffee lot.",
                         "conflicting": "Incompatible answers stated for this coffee lot."}}


def degree(name: str, boundary: str) -> dict:
    low, medium, high = DEGREES[name]
    return {"type": "choice", "instructions": COMMON + f"What degree of {name} is explicitly described? {boundary}", "criteria": {
        "low": f"Explicitly described as {low}.", "medium": f"Explicitly described as {medium}.",
        "high": f"Explicitly described as {high}.", "not_stated": f"No explicit degree of {name} is described.",
        "conflicting": f"Incompatible degrees of {name} are explicitly described."}}


COUNTRIES = sorted(set(COUNTRY_ALIASES.values()) | {"United States", "United Kingdom"})
ROASTER_COUNTRIES = sorted(set(COUNTRIES) | {"Australia", "Canada", "Japan", "South Korea", "New Zealand",
    "Germany", "France", "Italy", "Spain", "Netherlands", "Belgium", "Switzerland", "Singapore",
    "Hong Kong", "United Arab Emirates", "Czech Republic", "Malaysia", "Poland", "Sweden", "Norway",
    "Denmark", "Ireland", "South Africa", "Chile", "Argentina"})
QUESTIONS: dict[str, dict] = {
    "origin_country": category("Which country grows the coffee? Java as an island means Indonesia; Java as a cultivar is not origin. Select multiple_countries for a multi-country blend.",
        {**{c: None for c in COUNTRIES}, "multiple_countries": "Components are explicitly grown in several countries."}),
    "roaster_country": category("In which country is the roaster located? This is distinct from coffee origin.",
        {c: None for c in ROASTER_COUNTRIES}),
    "blend": evidence("that this coffee is a blend of coffees", "A coffee name containing blend counts only when it names this product."),
    "decaf": evidence("that this coffee is decaffeinated"),
    "producer_identified": evidence("a named farm, producer, estate, mill, or cooperative associated with this lot", "Generic claims about farmers do not count."),
    "altitude_provided": evidence("a growing altitude with a numeric value", "Keep valid altitude numbers; do not infer altitude from region."),
    "brewing_intent": category("What is the primary stated brewing intent? A tasting reference to espresso does not by itself make an espresso product.", {
        "espresso_primary": "Presented mainly for espresso.", "filter_primary": "Presented mainly for filter or drip brewing.",
        "both": "Presented for both espresso and filter.", "other_brew": "Another brewing method is primary."}),
}

for name, aliases in {
    "washed": "washed or wet processing", "natural": "natural or dry processing",
    "honey": "honey or pulped-natural processing", "anaerobic": "anaerobic fermentation or processing",
    "carbonic_maceration": "carbonic maceration", "wet_hulled": "wet-hulled processing",
    "lactic": "lactic fermentation or processing",
}.items():
    boundary = ("Honey flavor is not honey processing. Natural sweetness is not natural processing. "
                "Wet-hulled is not washed; anaerobic does not imply natural. "
                "Lactic needs lactic or lactic-acid-bacteria, not generic fermentation. "
                "A stated washed method does not directly deny natural processing.")
    QUESTIONS[f"process_{name}"] = evidence(aliases, boundary)

for name in ("gesha", "pink_bourbon", "bourbon", "typica", "caturra", "catuai", "sl28", "sl34", "pacamara", "maragogipe", "mokka", "mokkita", "ruiru", "castillo", "java"):
    boundary = "Geisha and Gesha are aliases. SL-28 and SL28 are aliases, as are SL-34 and SL34. Java as geography is not Java cultivar. Pink Bourbon is separate from unspecified Bourbon."
    claim = "unspecified Bourbon (not Pink Bourbon)" if name == "bourbon" else name.replace("_", " ")
    QUESTIONS[f"variety_{name}"] = evidence(f"that {claim} is a cultivar or variety in this lot", boundary)

for name, boundary in {
    "acidity": "Fruit names or 'balanced' alone do not establish acidity intensity.",
    "sweetness": "A honey note does not establish sweetness intensity or honey processing.",
    "body": "Viscous or syrupy mouthfeel indicates full body. Smooth, buttery, or silky alone describes texture, not body degree. Do not infer body from roast or brew recommendation.",
}.items():
    QUESTIONS[f"degree_{name}"] = degree(name, boundary)

for name, claim in {
    "floral": "floral aroma or flavor", "citrus": "citrus aroma or flavor",
    "non_citrus_fruit": "fruit aroma or flavor other than citrus", "chocolate": "chocolate or cocoa aroma or flavor",
    "nut": "nutty aroma or flavor", "roast_smoke": "roast-derived or smoky aroma or flavor",
}.items():
    QUESTIONS[f"sensory_{name}"] = evidence(claim, "Judge the tasting description, not the processing method.")


def question_hash() -> str:
    return hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
