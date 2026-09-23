"""Small, source-grounded JEV extension for finish and lot-specific status.

This is an incremental request over the frozen A+B catalog. The original answers
remain reusable, while this catalog has its own cache namespace and version.
"""

from __future__ import annotations

import hashlib
import json

from .questions import MODEL, category, evidence

CONTRACT_VERSION = "coffee-finish-lot-3"

QUESTIONS: dict[str, dict] = {
    "finish_length": category(
        "What finish duration is explicitly described for this coffee? Judge the aftertaste, "
        "not the brewing time or how long a flavor appears in the cup.",
        {
            "long": "The finish or aftertaste is explicitly long, lingering, lasting, or persistent.",
            "short": "The finish or aftertaste is explicitly short, brief, abrupt, or fades quickly.",
        },
    ),
    "finish_sweet_clean": evidence(
        "that this exact coffee's finish or aftertaste is sweet, sugary, or clean",
        "Require sweet, sugary, or clean to describe the finish itself, or a flavor explicitly "
        "said to carry into a sweet or clean finish. Sweetness elsewhere in the cup or bottom line, "
        "honey or flowers in the finish, and a long or pleasant finish do not count.",
    ),
    "finish_bitter_drying": evidence(
        "that this exact coffee's finish or aftertaste is bitter, astringent, or drying",
        "Require bitter, astringent, dry, or drying to describe the finish itself. "
        "Sharp, carbon, dark chocolate, or roast notes alone do not count. "
        "A positive flavor elsewhere does not deny a drying finish.",
    ),
    "sensory_balance": category(
        "Is this coffee's cup or a named sensory component explicitly described as balanced or unbalanced? "
        "Balanced acidity counts; a mere list of flavors or a numeric review component score does not.",
        {
            "balanced": "The cup or a named component such as acidity, sweetness, or body is explicitly described as balanced or harmonious.",
            "unbalanced": "The cup or a named component is explicitly described as unbalanced, disjointed, or dominated by an undesirable component.",
        },
    ),
    "sensory_defect": evidence(
        "an explicit sensory defect or off-flavor in this exact coffee",
        "Examples are moldy, musty, phenolic, medicinal, fermenty as a flaw, or tainted. "
        "Do not treat an intentionally fermented process or generic roast note as a defect. "
        "A merely disliked hint of carbon or roast note is insufficient without a direct defect claim. "
        "A statement that the coffee is clean is not an explicit denial of every possible defect.",
    ),
    "lot_auction": evidence(
        "that this exact coffee lot was offered or purchased through a named coffee auction",
        "A roaster's general auction history or a producer's other auction lots do not count. "
        "A mere auction-inspired product name does not count.",
    ),
    "lot_competition": evidence(
        "that this exact coffee lot won, placed, or was selected in a named coffee competition",
        "An award for the roaster, farm, producer, or another lot does not count. "
        "Require an independent lot competition such as Cup of Excellence or Best of Panama. "
        "A Coffee Review Top 50 ranking, review-site list, or direct review score is not a competition result.",
    ),
    "documented_scarcity": evidence(
        "a concrete limited quantity or numbered release for this exact lot",
        "Require an explicit lot quantity, production quantity, or numbered allocation. "
        "Generic rare, exclusive, limited edition, seasonal, or sold-out marketing is insufficient. "
        "Do not use retail price as evidence.",
    ),
}


def question_hash() -> str:
    return hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
