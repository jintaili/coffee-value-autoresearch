"""One price-focused JEV question about this exact coffee's auction status."""

from __future__ import annotations

import hashlib
import json

from .questions import MODEL

CONTRACT_VERSION = "coffee-auction-1"

QUESTIONS: dict[str, dict] = {
    "lot_auction_status": {
        "type": "choice",
        "instructions": (
            "Judge only the reviewed coffee offered here, using direct source evidence. "
            "Which statement about its auction status is supported? A name containing "
            "'Auction Lot' can identify this product, but a generic auction system, a "
            "producer's other lots, a historical price record, or a blend component "
            "must not be assigned to the whole reviewed lot. Ignore scores and prices."
        ),
        "criteria": {
            "exact_lot_auctioned": "This exact reviewed coffee lot is stated to have been offered, sold, or purchased at a coffee auction, or is explicitly named an auction lot.",
            "auction_component_only": "The reviewed product is a blend containing an auction-lot component, but the whole blend is not identified as an auction lot.",
            "other_lot_or_history": "The auction statement concerns another lot, a producer or roaster's history, or a generic auction system rather than this exact reviewed lot.",
            "explicit_non_auction": "This exact reviewed coffee is directly described as a non-auction lot.",
            "auction_mentioned_unclear": "Auction is mentioned, but the source does not establish which lot or product it applies to.",
            "not_stated": "No auction-related claim is stated for this reviewed coffee.",
            "conflicting": "Direct statements about this exact reviewed lot's auction status are incompatible.",
        },
    }
}


def question_hash() -> str:
    return hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
