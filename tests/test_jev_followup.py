"""Contract isolation and the fixed, price-only interaction layer."""

import unittest

import numpy as np
from scipy import sparse

from coffee_value.extraction.cache import cache_key
from coffee_value.extraction.catalog import AUCTION, FOLLOWUP
from coffee_value.extraction.encoding import encode, feature_names
from coffee_value.extraction.state import build_review_state
from scripts.evaluate_jev_followup import INTERACTIONS, interaction_matrix
from scripts.evaluate_jev_auction import GESHA, PANAMA, pg_interaction


class JevFollowupTests(unittest.TestCase):
    def test_auction_contract_distinguishes_current_lot_from_context(self):
        options = AUCTION.questions["lot_auction_status"]["criteria"]
        self.assertEqual(len(AUCTION.questions), 1)
        self.assertTrue({"exact_lot_auctioned", "auction_component_only", "other_lot_or_history",
                         "explicit_non_auction", "not_stated"} <= set(options))
        state = build_review_state({"bean": "An auction lot"})
        self.assertNotEqual(cache_key(state), cache_key(state, contract_version=AUCTION.version,
                                           questions_hash=AUCTION.questions_hash, model=AUCTION.model))

    def test_incremental_answers_are_versioned_and_strict(self):
        self.assertEqual(len(FOLLOWUP.questions), 8)
        state = build_review_state({"bean": "A coffee", "notes": "Long, clean finish"})
        self.assertNotEqual(cache_key(state), cache_key(state, contract_version=FOLLOWUP.version,
                                           questions_hash=FOLLOWUP.questions_hash, model=FOLLOWUP.model))
        answers = {}
        for qid, question in FOLLOWUP.questions.items():
            options = list(question["criteria"])
            answers[qid] = {"type": "choice", "choice": options[0],
                            "probabilities": {option: float(option == options[0]) for option in options}}
        record = {"status": "complete", "contract_version": FOLLOWUP.version,
                  "question_hash": FOLLOWUP.questions_hash, "requested_model": FOLLOWUP.model,
                  "resolved_model": FOLLOWUP.model, "answers": answers}
        self.assertEqual(len(encode(record, questions=FOLLOWUP.questions,
                                    contract_version=FOLLOWUP.version,
                                    questions_hash=FOLLOWUP.questions_hash,
                                    model=FOLLOWUP.model)), len(feature_names(questions=FOLLOWUP.questions)))
        record["answers"].pop("documented_scarcity")
        with self.assertRaises(ValueError):
            encode(record, questions=FOLLOWUP.questions, contract_version=FOLLOWUP.version,
                   questions_hash=FOLLOWUP.questions_hash, model=FOLLOWUP.model)

    def test_price_interactions_are_probability_products(self):
        base_names = feature_names()
        extra_names = feature_names(questions=FOLLOWUP.questions)
        base = np.zeros((2, len(base_names)))
        extra = np.zeros((2, len(extra_names)))
        base[0, base_names.index("jev:origin_country:Panama")] = 0.8
        base[0, base_names.index("jev:variety_gesha:supported")] = 0.5
        extra[0, extra_names.index("jev:lot_auction:supported")] = 0.3
        extra[0, extra_names.index("jev:documented_scarcity:supported")] = 0.2
        matrix = interaction_matrix(sparse.csr_matrix(base), sparse.csr_matrix(extra))
        self.assertEqual(matrix.shape, (2, len(INTERACTIONS)))
        np.testing.assert_allclose(matrix.toarray(), [[0.4, 0.06], [0.0, 0.0]])

    def test_panama_gesha_interaction_uses_existing_jev_probabilities(self):
        base = np.zeros((2, len(feature_names())))
        base[0, PANAMA], base[0, GESHA] = 0.8, 0.5
        base[1, PANAMA], base[1, GESHA] = 0.1, 0.9
        np.testing.assert_allclose(pg_interaction(sparse.csr_matrix(base)).toarray(), [[0.4], [0.09]])


if __name__ == "__main__":
    unittest.main()
