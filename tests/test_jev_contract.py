"""Tests for target isolation and strict reuse of cached TypeSafe answers."""

import unittest
import pickle

import numpy as np
from scipy import sparse

from coffee_value.extraction.cache import cache_key
from coffee_value.extraction.encoding import encode, feature_names
from coffee_value.extraction.questions import CONTRACT_VERSION, MODEL, QUESTIONS, question_hash
from coffee_value.extraction.state import build_review_state
from scripts.evaluate_jev import bundle_matrix, clean_text_rows, predict_rating
from autoresearch.price.train import FeatureEncoder, PACKAGE_FEATURE_NAMES


def valid_record():
    answers = {}
    for qid, question in QUESTIONS.items():
        options = list(question["criteria"])
        answers[qid] = {"type": "choice", "choice": options[0],
                        "probabilities": {option: float(i == 0) for i, option in enumerate(options)},
                        "confidence": 1.0}
    return {"status": "complete", "contract_version": CONTRACT_VERSION,
            "question_hash": question_hash(), "requested_model": MODEL,
            "resolved_model": MODEL, "answers": answers}


class JevContractTests(unittest.TestCase):
    def test_review_state_excludes_targets_and_keeps_altitude(self):
        row = {"bean": "Coffee", "origin": "Ethiopia, 1800 masl", "blind_assessment": "Score 94. Floral coffee, $30 / 250g.",
               "notes": "Lactic process, 48 hours", "rating": "94", "aroma": "8", "body": "9",
               "est_price": "$30", "price_usd_per_100g_real": "12"}
        state = build_review_state(row)
        text = str(state)
        self.assertNotIn("94", text)
        self.assertNotIn("$30", text)
        self.assertNotIn("aroma", text)
        self.assertNotIn("price_usd", text)
        self.assertIn("1800 masl", text)
        self.assertIn("48 hours", text)
        self.assertNotIn("95-point", str(build_review_state({"notes": "95-point coffee"})))
        self.assertNotIn("100 USD", str(build_review_state({"notes": "100 USD retail"})))
        self.assertNotIn("(93)", str(build_review_state({"blind_assessment": "The critic wrote (93)."})))
        self.assertIn("(90) hours", str(build_review_state({"notes": "Fermented for (90) hours"})))
        self.assertNotIn("scored an 85", str(build_review_state({"blind_assessment": "It scored an 85 in review."})))

    def test_complete_distribution_and_version_required(self):
        record = valid_record()
        self.assertEqual(len(encode(record)), len(feature_names()))
        record["answers"].pop("blend")
        with self.assertRaises(ValueError):
            encode(record)
        record = valid_record()
        record["resolved_model"] = "jev-latest"
        with self.assertRaises(ValueError):
            encode(record)
        record = valid_record()
        record["answers"]["blend"] = None
        with self.assertRaises(ValueError):
            encode(record)

    def test_cache_key_changes_with_evidence(self):
        a = build_review_state({"origin": "Kenya", "blind_assessment": "floral"})
        b = build_review_state({"origin": "Kenya", "blind_assessment": "chocolate"})
        self.assertNotEqual(cache_key(a), cache_key(b))

    def test_saved_candidate_transforms_reproduce_matrix(self):
        items = [{"row_id": str(i), "sensory_text": "Floral coffee, 95-point score. Bright citrus.",
                  "producer_text": "Farm lot, 20 USD", "package_grams": "250"} for i in range(10)]
        vectors = {item["row_id"]: np.full(len(feature_names()), i / 10)
                   for i, item in enumerate(items)}
        encoder = FeatureEncoder(min_df=1)
        encoder.fit(clean_text_rows(items))
        offset = len(encoder.structured_vocab)
        expected = sparse.hstack((sparse.csr_matrix(np.stack([vectors[i["row_id"]] for i in items])),
                                  encoder.transform(clean_text_rows(items))[:, offset:]), format="csr")
        bundle = {"run": "hybrid", "task": "price", "text_encoder": encoder,
                  "text_column_offset": offset, "feature_names": feature_names() + encoder.feature_names[offset:]}
        restored = pickle.loads(pickle.dumps(bundle))
        actual = bundle_matrix(restored, items, vectors)
        self.assertEqual(actual.shape[1], len(restored["feature_names"]))
        np.testing.assert_allclose(actual.toarray(), expected.toarray())
        only = {"run": "jev_only", "task": "price", "text_encoder": None,
                "package_transform": {"package_log_mean": np.log(250), "package_log_std": 1.0},
                "feature_names": feature_names() + PACKAGE_FEATURE_NAMES}
        restored_only = pickle.loads(pickle.dumps(only))
        self.assertEqual(bundle_matrix(restored_only, items, vectors).shape[1], len(restored_only["feature_names"]))

    def test_both_rating_candidate_models_survive_serialization(self):
        train = [{"rating": str(88 + i % 7)} for i in range(14)]
        val = [{"rating": "91"}, {"rating": "94"}]
        base = sparse.csr_matrix(np.arange(14 * 6, dtype=float).reshape(14, 6) / 20)
        held = sparse.csr_matrix(np.arange(2 * 6, dtype=float).reshape(2, 6) / 20)
        for x_train, x_val in ((base, held),
                               (sparse.hstack((base, base[:, :2]), format="csr"),
                                sparse.hstack((held, held[:, :2]), format="csr"))):
            _, preds, model = predict_rating(x_train, x_val, train, val)
            restored = pickle.loads(pickle.dumps(model))
            np.testing.assert_allclose(restored.predict(x_val), preds)


if __name__ == "__main__":
    unittest.main()
