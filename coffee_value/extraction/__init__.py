"""Shared, versioned TypeSafe feature extraction for coffee descriptions."""

from .questions import CONTRACT_VERSION, MODEL, QUESTIONS, question_hash
from .state import build_review_state
from .encoding import feature_names, encode

__all__ = ["CONTRACT_VERSION", "MODEL", "QUESTIONS", "question_hash", "build_review_state", "feature_names", "encode"]
