"""Versioned question sets that can be cached and evaluated independently."""

from __future__ import annotations

from dataclasses import dataclass

from . import auction, followup, questions


@dataclass(frozen=True)
class Catalog:
    name: str
    version: str
    model: str
    questions: dict[str, dict]
    questions_hash: str


BASE = Catalog("base", questions.CONTRACT_VERSION, questions.MODEL,
               questions.QUESTIONS, questions.question_hash())
FOLLOWUP = Catalog("followup", followup.CONTRACT_VERSION, followup.MODEL,
                   followup.QUESTIONS, followup.question_hash())
AUCTION = Catalog("auction", auction.CONTRACT_VERSION, auction.MODEL,
                  auction.QUESTIONS, auction.question_hash())


def selected(name: str) -> Catalog:
    if name == "base":
        return BASE
    if name == "followup":
        return FOLLOWUP
    if name == "auction":
        return AUCTION
    raise ValueError(f"unknown JEV catalog: {name}")
