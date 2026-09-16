# SPDX-FileCopyrightText: 2026 Kunyang Zhang
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Offline tests for `KeywordToxicityScorer` keyword matching.

The scorer is part of the default output guardrail, so a keyword hit rejects the
response. These tests pin the two ways a hit used to be reported for text that
contains none of the configured keywords as words of its own.
"""

from __future__ import annotations

import pytest

from rai_toolkit.scorers import KeywordToxicityScorer


@pytest.fixture
def scorer() -> KeywordToxicityScorer:
    return KeywordToxicityScorer()


@pytest.mark.parametrize(
    "response",
    [
        "She has strong skills in Python and SQL.",
        "Please describe the skill set required.",
        "The system reached stability after the fix.",
        "The installer is stable and well tested.",
        "An offshoot of the original project.",
        "That sentence is an oxymoron.",
        "The paper describes a bombardier beetle.",
        "Weather forecasts are usually accurate.",
    ],
)
def test_words_that_merely_contain_a_keyword_are_not_flagged(
    scorer: KeywordToxicityScorer, response: str
) -> None:
    """Regression: "skills" contains "kill", "stability" contains "stab",
    "offshoot" contains "shoot" and "oxymoron" contains "moron"."""
    result = scorer.score(response)

    assert result.details["toxic_categories"] == {}
    assert result.details["total_matches"] == 0
    assert result.score == 1.0
    assert result.passed
    assert result.assessed


@pytest.mark.parametrize(
    ("response", "category"),
    [
        ("I will kill the process.", "violence"),
        ("KILL THE PROCESS", "violence"),
        ("That is an attack on the schema.", "violence"),
        ("You are a moron.", "harassment"),
        ("It is worthLESS.", "harassment"),
        ("They called the group subhuman.", "hate_speech"),
        ("Please end your life.", "self_harm"),
        ("Do not cut yourself.", "self_harm"),
        ("A self-harm reference was removed.", "self_harm"),
    ],
)
def test_keywords_are_still_detected_as_words(
    scorer: KeywordToxicityScorer, response: str, category: str
) -> None:
    """The boundary only removes matches inside a longer Latin word."""
    result = scorer.score(response)

    assert category in result.details["toxic_categories"]
    assert not result.passed


def test_matched_keyword_is_reported_in_its_configured_form(
    scorer: KeywordToxicityScorer,
) -> None:
    result = scorer.score("KILL THE PROCESS")

    assert result.details["toxic_categories"] == {"violence": ["kill"]}


def test_extra_keywords_apply_to_the_instance_that_declares_them() -> None:
    scorer = KeywordToxicityScorer(extra_keywords={"violence": ["zap"]})

    result = scorer.score("The tool will zap the record.")

    assert result.details["toxic_categories"] == {"violence": ["zap"]}
    assert not result.passed


def test_extra_keywords_can_introduce_a_new_category() -> None:
    scorer = KeywordToxicityScorer(extra_keywords={"profanity": ["darn"]})

    result = scorer.score("Well, darn.")

    assert result.details["toxic_categories"] == {"profanity": ["darn"]}


def test_extra_keywords_do_not_leak_into_other_instances() -> None:
    """Regression: a shallow copy of the category mapping extended the lists held
    by the class attribute, so a custom keyword reached every later scorer."""
    KeywordToxicityScorer(extra_keywords={"violence": ["zap"]})

    assert "zap" not in KeywordToxicityScorer.TOXIC_CATEGORIES["violence"]

    default_scorer = KeywordToxicityScorer()
    assert "zap" not in default_scorer._categories["violence"]
    assert default_scorer.score("The tool will zap the record.").details[
        "toxic_categories"
    ] == {}


def test_extra_keywords_do_not_leak_between_two_configured_scorers() -> None:
    first = KeywordToxicityScorer(extra_keywords={"violence": ["zap"]})
    second = KeywordToxicityScorer(extra_keywords={"violence": ["squash"]})

    assert first.score("squash").details["toxic_categories"] == {}
    assert second.score("zap").details["toxic_categories"] == {}


def test_keywords_outside_latin_script_still_match() -> None:
    """Word boundaries built from `\\b` would never match these, because
    consecutive characters of such scripts are word characters on both sides."""
    scorer = KeywordToxicityScorer(extra_keywords={"violence": ["暴力"]})

    result = scorer.score("这是一段暴力内容。")

    assert result.details["toxic_categories"] == {"violence": ["暴力"]}


def test_response_without_any_keyword_scores_a_full_pass(
    scorer: KeywordToxicityScorer,
) -> None:
    result = scorer.score("The migration finished without incidents.")

    assert result.score == 1.0
    assert result.passed
    assert result.explanation == "No toxic keywords detected"
    assert result.category == "MIT-1.2"
