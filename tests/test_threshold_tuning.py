"""Threshold metric and recommendation tests."""

import pytest

from app.services.threshold_tuning import (
    LabeledSimilarity,
    evaluate_thresholds,
)


def test_evaluates_thresholds_and_recommends_lowest_balanced_error() -> None:
    samples = [
        LabeledSimilarity(similarity=0.90, is_match=True),
        LabeledSimilarity(similarity=0.70, is_match=True),
        LabeledSimilarity(similarity=0.45, is_match=False),
        LabeledSimilarity(similarity=0.20, is_match=False),
    ]

    evaluation = evaluate_thresholds(samples, [0.40, 0.50, 0.80])

    assert evaluation.recommended.threshold == 0.50
    assert evaluation.recommended.false_accept_rate == 0.0
    assert evaluation.recommended.false_reject_rate == 0.0
    assert evaluation.recommended.accuracy == 1.0


def test_similarity_equal_to_threshold_is_accepted() -> None:
    samples = [
        LabeledSimilarity(similarity=0.50, is_match=True),
        LabeledSimilarity(similarity=0.20, is_match=False),
    ]

    evaluation = evaluate_thresholds(samples, [0.50])

    assert evaluation.recommended.true_accepts == 1
    assert evaluation.recommended.false_rejects == 0


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        ([], "At least one labeled"),
        (
            [LabeledSimilarity(similarity=0.5, is_match=False)],
            "same-person",
        ),
        (
            [LabeledSimilarity(similarity=0.5, is_match=True)],
            "different-person",
        ),
    ],
)
def test_requires_a_meaningful_labeled_dataset(
    samples: list[LabeledSimilarity],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_thresholds(samples, [0.5])


def test_rejects_an_invalid_threshold() -> None:
    samples = [
        LabeledSimilarity(similarity=0.5, is_match=True),
        LabeledSimilarity(similarity=0.2, is_match=False),
    ]

    with pytest.raises(ValueError, match="between -1.0 and 1.0"):
        evaluate_thresholds(samples, [1.1])
