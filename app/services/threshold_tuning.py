"""Threshold evaluation metrics for labeled face-pair similarities."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LabeledSimilarity:
    similarity: float
    is_match: bool


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    true_accepts: int
    false_accepts: int
    true_rejects: int
    false_rejects: int
    false_accept_rate: float
    false_reject_rate: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    half_total_error_rate: float


@dataclass(frozen=True)
class ThresholdEvaluation:
    metrics: tuple[ThresholdMetrics, ...]
    recommended: ThresholdMetrics


def evaluate_thresholds(
    samples: list[LabeledSimilarity],
    thresholds: list[float],
) -> ThresholdEvaluation:
    """Evaluate thresholds and recommend the lowest balanced error point."""

    if not samples:
        raise ValueError("At least one labeled similarity is required.")
    if not any(sample.is_match for sample in samples):
        raise ValueError("At least one genuine same-person pair is required.")
    if not any(not sample.is_match for sample in samples):
        raise ValueError("At least one different-person pair is required.")
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    if any(not -1.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("Every threshold must be between -1.0 and 1.0.")

    results = tuple(
        _evaluate_threshold(samples, threshold)
        for threshold in sorted(set(thresholds))
    )
    recommended = min(
        results,
        key=lambda result: (
            result.half_total_error_rate,
            abs(result.false_accept_rate - result.false_reject_rate),
            -result.threshold,
        ),
    )
    return ThresholdEvaluation(metrics=results, recommended=recommended)


def _evaluate_threshold(
    samples: list[LabeledSimilarity],
    threshold: float,
) -> ThresholdMetrics:
    true_accepts = 0
    false_accepts = 0
    true_rejects = 0
    false_rejects = 0

    for sample in samples:
        predicted_match = sample.similarity >= threshold
        if sample.is_match and predicted_match:
            true_accepts += 1
        elif sample.is_match:
            false_rejects += 1
        elif predicted_match:
            false_accepts += 1
        else:
            true_rejects += 1

    genuine_count = true_accepts + false_rejects
    impostor_count = false_accepts + true_rejects
    false_accept_rate = false_accepts / impostor_count
    false_reject_rate = false_rejects / genuine_count
    accuracy = (true_accepts + true_rejects) / len(samples)

    predicted_positive_count = true_accepts + false_accepts
    precision = (
        true_accepts / predicted_positive_count
        if predicted_positive_count
        else 0.0
    )
    recall = true_accepts / genuine_count
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return ThresholdMetrics(
        threshold=threshold,
        true_accepts=true_accepts,
        false_accepts=false_accepts,
        true_rejects=true_rejects,
        false_rejects=false_rejects,
        false_accept_rate=false_accept_rate,
        false_reject_rate=false_reject_rate,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        half_total_error_rate=(false_accept_rate + false_reject_rate) / 2,
    )
