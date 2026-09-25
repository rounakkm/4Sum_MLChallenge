"""Macro-averaged F0.5 scoring, matching the challenge's evaluation formula exactly.

F_beta = (1 + beta^2) * P * R / (beta^2 * P + R), beta = 0.5.
Computed per Source 1 entity, then averaged across all entities (singletons
included, per the problem statement).
"""


def f_beta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_entity(pred_ids, true_ids):
    """Score a single Source 1 entity's predicted vs true match set."""
    pred_ids, true_ids = set(pred_ids), set(true_ids)

    if not true_ids:
        # Singleton: correct empty prediction = 1.0, any predicted match = 0.0
        return 1.0 if not pred_ids else 0.0

    if not pred_ids:
        return 0.0

    tp = len(pred_ids & true_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    return f_beta(precision, recall, beta=0.5)


def macro_f_beta(predictions, ground_truth):
    """predictions, ground_truth: dict[source1_entity_id] -> iterable of ids."""
    scores = [score_entity(predictions.get(eid, []), true_ids) for eid, true_ids in ground_truth.items()]
    return sum(scores) / len(scores) if scores else 0.0
