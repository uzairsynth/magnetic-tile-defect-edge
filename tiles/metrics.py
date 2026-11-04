"""Metrics for an imbalanced defect classifier.

Accuracy is reported but not used for decisions: 71% of tiles are defect-free,
so a model that calls everything "Free" is already 71% accurate. Macro-F1 weighs
every class equally, and the defect-detection view answers the question a
production line cares about: how many bad tiles get through?
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_summary(y_true, probs, classes: list[str]) -> dict:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    pred = probs.argmax(1)
    labels = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=0)

    defect_true = y_true != 0
    defect_pred = pred != 0
    defect_score = 1.0 - probs[:, 0]
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "per_class": {
            c: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(classes)
        },
        "defect_detection": {
            "defects": int(defect_true.sum()),
            "caught": int((defect_true & defect_pred).sum()),
            "recall": float((defect_true & defect_pred).sum() / max(defect_true.sum(), 1)),
            "good_tiles": int((~defect_true).sum()),
            "false_alarms": int((~defect_true & defect_pred).sum()),
            "false_alarm_rate": float((~defect_true & defect_pred).sum() / max((~defect_true).sum(), 1)),
            "pr_auc": float(average_precision_score(defect_true, defect_score)) if defect_true.any() else 0.0,
        },
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
    }


def choose_defect_threshold(y_true, probs, target_recall: float) -> float:
    """Highest defect-score threshold (fewest alarms) that still catches `target_recall` of defects.

    The defect score is 1 - P(Free): an alarm fires when the model is not confident the tile is good.
    """
    scores = np.sort(1.0 - np.asarray(probs)[np.asarray(y_true) != 0, 0])[::-1]
    if len(scores) == 0:
        return 0.5
    k = int(np.ceil(target_recall * len(scores))) - 1
    return float(scores[k])


def defect_alarm(y_true, probs, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    alarm = (1.0 - np.asarray(probs)[:, 0]) >= threshold
    defect = y_true != 0
    return {
        "threshold": float(threshold),
        "caught": int((alarm & defect).sum()),
        "defects": int(defect.sum()),
        "recall": float((alarm & defect).sum() / max(defect.sum(), 1)),
        "false_alarms": int((alarm & ~defect).sum()),
        "good_tiles": int((~defect).sum()),
        "false_alarm_rate": float((alarm & ~defect).sum() / max((~defect).sum(), 1)),
    }


def group_bootstrap(y_true, probs, groups, statistic, n: int = 2000, seed: int = 0) -> list[float]:
    """95% interval of `statistic(y, probs)` when whole tiles, not images, are resampled.

    Views of one tile are not independent samples, so groups are drawn with
    replacement, stratified by the group's class so rare defects stay present.
    """
    y_true, probs, groups = np.asarray(y_true), np.asarray(probs), np.asarray(groups)
    members = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    by_class: dict[int, list] = {}
    for g, idx in members.items():
        by_class.setdefault(int(np.bincount(y_true[idx]).argmax()), []).append(g)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        picked = [g for gs in by_class.values() for g in rng.choice(gs, size=len(gs), replace=True)]
        idx = np.concatenate([members[g] for g in picked])
        values.append(statistic(y_true[idx], probs[idx]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def macro_f1(y_true, probs) -> float:
    present = np.unique(y_true)
    return float(f1_score(y_true, np.asarray(probs).argmax(1), labels=present, average="macro", zero_division=0))


def defect_recall(y_true, probs) -> float:
    defect = np.asarray(y_true) != 0
    return float(((np.asarray(probs).argmax(1) != 0) & defect).sum() / max(defect.sum(), 1))
