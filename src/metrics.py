"""Precision/recall(sensitivity)/F1 for multi-label classification, per-label
+ micro/macro. Matches the helio-pii test harness: each document has a gold
`labels` list (canonical names, empty == non-PII) to compare against the
predicted label set for that same document.
"""

from __future__ import annotations


def evaluate(
    gold_label_sets: list[list[str]],
    pred_label_sets: list[list[str]],
    labels: list[str],
) -> tuple[list[dict], dict, dict]:
    """
    Args:
        gold_label_sets: one list of canonical label names per document (ground truth).
        pred_label_sets: one list of canonical label names per document (model output).
        labels: canonical label universe to report on.

    Returns:
        (per_label_rows, micro_totals, macro_totals)
    """
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}

    for gold, pred in zip(gold_label_sets, pred_label_sets):
        gold_set, pred_set = set(gold), set(pred)
        for label in labels:
            in_gold, in_pred = label in gold_set, label in pred_set
            if in_gold and in_pred:
                counts[label]["tp"] += 1
            elif in_pred and not in_gold:
                counts[label]["fp"] += 1
            elif in_gold and not in_pred:
                counts[label]["fn"] += 1

    rows = []
    for label, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0  # a.k.a. sensitivity
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {"label": label, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
        )

    macro = {
        "precision": sum(r["precision"] for r in rows) / len(rows),
        "recall": sum(r["recall"] for r in rows) / len(rows),
        "f1": sum(r["f1"] for r in rows) / len(rows),
    }

    tp_sum = sum(c["tp"] for c in counts.values())
    fp_sum = sum(c["fp"] for c in counts.values())
    fn_sum = sum(c["fn"] for c in counts.values())
    micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
    micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    micro = {"precision": micro_p, "recall": micro_r, "f1": micro_f1}

    return rows, micro, macro


def sweep_thresholds(
    gold_label_sets: list[list[str]],
    canonical_scores: list[dict[str, float]],
    labels: list[str],
    grid: list[float] | None = None,
) -> list[dict]:
    """Per-label threshold sweep: for each canonical label, scan `grid` and
    keep the threshold with the best F1 (ties broken toward higher recall).
    Each label is optimized independently -- they don't have to move together.

    Args:
        gold_label_sets: one list of canonical label names per document.
        canonical_scores: one {label: score} dict per document -- the same
            per-canonical-label score `evaluate`'s predictions were
            thresholded from, collected once and reused here (no re-running
            the model). A label missing from a doc's dict is treated as 0.0.
        labels: canonical label universe to sweep.
        grid: candidate thresholds; defaults to 0.05..0.95 in steps of 0.05.

    Returns:
        One row per label: {label, threshold, precision, recall, f1}.

    Note: this tunes thresholds on the same data you're evaluating on, which
    overfits to that harness. Prefer sweeping on a held-out split of the
    harness (or a separate one) and only trusting the number on data the
    threshold wasn't picked from.
    """
    if grid is None:
        grid = [round(i / 100, 2) for i in range(5, 100, 5)]

    rows = []
    for label in labels:
        gold = [label in set(g) for g in gold_label_sets]
        scores = [s.get(label, 0.0) for s in canonical_scores]
        best = None
        for t in grid:
            pred = [s >= t for s in scores]
            tp = sum(g and p for g, p in zip(gold, pred))
            fp = sum(p and not g for g, p in zip(gold, pred))
            fn = sum(g and not p for g, p in zip(gold, pred))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            if best is None or f1 > best["f1"] or (f1 == best["f1"] and recall > best["recall"]):
                best = {"label": label, "threshold": t, "precision": precision, "recall": recall, "f1": f1}
        rows.append(best)
    return rows
