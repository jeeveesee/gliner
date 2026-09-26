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
