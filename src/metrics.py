"""Exact-span precision/recall(sensitivity)/F1, per-label + micro/macro."""

from __future__ import annotations


def _span_key(ent: dict) -> tuple[int, int, str]:
    return (ent["start"], ent["end"], ent["label"])


def evaluate(
    gold_docs: list[list[dict]],
    pred_docs: list[list[dict]],
    labels: list[str],
) -> tuple[list[dict], dict, dict]:
    """Exact (start, end, label) match scoring.

    Args:
        gold_docs: one list of {start, end, label} dicts per document (ground truth).
        pred_docs: one list of {start, end, label} dicts per document (model output).
        labels: label universe to report on.

    Returns:
        (per_label_rows, micro_totals, macro_totals)
    """
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}

    for gold, pred in zip(gold_docs, pred_docs):
        gold_set = {_span_key(e) for e in gold}
        pred_set = {_span_key(e) for e in pred}

        for key in pred_set:
            if key[2] not in counts:
                continue
            counts[key[2]]["tp" if key in gold_set else "fp"] += 1

        for key in gold_set - pred_set:
            if key[2] in counts:
                counts[key[2]]["fn"] += 1

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
