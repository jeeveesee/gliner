"""Loading + splitting for the helio-pii test harness format
(data/test_harness.jsonl): one JSON object per line with at least `id`,
`text`, `labels`, and `scenario_type`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def load_harness(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def stratified_split(
    docs: list[dict],
    holdout_frac: float = 0.3,
    seed: int = 0,
    key: str = "scenario_type",
) -> tuple[list[dict], list[dict]]:
    """Split docs into (tune, holdout), keeping each `key` group's share
    roughly equal in both halves so a rare scenario_type isn't dropped
    entirely from one side.

    Use the tune half to pick thresholds or choose between models; only
    trust the holdout half's metrics as the non-overfit answer -- anything
    tuned on the same data it's then scored on looks better than it is.
    """
    rng = random.Random(seed)
    groups: dict[object, list[dict]] = {}
    for d in docs:
        groups.setdefault(d.get(key), []).append(d)

    tune, holdout = [], []
    for group_docs in groups.values():
        shuffled = group_docs[:]
        rng.shuffle(shuffled)
        n_holdout = round(len(shuffled) * holdout_frac) if len(shuffled) > 1 else 0
        holdout.extend(shuffled[:n_holdout])
        tune.extend(shuffled[n_holdout:])
    return tune, holdout
