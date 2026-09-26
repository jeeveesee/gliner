"""Loader for the helio-pii taxonomy + threshold config format.

Taxonomy (`configs/taxonomy.json`): each canonical label carries one or more
`prompt_labels` -- the actual zero-shot strings handed to GLiNER. Your own
zero-shot prompts are added the same way: append a string to a label's
`prompt_labels`, or add a whole new label entry.

Thresholds (`configs/thresholds.json`): keyed by canonical label *name*, not
by individual prompt, with a `default_threshold` fallback for any canonical
label without an explicit entry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Taxonomy:
    labels: list[dict]  # raw entries: {id, name, description, prompt_labels}
    thresholds: dict[str, float]
    default_threshold: float

    @property
    def canonical_names(self) -> list[str]:
        return [label["name"] for label in self.labels]

    @property
    def all_prompts(self) -> list[str]:
        """Flattened prompt list, in a fixed order -- this is the label list
        passed to GLiNER, and (at export time) the ONNX output class order."""
        return [prompt for label in self.labels for prompt in label["prompt_labels"]]

    @property
    def prompt_to_canonical(self) -> dict[str, str]:
        return {prompt: label["name"] for label in self.labels for prompt in label["prompt_labels"]}

    def threshold_for(self, canonical_name: str) -> float:
        return self.thresholds.get(canonical_name, self.default_threshold)

    @property
    def score_floor(self) -> float:
        """Lowest threshold in play -- use as GLiNER's single `threshold=`
        arg, then re-filter per canonical label threshold afterwards."""
        return min([*self.thresholds.values(), self.default_threshold])


def load_taxonomy(taxonomy_path: str | Path, thresholds_path: str | Path) -> Taxonomy:
    taxonomy_json = json.loads(Path(taxonomy_path).read_text())
    thresholds_json = json.loads(Path(thresholds_path).read_text())
    return Taxonomy(
        labels=taxonomy_json["labels"],
        thresholds=thresholds_json["thresholds"],
        default_threshold=thresholds_json.get("default_threshold", 0.5),
    )
