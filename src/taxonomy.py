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
import re
from dataclasses import dataclass
from pathlib import Path

from validators import CHECKSUMS


@dataclass(frozen=True)
class Taxonomy:
    labels: list[dict]  # raw entries: {id, name, description, prompt_labels, validation_regex?}
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

    @property
    def validation_regex(self) -> dict[str, str]:
        """Canonical labels with a fixed, checkable shape (email, phone,
        ssn, npi, ...), from each label's optional `validation_regex`.
        Labels without one (name, diagnosis, address, ...) have no fixed
        shape to check against -- there's nothing to add here for them."""
        return {label["name"]: label["validation_regex"] for label in self.labels if label.get("validation_regex")}

    @property
    def validation_checksum(self) -> dict[str, str]:
        """Canonical labels that additionally require a named checksum
        (see validators.CHECKSUMS), from each label's optional
        `validation_checksum` field. Opt-in and off by default: a checksum
        requires the exact real math to work, so turning it on for data it
        hasn't been checked against (e.g. a synthetic test harness whose
        fake numbers were never computed to be checksum-valid) can silently
        zero out recall instead of cleaning up precision."""
        return {
            label["name"]: label["validation_checksum"] for label in self.labels if label.get("validation_checksum")
        }

    def is_valid(self, canonical: str, text: str) -> bool:
        """Format-check a flagged span's exact text for a canonical label
        that has a fixed shape. A label with no validation_regex/checksum
        always passes -- there's nothing to check it against. Strips
        surrounding punctuation first (a trailing "." from the end of a
        sentence shouldn't fail an otherwise-correct match)."""
        text = text.strip().strip(".,;:()\"'")
        pattern = self.validation_regex.get(canonical)
        if pattern and not re.fullmatch(pattern, text):
            return False
        checksum_name = self.validation_checksum.get(canonical)
        if checksum_name:
            checksum = CHECKSUMS.get(checksum_name)
            if checksum and not checksum(text):
                return False
        return True


def load_taxonomy(taxonomy_path: str | Path, thresholds_path: str | Path) -> Taxonomy:
    taxonomy_json = json.loads(Path(taxonomy_path).read_text())
    thresholds_json = json.loads(Path(thresholds_path).read_text())
    return Taxonomy(
        labels=taxonomy_json["labels"],
        thresholds=thresholds_json["thresholds"],
        default_threshold=thresholds_json.get("default_threshold", 0.5),
    )
