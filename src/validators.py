"""Format checks beyond a plain regex, for labels with a real checksum digit.

A regex only checks shape ("10 digits"); a checksum checks that the last
digit is mathematically consistent with the rest. It's strict in a way a
regex can't be -- which also makes it risky to apply blindly: a checksum
requires the EXACT real math to work, so it silently rejects a genuine
value that just has a stray trailing character, or an entire synthetic test
set whose fake numbers were never computed to be checksum-valid in the
first place (a real credit card number always passes Luhn; a placeholder
one typed for a test fixture usually doesn't). Applying a checksum to the
model's OWN detections can therefore *destroy* recall rather than clean up
precision, on data it wasn't checked against.

CHECKSUMS is keyed by algorithm name (not canonical label) and is opt-in --
`Taxonomy.validation_checksum` (from an explicit `validation_checksum`
field in taxonomy.json) decides which labels actually use one. Nothing here
runs unless a label's taxonomy entry asks for it by name.
"""

from __future__ import annotations

import re


def luhn_ok(digits: str) -> bool:
    """The standard Luhn (mod 10) checksum used by credit cards and NPIs."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def npi_checksum_ok(text: str) -> bool:
    """CMS NPI check digit: prepend the fixed prefix 80840, then Luhn."""
    return len(text) == 10 and text.isdigit() and luhn_ok("80840" + text)


def credit_card_checksum_ok(text: str) -> bool:
    digits = re.sub(r"[ -]", "", text)
    return digits.isdigit() and 13 <= len(digits) <= 19 and luhn_ok(digits)


# Extra checks applied on top of validation_regex, keyed by algorithm name --
# opt-in per label via taxonomy.json's `validation_checksum` field, not
# automatically applied to every label with that name.
CHECKSUMS = {
    "npi_luhn": npi_checksum_ok,
    "luhn": credit_card_checksum_ok,
}
