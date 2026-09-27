"""Format checks beyond a plain regex, for labels with a real checksum digit.

A regex only checks shape ("10 digits"); a checksum checks that the last
digit is mathematically consistent with the rest, which is what actually
makes NPI and credit-card checks strict. Keyed by canonical label name since
a checksum algorithm can't be expressed as JSON, unlike `validation_regex`
in taxonomy.json.
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


# Extra checks applied on top of validation_regex, keyed by canonical label.
CHECKSUMS = {
    "npi": npi_checksum_ok,
    "credit_card": credit_card_checksum_ok,
}
