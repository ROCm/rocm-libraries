"""JSON rules the runtime descriptor loader applies to every document.

Mirrored here so a document the runtime would refuse fails the pack. The loader
(DescriptorLoader.hpp) is authoritative.
"""

import json

from .errors import HkpPackError


class DuplicateKeyError(ValueError):
    """A JSON object names the same key twice."""

    def __init__(self, key):
        super().__init__(f"duplicate key '{key}'")
        self.key = key


def _reject_duplicate_pairs(pairs):
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise DuplicateKeyError(key)
        seen.add(key)
    return dict(pairs)


class NonStandardLiteralError(ValueError):
    """A JSON document uses NaN, Infinity or -Infinity, which JSON does not define."""

    def __init__(self, literal):
        super().__init__(f"non-standard literal '{literal}'")
        self.literal = literal


def _reject_constant(literal):
    raise NonStandardLiteralError(literal)


def loads_strict(text):
    """Parse JSON text as the runtime loader does.

    Rejects a duplicate key in any object at any depth, and the NaN, Infinity
    and -Infinity literals. A plain parse keeps the last of two equal keys and
    accepts those literals, so a document the runtime refuses would pack
    silently. Raises DuplicateKeyError, NonStandardLiteralError or
    json.JSONDecodeError.
    """
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_constant,
    )


def is_extension_key(key):
    """Return True if the runtime ignores key: an `x-`/`_` prefix, or `provenance`."""
    return key.startswith(("x-", "_")) or key == "provenance"


def require_known_keys(doc, allowed, where):
    """Raise HkpPackError on a key that is neither allowed nor an extension key.

    Keys are checked in sorted order, the order the runtime walks them, so a
    document with several unknown keys reports the same one on both sides.
    """
    for key in sorted(doc):
        if key not in allowed and not is_extension_key(key):
            raise HkpPackError(
                f"unknown key '{key}' in {where}; "
                "extension keys must start with 'x-' or '_'"
            )
