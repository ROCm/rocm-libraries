"""JSON conventions the runtime descriptor loader applies to every document.

Mirrored here so a document the runtime would refuse is a pack-time error.
The loader is authoritative (DescriptorLoader.hpp).
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


def loads_strict(text):
    """Parse JSON text, rejecting a duplicate key in any object at any depth.

    A plain dict parse keeps the last of two equal keys, so a document the
    runtime refuses would otherwise pack silently. Raises DuplicateKeyError or
    json.JSONDecodeError.
    """
    return json.loads(text, object_pairs_hook=_reject_duplicate_pairs)


def is_extension_key(key):
    """The runtime ignores `x-`/`_`-prefixed keys and `provenance`."""
    return key.startswith(("x-", "_")) or key == "provenance"


def require_known_keys(doc, allowed, where):
    """Allow known fields and the runtime's extension keys."""
    for key in doc:
        if key not in allowed and not is_extension_key(key):
            raise HkpPackError(f"{where} has unknown key '{key}'")
