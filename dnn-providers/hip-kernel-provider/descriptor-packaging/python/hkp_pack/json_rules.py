"""JSON rules the runtime descriptor loader applies to every document.

Mirrored here so a document the runtime would refuse fails the pack. The loader
(DescriptorLoader.hpp) is authoritative.
"""

import json
import math
import re

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


class UnrepresentableValueError(ValueError):
    """A JSON value the runtime's parser (nlohmann) cannot represent.

    Either a number outside the range of a double, or a UTF-16 surrogate escape
    without its pair. The message is the parser's own wording, so the packer
    and the runtime report the same text.
    """


def _finite_number(convert):
    """A parse_int/parse_float hook rejecting a number that overflows a double.

    nlohmann parses an integer that does not fit in 64 bits as a double, and
    refuses any number whose double is not finite. Python would keep the
    integer exactly and turn the float into inf. Checking the token's double
    first also keeps int() from raising on an integer longer than its digit
    limit, which is far beyond a double's range.
    """

    def parse(token):
        if not math.isfinite(float(token)):
            raise UnrepresentableValueError(f"number overflow parsing '{token}'")
        return convert(token)

    return parse


# A \uD800-\uDFFF escape, in either case. Only an escape can put a surrogate in
# decoded text, because UTF-8 cannot encode one.
_SURROGATE_ESCAPE = re.compile(r"\\u[dD][89a-fA-F]")
_SURROGATE = re.compile("[\ud800-\udfff]")


def _reject_unpaired_surrogates(value):
    """Raise UnrepresentableValueError on the first unpaired surrogate.

    json.loads joins a valid high/low escape pair into one code point and keeps
    any other surrogate escape as a lone surrogate, which nlohmann refuses.
    Keys and values are walked in document order.
    """
    if isinstance(value, str):
        match = _SURROGATE.search(value)
        if match is None:
            return
        if match.group() <= "\udbff":
            message = "surrogate U+D800..U+DBFF must be followed by U+DC00..U+DFFF"
        else:
            message = "surrogate U+DC00..U+DFFF must follow U+D800..U+DBFF"
        raise UnrepresentableValueError(f"invalid string: {message}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_unpaired_surrogates(key)
            _reject_unpaired_surrogates(item)
    elif isinstance(value, list):
        for item in value:
            _reject_unpaired_surrogates(item)


def loads_strict(text):
    """Parse JSON text as the runtime loader does.

    Rejects a duplicate key in any object at any depth, the NaN, Infinity and
    -Infinity literals, a number outside the range of a double, and an unpaired
    surrogate escape. A plain parse keeps the last of two equal keys and
    accepts the rest, so a document the runtime refuses would pack silently.
    Raises DuplicateKeyError, NonStandardLiteralError, UnrepresentableValueError
    or json.JSONDecodeError.
    """
    doc = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_constant,
        parse_float=_finite_number(float),
        parse_int=_finite_number(int),
    )
    if _SURROGATE_ESCAPE.search(text):
        _reject_unpaired_surrogates(doc)
    return doc


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
