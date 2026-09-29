# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Config identity of tuned specs.

A tuned spec is one registered variant plus a canonical knob dict: the kernel
fields set away from the variant's default spec (``waves_per_eu`` included).
``config_key`` hashes an explicit, versioned payload over exactly that -- the
family ABI, arch, path, variant id, knobs -- plus a fingerprint of the
defaults the knobs are relative to. Problem fields and runtime specializations
are not in it, so one key names the same configuration on every problem the
variant admits.

The defaults fingerprint is what keeps a stored pin honest across releases:
knobs are a delta, so if an in-tree default changes (a base-spec constant, a
kernel dataclass default) the same knobs would silently build a different
kernel. With the defaults in the key, the recomputed key no longer matches the
stored one and the pin is refused instead.

``tuning_id`` is ``{variant_id}_wpe{N}@{config_key}``. The stem is display
only: pins are matched on the ``config_key`` suffix (:func:`key_of`), never on
the stem. Bump :data:`TUNING_ID_VERSION` whenever the payload or the
canonical-knob rules change.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Callable, Iterable, Mapping, Optional, Union

from ..core import stable_json_hash
from .axes import Knobs

TUNING_ID_VERSION = 2

_KNOB_VALUE_TYPES = (bool, int, float, str, type(None))


def knob_items(knobs: Mapping[str, object]) -> Knobs:
    """``knobs`` as the sorted pair tuple specs and requests carry.

    Values must be JSON scalars: the dict is hashed into ``config_key`` and
    written to benchmark rows, so it has to round-trip through JSON.
    """
    for name, value in knobs.items():
        if not isinstance(name, str) or not isinstance(value, _KNOB_VALUE_TYPES):
            raise TypeError(f"knob {name!r}={value!r} is not a JSON scalar")
    return tuple(sorted(knobs.items()))


def normalize_knobs(value: Union[Mapping[str, object], Iterable, None]) -> Knobs:
    """A request's ``tuning_knobs`` from what a caller stored: a mapping, or
    ``(name, value)`` pairs (lists, as JSON gives them back, included).
    Raises ``TypeError`` / ``ValueError`` at the request boundary rather than
    deep inside a cache."""
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return knob_items(value)
    pairs = {}
    for pair in value:
        if isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise ValueError(f"tuning_knobs entry {pair!r} is not a (name, value) pair")
        name, knob = pair
        if name in pairs:
            raise ValueError(f"tuning_knobs sets {name!r} twice")
        pairs[name] = knob
    return knob_items(pairs)


def jsonable(value):
    """``value`` as JSON-stable data; raises ``TypeError`` on anything whose
    serialization could change between runs."""
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, _KNOB_VALUE_TYPES):
        return value
    raise TypeError(f"{value!r} ({type(value).__name__}) has no stable JSON form")


def defaults_fingerprint(defaults: Mapping[str, object]) -> str:
    """Hash of the problem-independent defaults a variant's knobs are relative to."""
    return stable_json_hash({"defaults": jsonable(defaults)}, n=16)


def config_key(
    *, abi: str, arch: str, path: str, variant_id: str, knobs: Knobs, defaults: str = ""
) -> str:
    """``abi`` is the family's ABI version string, which also names the family;
    ``defaults`` is :func:`defaults_fingerprint` of the variant's defaults."""
    return stable_json_hash(
        {
            "v": TUNING_ID_VERSION,
            "abi": abi,
            "arch": arch,
            "path": path,
            "variant": variant_id,
            "knobs": [[name, value] for name, value in knobs],
            "defaults": defaults,
        },
        n=16,
    )


def tuning_id(variant_id: str, waves_per_eu: Optional[int], key: str) -> str:
    wpe = "none" if waves_per_eu is None else int(waves_per_eu)
    return f"{variant_id}_wpe{wpe}@{key}"


def key_of(wanted: str) -> str:
    """The ``config_key`` a ``tuning_id`` carries -- the part pins are matched
    on. An id without one is its own key (it will match nothing)."""
    return wanted.rsplit("@", 1)[-1]


def names_variant(wanted: str, variant_id: str) -> bool:
    """Cheap reject before a space walk: an id always starts with its variant's
    ``{variant_id}_wpe`` stem (the variant id is identity, the WPE is not)."""
    return wanted.split("@", 1)[0].startswith(f"{variant_id}_wpe")


def drop_base_equal(
    knobs: Mapping[str, object], base_value: Callable[[str], object]
) -> dict:
    """``knobs`` without the entries that restate the base value.

    Setting a field to what the default spec already has is the default
    kernel again, so it must not mint a second id. ``base_value`` returns
    ``dataclasses.MISSING`` for a field it cannot judge; such knobs are kept.
    """
    kept = {}
    for name, value in knobs.items():
        base = base_value(name)
        if base is MISSING or base != value:
            kept[name] = value
    return kept
