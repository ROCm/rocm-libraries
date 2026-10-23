# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generic GPU target table and arch-list tier algebra; see descriptor-packaging/README.md.

Mirrors ``hkp_pack.generic_targets`` and the C++ ``archTier`` (not importable from here).
"""

import json
from functools import cache
from pathlib import Path

TIER_EXPLICIT = 0
TIER_GENERIC = 1
TIER_UNRESTRICTED = 2

DEFAULT_TABLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "plugin_sdk"
    / "data"
    / "gpu_generic_targets.json"
)

_GENERIC_SUFFIX = "-generic"


class GenericTargets:
    def __init__(self, path: Path, generics: dict):
        self._path = Path(path)
        self._generics = generics

    @classmethod
    def load(cls, path) -> "GenericTargets":
        """Read the table at @path; ``ValueError`` on any defect."""
        path = Path(path)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(
                f"{path}: cannot read generic target table: {exc}"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"{path}: generic target table is not valid JSON: {exc}"
            ) from exc
        raw = doc.get("generics") if isinstance(doc, dict) else None
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: 'generics' must be an object")
        generics = {}
        for name, members in raw.items():
            if (
                not isinstance(members, list)
                or not members
                or not all(isinstance(m, str) and m for m in members)
            ):
                raise ValueError(
                    f"{path}: generics.{name} must be a non-empty array of names"
                )
            generics[name] = tuple(members)
        return cls(path, generics)

    @property
    def path(self) -> Path:
        return self._path

    def has(self, name: str) -> bool:
        return name in self._generics

    def members(self, name: str) -> tuple:
        return self._generics[name]

    def names(self) -> tuple:
        return tuple(self._generics)


@cache
def default_table() -> GenericTargets:
    return GenericTargets.load(DEFAULT_TABLE_PATH)


def is_generic_shaped(name: str) -> bool:
    return name.endswith(_GENERIC_SUFFIX)


def _strip_features(arch: str) -> str:
    return arch.split(":", 1)[0]


def entry_tier(entry: str, device: str, table: GenericTargets):
    """Tier of one entry for @device, or None when it does not admit it."""
    if is_generic_shaped(entry):
        if table.has(entry) and _strip_features(device) in table.members(entry):
            return TIER_GENERIC
        return None
    if device == entry or device.startswith(entry + ":"):
        return TIER_EXPLICIT
    return None


def list_tier(entries, device: str, table: GenericTargets):
    """Best tier of @entries for @device; None = no match."""
    if not entries:
        return TIER_UNRESTRICTED
    tiers = [
        t for t in (entry_tier(e, device, table) for e in entries) if t is not None
    ]
    return min(tiers) if tiers else None


def expand(entries, table: GenericTargets):
    """Device ids @entries admits; None when unrestricted."""
    if not entries:
        return None
    devices: set = set()
    for entry in entries:
        if is_generic_shaped(entry):
            if table.has(entry):
                devices.update(table.members(entry))
        else:
            devices.add(entry)
    return frozenset(devices)


def covers(outer, inner, table: GenericTargets) -> bool:
    """Is every device @inner admits also admitted by @outer?"""
    expanded_outer, expanded_inner = expand(outer, table), expand(inner, table)
    if expanded_outer is None or expanded_inner is None:
        return True
    return expanded_inner <= expanded_outer


def compete(a, b, table: GenericTargets) -> bool:
    """Do the lists match some device at the same tier?"""
    if not a and not b:
        return True
    candidates: set = set()
    for entry in (*a, *b):
        if is_generic_shaped(entry):
            if table.has(entry):
                candidates.update(table.members(entry))
        else:
            candidates.add(entry)
    for device in candidates:
        tier = list_tier(a, device, table)
        if tier is not None and tier == list_tier(b, device, table):
            return True
    return False
