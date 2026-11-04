"""Generic GPU target table and arch-tier algebra; see descriptor-packaging/README.md."""

import json
from pathlib import Path

from .errors import HkpPackError

TIER_EXPLICIT = 0
TIER_GENERIC = 1
TIER_UNRESTRICTED = 2

# .../dnn-providers/hip-kernel-provider/descriptor-packaging/python/hkp_pack/<this file>
DEFAULT_TABLE_PATH = (
    Path(__file__).resolve().parents[5]
    / "projects"
    / "hipdnn"
    / "plugin_sdk"
    / "data"
    / "gpu_generic_targets.json"
)

_GENERIC_SUFFIX = "-generic"


class GenericTargets:
    """Generic name -> member processor names."""

    def __init__(self, path, generics):
        self._path = Path(path)
        self._generics = generics

    @classmethod
    def load(cls, path):
        path = Path(path)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc["schemaVersion"] != 1:
                raise ValueError(f"unsupported schemaVersion {doc['schemaVersion']!r}")
            generics = {n: tuple(m) for n, m in doc["generics"].items()}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise HkpPackError(f"{path}: unusable generic target table: {exc!r}")
        return cls(path, generics)

    @property
    def path(self):
        return self._path

    def has(self, name):
        return name in self._generics

    def members(self, name):
        """Members of generic @name; KeyError when unlisted."""
        return self._generics[name]

    def names(self):
        return tuple(self._generics)


def is_generic_shaped(name):
    """Name shape only; GenericTargets.has tests membership."""
    return name.endswith(_GENERIC_SUFFIX)


def _strip_features(arch):
    return arch.split(":", 1)[0]


def _prefix_match(device, entry):
    """The packer-side twin of archMatches PREFIX: @entry is @device's base id."""
    return device == entry or device.startswith(entry + ":")


def entry_tier(entry, device, table):
    """Tier of one entry for @device, or None."""
    if is_generic_shaped(entry):
        if table.has(entry) and _strip_features(device) in table.members(entry):
            return TIER_GENERIC
        return None
    if _prefix_match(device, entry):
        return TIER_EXPLICIT
    return None


def list_tier(entries, device, table):
    """Best tier of @entries for @device; None = no match."""
    if not entries:
        return TIER_UNRESTRICTED
    best = None
    for entry in entries:
        tier = entry_tier(entry, device, table)
        if tier == TIER_EXPLICIT:
            return tier
        if tier is not None:
            best = tier
    return best


def expand(entries, table):
    """Device ids @entries admits; None for an empty list."""
    if not entries:
        return None
    devices = set()
    for entry in entries:
        if is_generic_shaped(entry):
            if table.has(entry):
                devices.update(table.members(entry))
        else:
            devices.add(entry)
    return frozenset(devices)


def covers(outer, inner, table):
    """Does @outer admit every device @inner admits?"""
    expanded_outer = expand(outer, table)
    expanded_inner = expand(inner, table)
    if expanded_outer is None or expanded_inner is None:
        return True
    return expanded_inner <= expanded_outer


def compete(a, b, table):
    """Do the lists tie at one tier on some device?"""
    if not a and not b:
        return True
    candidates = set()
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


def admits_target(entries, target, table):
    """Does @entries admit @target?"""
    return list_tier(entries, target, table) is not None
