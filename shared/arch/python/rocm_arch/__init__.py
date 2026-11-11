# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Small, device-independent GPU target catalog (prototype, not a support list)."""

import hashlib
import json
import re
from pathlib import Path

_DEFINITIONS = Path(__file__).parent / "architectures"
_ID_PATTERN = re.compile(r"gfx[0-9][0-9a-z]*(?:-[0-9a-z]+)*")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_catalog(directory=None):
    """Load validated (id, kind) records, sorted by ID, from individual JSON files.

    An alternate directory supports generation and catalog-authoring checks.
    There is deliberately no implicit device or compiler filtering.
    """
    directory = _DEFINITIONS if directory is None else Path(directory)
    files = sorted(directory.glob("*.json"))
    if not files:
        raise ValueError(f"empty architecture catalog: {directory}")
    records = []
    for path in files:
        record = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
        if not isinstance(record, dict) or set(record) != {
            "schema_version",
            "id",
            "kind",
        }:
            raise ValueError(f"{path}: expected schema_version, id, and kind")
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ValueError(f"{path}: unsupported schema version")
        name, kind = record["id"], record["kind"]
        if not isinstance(name, str) or not _ID_PATTERN.fullmatch(name):
            raise ValueError(f"{path}: invalid architecture ID")
        if path.stem != name:
            raise ValueError(f"{path}: filename must match architecture ID")
        if kind not in ("processor", "generic"):
            raise ValueError(f"{path}: invalid architecture kind")
        if (kind == "generic") != name.endswith("-generic"):
            raise ValueError(f"{path}: generic ID and kind must agree")
        records.append((name, kind))
    return tuple(sorted(records))


def architectures():
    """Return the immutable list of registered processor names, including steppings."""
    return tuple(name for name, kind in load_catalog() if kind == "processor")


def generic_targets():
    """Return registered generic compiler target names; no compatibility is implied."""
    return tuple(name for name, kind in load_catalog() if kind == "generic")


def is_known(name):
    """Exact, case-sensitive identity lookup; feature suffixes are not identities."""
    return any(name == registered for registered, _ in load_catalog())


def catalog_digest(records=None):
    """Fingerprint shared by generated C++/CMake views and the Python package."""
    records = load_catalog() if records is None else records
    canonical = json.dumps(records, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
