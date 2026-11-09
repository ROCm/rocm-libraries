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


def load_definitions(directory=None):
    """Load validated definitions sorted by ID from individual JSON files.

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
        required = {"schema_version", "id", "kind"}
        optional = {"release_year"}
        if (
            not isinstance(record, dict)
            or not required <= set(record)
            or set(record) - required - optional
        ):
            raise ValueError(
                f"{path}: expected identity fields and optional release metadata"
            )
        if "release_year" in record:
            validate_year(record["release_year"])
            if record["kind"] != "processor":
                raise ValueError(
                    f"{path}: hardware release years apply only to processors"
                )
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
        records.append(record)
    return tuple(sorted(records, key=lambda record: record["id"]))


def validate_year(year):
    if type(year) is not int or not 1900 <= year <= 9999:
        raise ValueError("release year must be an integer from 1900 to 9999")


def load_catalog(directory=None):
    """Return identity pairs without discarding the validation of release metadata."""
    return tuple(
        (record["id"], record["kind"]) for record in load_definitions(directory)
    )


def architectures(min_year=None, max_year=None, order="name"):
    """Enumerate processors with optional inclusive hardware release-year bounds.

    Unknown years are excluded when either bound is set. Release ordering places
    unknown years last and uses canonical names to break ties.
    """
    for year in (min_year, max_year):
        if year is not None:
            validate_year(year)
    if min_year is not None and max_year is not None and min_year > max_year:
        raise ValueError("minimum release year exceeds maximum")
    if order not in ("name", "release"):
        raise ValueError("order must be name or release")
    records = [record for record in load_definitions() if record["kind"] == "processor"]
    if min_year is not None or max_year is not None:
        records = [
            record
            for record in records
            if "release_year" in record
            and (min_year is None or record["release_year"] >= min_year)
            and (max_year is None or record["release_year"] <= max_year)
        ]
    if order == "release":
        records.sort(
            key=lambda record: (record.get("release_year", 10000), record["id"])
        )
    return tuple(record["id"] for record in records)


def release_year(name):
    """Return a processor's hardware release year, or None when unknown."""
    return next(
        (
            record.get("release_year")
            for record in load_definitions()
            if record["id"] == name
        ),
        None,
    )


def generic_targets():
    """Return registered generic compiler target names; no compatibility is implied."""
    return tuple(name for name, kind in load_catalog() if kind == "generic")


def is_known(name):
    """Exact, case-sensitive identity lookup; feature suffixes are not identities."""
    return any(name == registered for registered, _ in load_catalog())


def catalog_digest(records=None):
    """Fingerprint shared by generated C++/CMake views and the Python package."""
    records = load_definitions() if records is None else records
    canonical = json.dumps(
        records, separators=(",", ":"), ensure_ascii=True, sort_keys=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
