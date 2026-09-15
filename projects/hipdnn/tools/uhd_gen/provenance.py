# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Capture actual descriptor identities and semantic revisions before training."""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

ROLES = ("sort_kernel_catalog", "predict_engine_tflops", "predict_applicable_kernels")
_REVISION = re.compile(r"^[0-9]+\.[0-9]+$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")


class ProvenanceError(ValueError):
    """A dependency is absent, ambiguous, malformed, or incompatible."""


def descriptor_id(value: object, where: str) -> str:
    if not isinstance(value, str) or not _UUID.fullmatch(value):
        raise ProvenanceError(f"{where}: expected a UUID, got {value!r}")
    return str(uuid.UUID(value))


def revision(value: object, where: str) -> tuple[int, int]:
    if not isinstance(value, str) or not _REVISION.fullmatch(value):
        raise ProvenanceError(f"{where}: expected semantic revision '<major>.<minor>', got {value!r}")
    return tuple(int(part) for part in value.split("."))


def _dependency(value: object, where: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"id", "revision"}:
        raise ProvenanceError(f"{where}: requires exactly id and revision")
    identity = descriptor_id(value["id"], f"{where}.id")
    major, minor = revision(value["revision"], f"{where}.revision")
    return {"id": identity, "revision": f"{major}.{minor}"}


def validate_provenance(snapshot: object) -> dict:
    """Validate a recorded snapshot without consulting or inventing dependencies.

    RFC 0019 Section 4.1: a UHD names the descriptor set it was generated against, and
    only that. Every role -- L1 included -- is reached through the owning UED's role map,
    so there is no engine-selector variant to record.
    """
    if not isinstance(snapshot, dict) or set(snapshot) != {"ued", "kmd", "umd"}:
        raise ProvenanceError("trained_against requires exactly ued, kmd and umd")
    if not isinstance(snapshot["umd"], list):
        raise ProvenanceError("trained_against.umd must be an array")
    matchers = [_dependency(item, "trained_against.umd") for item in snapshot["umd"]]
    ids = [item["id"] for item in matchers]
    if len(ids) != len(set(ids)):
        raise ProvenanceError("trained_against.umd has duplicate matcher identities")
    return {
        "ued": _dependency(snapshot["ued"], "trained_against.ued"),
        "kmd": _dependency(snapshot["kmd"], "trained_against.kmd"),
        "umd": sorted(matchers, key=lambda item: item["id"]),
    }


def compare_provenance(trained: object, actual: object) -> None:
    """Existing dependencies must retain identity/major and not regress minor.

    Additional pack matchers are coverage changes, not contract breakages.
    """
    trained = validate_provenance(trained)
    actual = validate_provenance(actual)
    for kind in ("ued", "kmd", "umd"):
        recorded = trained[kind] if kind == "umd" else [trained[kind]]
        available = actual[kind] if kind == "umd" else [actual[kind]]
        by_id = {item["id"]: item for item in available}
        for dependency in recorded:
            identity = dependency["id"]
            current = by_id.get(identity)
            if current is None:
                raise ProvenanceError(f"trained_against.{kind}: dependency {identity} is missing or not owned by this engine/architecture")
            expected = revision(dependency["revision"], kind)
            found = revision(current["revision"], kind)
            if found[0] != expected[0] or found[1] < expected[1]:
                raise ProvenanceError(f"trained_against.{kind}: {identity} revision {current['revision']} is incompatible with trained revision {dependency['revision']}")


def load_descriptor_tree(descriptor_tree: Path) -> dict[str, dict[str, tuple[Path, dict]]]:
    """Index dependency descriptors, rejecting even identical duplicate definitions."""
    root = Path(descriptor_tree)
    if not root.is_dir():
        raise ProvenanceError(f"descriptor tree {root} is not a directory")
    result = {kind: {} for kind in ("ued", "kmd", "umd", "kdp")}
    identities = {}
    for kind, entries in result.items():
        for path in sorted(root.rglob(f"*.{kind}.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ProvenanceError(f"cannot read descriptor {path}: {error}") from error
            if not isinstance(doc, dict):
                raise ProvenanceError(f"{path}: descriptor must be an object")
            identity = descriptor_id(doc.get("id"), str(path))
            if identity in identities:
                raise ProvenanceError(f"duplicate/conflicting descriptor {identity}: {identities[identity]} and {path}")
            identities[identity] = path
            if doc.get("version") != "1.0":
                raise ProvenanceError(f"{path}: unsupported file-format version {doc.get('version')!r}; expected 1.0")
            if kind != "kdp":
                revision(doc.get("revision", "1.0"), str(path))
            entries[identity] = (path, doc)
    return result


def select_engine(index: dict, engine: str | None) -> tuple[Path, dict]:
    entries = list(index["ued"].values())
    matches = entries if engine is None else [entry for entry in entries if entry[1].get("name") == engine or entry[1]["id"] == engine]
    if len(matches) != 1:
        raise ProvenanceError(f"expected one UED for --engine {engine!r}, found {len(matches)}; select an unambiguous engine")
    return matches[0]


def provenance_for_engine(index: dict, ued: dict, arch: str | None = None) -> dict:
    def resolve(kind: str, identity: object) -> dict:
        identity = descriptor_id(identity, kind)
        entry = index[kind].get(identity)
        if entry is None:
            raise ProvenanceError(f"missing {kind.upper()} dependency {identity}")
        return {"id": identity, "revision": entry[1].get("revision", "1.0")}

    ued_id = descriptor_id(ued.get("id"), "UED")
    # Use the supplied UED revision so promotion can evaluate an explicitly planned
    # knob-removal revision, without modifying the training snapshot.
    ued_dependency = resolve("ued", ued_id)
    ued_dependency["revision"] = ued.get("revision", "1.0")
    matchers = set()
    for path, pack in index["kdp"].values():
        owner = descriptor_id(pack.get("engine"), f"{path}.engine")
        if owner != ued_id:
            continue
        arches = pack.get("arch", [])
        if not isinstance(arches, list) or any(not isinstance(item, str) for item in arches):
            raise ProvenanceError(f"{path}.arch must be an array of strings")
        if arch not in (None, "default") and arches and arch not in arches:
            continue
        refs = pack.get("matchers", [])
        if not isinstance(refs, list):
            raise ProvenanceError(f"{path}.matchers must be an array")
        for identity in refs:
            matchers.add(descriptor_id(identity, f"{path}.matchers"))
    return validate_provenance({
        "ued": ued_dependency,
        "kmd": resolve("kmd", ued.get("metadata")),
        "umd": [resolve("umd", identity) for identity in sorted(matchers)],
    })


def snapshot_provenance(descriptor_tree: Path, engine: str | None = None, arch: str | None = None) -> dict:
    """Snapshot the selected engine, its KMD, and all relevant pack matchers."""
    index = load_descriptor_tree(descriptor_tree)
    _, ued = select_engine(index, engine)
    return provenance_for_engine(index, ued, arch)
