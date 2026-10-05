# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Capture actual descriptor identities and semantic revisions before training."""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

ROLES = ("sort_kernel_catalog", "predict_engine", "predict_applicable_kernels")
_REVISION = re.compile(r"^[0-9]+\.[0-9]+$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")
#: `trained_against` member: what published feature values mean (`FeatureSemantics.hpp`).
FEATURE_SEMANTICS_REVISION = "feature_semantics_revision"
#: The loader reads the revision as int64_t.
_MAX_INT64 = 2**63 - 1


class ProvenanceError(ValueError):
    """A dependency is absent, ambiguous, malformed, or incompatible."""


def descriptor_id(value: object, where: str) -> str:
    if not isinstance(value, str) or not _UUID.fullmatch(value):
        raise ProvenanceError(f"{where}: expected a UUID, got {value!r}")
    return str(uuid.UUID(value))


def revision(value: object, where: str) -> tuple[int, int]:
    if not isinstance(value, str) or not _REVISION.fullmatch(value):
        raise ProvenanceError(
            f"{where}: expected semantic revision '<major>.<minor>', got {value!r}"
        )
    return tuple(int(part) for part in value.split("."))


def _dependency(value: object, where: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"id", "revision"}:
        raise ProvenanceError(f"{where}: requires exactly id and revision")
    identity = descriptor_id(value["id"], f"{where}.id")
    major, minor = revision(value["revision"], f"{where}.revision")
    return {"id": identity, "revision": f"{major}.{minor}"}


def validate_provenance(snapshot: object) -> dict:
    """Validate a recorded snapshot without consulting or inventing dependencies.

    Accepts what the loader accepts (RFC 0019 §4.1): a descriptor set (`ued`/`kmd`/`umd`,
    all or none) and/or a `selector_revision` for engines with no UED, plus an optional
    `feature_semantics_revision` (absent means 1).
    """
    if not isinstance(snapshot, dict):
        raise ProvenanceError("trained_against must be an object")
    unknown = set(snapshot) - {
        "ued",
        "kmd",
        "umd",
        "selector_revision",
        FEATURE_SEMANTICS_REVISION,
    }
    if unknown:
        raise ProvenanceError(f"trained_against has unknown members: {sorted(unknown)}")
    semantics = {}
    if FEATURE_SEMANTICS_REVISION in snapshot:
        recorded_semantics = snapshot[FEATURE_SEMANTICS_REVISION]
        # Strict int: True or 1.0 would compare equal to 1 here but the loader refuses them.
        if (
            isinstance(recorded_semantics, bool)
            or not isinstance(recorded_semantics, int)
            or not 1 <= recorded_semantics <= _MAX_INT64
        ):
            raise ProvenanceError(
                f"trained_against.{FEATURE_SEMANTICS_REVISION} must be an integer >= 1"
            )
        semantics[FEATURE_SEMANTICS_REVISION] = recorded_semantics
    names_descriptor_set = bool({"ued", "kmd", "umd"} & set(snapshot))
    if not names_descriptor_set:
        if "selector_revision" not in snapshot:
            raise ProvenanceError(
                "trained_against must name a descriptor set or a selector_revision"
            )
        revision_text = snapshot["selector_revision"]
        if not isinstance(revision_text, str) or not revision_text:
            raise ProvenanceError(
                "trained_against.selector_revision must be a non-empty string"
            )
        return {"selector_revision": revision_text, **semantics}
    # All three or none: a partial descriptor set is unverifiable.
    if not {"ued", "kmd", "umd"} <= set(snapshot):
        raise ProvenanceError("trained_against requires exactly ued, kmd and umd")
    if not isinstance(snapshot["umd"], list):
        raise ProvenanceError("trained_against.umd must be an array")
    matchers = [_dependency(item, "trained_against.umd") for item in snapshot["umd"]]
    ids = [item["id"] for item in matchers]
    if len(ids) != len(set(ids)):
        raise ProvenanceError("trained_against.umd has duplicate matcher identities")
    recorded = {
        "ued": _dependency(snapshot["ued"], "trained_against.ued"),
        "kmd": _dependency(snapshot["kmd"], "trained_against.kmd"),
        "umd": sorted(matchers, key=lambda item: item["id"]),
    }
    if "selector_revision" in snapshot:
        # A descriptor-backed engine may also record its provider build; the loader
        # checks each independently.
        if (
            not isinstance(snapshot["selector_revision"], str)
            or not snapshot["selector_revision"]
        ):
            raise ProvenanceError(
                "trained_against.selector_revision must be a non-empty string"
            )
        recorded["selector_revision"] = snapshot["selector_revision"]
    return {**recorded, **semantics}


def require_feature_semantics(trained: object, current: int) -> None:
    """Refuse a model whose recorded feature-semantics revision is not `current`.

    No record (or `trained` is None) means revision 1, as in the loader's
    `featureSemanticsMismatch`.
    """
    recorded = (
        1
        if trained is None
        else validate_provenance(trained).get(FEATURE_SEMANTICS_REVISION, 1)
    )
    if recorded != current:
        raise ProvenanceError(
            f"trained_against.{FEATURE_SEMANTICS_REVISION}: model was trained against feature "
            f"semantics revision {recorded}, the feature evaluator computes revision {current}"
        )


def record_feature_semantics(snapshot: object, current: int) -> dict:
    """`snapshot` with the evaluator's feature-semantics revision recorded.

    An existing record must already equal `current`; it is never overwritten.
    """
    recorded = validate_provenance(snapshot)
    if FEATURE_SEMANTICS_REVISION in recorded:
        require_feature_semantics(recorded, current)
    return {**recorded, FEATURE_SEMANTICS_REVISION: current}


def compare_provenance(
    trained: object, actual: object, *, foreign_matchers=frozenset()
) -> None:
    """Existing dependencies must retain identity/major and not regress minor.

    Extra pack matchers are fine. Recorded matchers in `foreign_matchers` (owned only by
    other arches' packs) are ignored, as in `DescriptorLoader.hpp`. Selector revisions are
    opaque and must match exactly.
    """
    trained = validate_provenance(trained)
    actual = validate_provenance(actual)
    recorded_revision = trained.get("selector_revision")
    # Only when `actual` has one: snapshots built from descriptors never carry the
    # provider build.
    if (
        recorded_revision is not None
        and "selector_revision" in actual
        and actual["selector_revision"] != recorded_revision
    ):
        raise ProvenanceError(
            f"trained_against.selector_revision: model records {recorded_revision!r}, "
            f"the provider reports {actual.get('selector_revision')!r}"
        )
    if "ued" not in trained:
        return
    if "ued" not in actual:
        raise ProvenanceError(
            "trained_against names a descriptor set the engine does not have"
        )
    for kind in ("ued", "kmd", "umd"):
        recorded = trained[kind] if kind == "umd" else [trained[kind]]
        available = actual[kind] if kind == "umd" else [actual[kind]]
        by_id = {item["id"]: item for item in available}
        for dependency in recorded:
            identity = dependency["id"]
            current = by_id.get(identity)
            if current is None and kind == "umd" and identity in foreign_matchers:
                continue
            if current is None:
                raise ProvenanceError(
                    f"trained_against.{kind}: dependency {identity} is missing or not owned by this engine/architecture"
                )
            expected = revision(dependency["revision"], kind)
            found = revision(current["revision"], kind)
            if found[0] != expected[0] or found[1] < expected[1]:
                raise ProvenanceError(
                    f"trained_against.{kind}: {identity} revision {current['revision']} is incompatible with trained revision {dependency['revision']}"
                )


def load_descriptor_tree(
    descriptor_tree: Path,
) -> dict[str, dict[str, tuple[Path, dict]]]:
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
                raise ProvenanceError(
                    f"cannot read descriptor {path}: {error}"
                ) from error
            if not isinstance(doc, dict):
                raise ProvenanceError(f"{path}: descriptor must be an object")
            identity = descriptor_id(doc.get("id"), str(path))
            if identity in identities:
                raise ProvenanceError(
                    f"duplicate/conflicting descriptor {identity}: {identities[identity]} and {path}"
                )
            identities[identity] = path
            if doc.get("version") != "1.0":
                raise ProvenanceError(
                    f"{path}: unsupported file-format version {doc.get('version')!r}; expected 1.0"
                )
            if kind != "kdp":
                revision(doc.get("revision", "1.0"), str(path))
            entries[identity] = (path, doc)
    return result


def select_engine(index: dict, engine: str | None) -> tuple[Path, dict]:
    entries = list(index["ued"].values())
    matches = (
        entries
        if engine is None
        else [
            entry
            for entry in entries
            if entry[1].get("name") == engine or entry[1]["id"] == engine
        ]
    )
    if len(matches) != 1:
        raise ProvenanceError(
            f"expected one UED for --engine {engine!r}, found {len(matches)}; select an unambiguous engine"
        )
    return matches[0]


def _pack_matchers(index: dict, ued_id: str, arch: str | None) -> set[str]:
    """Matcher ids of the engine's packs that serve `arch` (every pack for None/default).

    A pack with no `arch` list serves every architecture, exactly as the loader reads it.
    """
    matchers = set()
    for path, pack in index["kdp"].values():
        owner = descriptor_id(pack.get("engine"), f"{path}.engine")
        if owner != ued_id:
            continue
        arches = pack.get("arch", [])
        if not isinstance(arches, list) or any(
            not isinstance(item, str) for item in arches
        ):
            raise ProvenanceError(f"{path}.arch must be an array of strings")
        if arch not in (None, "default") and arches and arch not in arches:
            continue
        refs = pack.get("matchers", [])
        if not isinstance(refs, list):
            raise ProvenanceError(f"{path}.matchers must be an array")
        for identity in refs:
            matchers.add(descriptor_id(identity, f"{path}.matchers"))
    return matchers


def foreign_matcher_ids(index: dict, ued: dict, arch: str | None) -> frozenset[str]:
    """Matchers owned only by the engine's packs for architectures other than `arch`."""
    ued_id = descriptor_id(ued.get("id"), "UED")
    return frozenset(
        _pack_matchers(index, ued_id, None) - _pack_matchers(index, ued_id, arch)
    )


def provenance_for_engine(index: dict, ued: dict, arch: str | None = None) -> dict:
    def resolve(kind: str, identity: object) -> dict:
        identity = descriptor_id(identity, kind)
        entry = index[kind].get(identity)
        if entry is None:
            raise ProvenanceError(f"missing {kind.upper()} dependency {identity}")
        return {"id": identity, "revision": entry[1].get("revision", "1.0")}

    ued_id = descriptor_id(ued.get("id"), "UED")
    # Take the revision from the supplied UED so promote can check a planned
    # knob-removal revision.
    ued_dependency = resolve("ued", ued_id)
    ued_dependency["revision"] = ued.get("revision", "1.0")
    matchers = _pack_matchers(index, ued_id, arch)
    return validate_provenance(
        {
            "ued": ued_dependency,
            "kmd": resolve("kmd", ued.get("metadata")),
            "umd": [resolve("umd", identity) for identity in sorted(matchers)],
        }
    )


def snapshot_provenance(
    descriptor_tree: Path, engine: str | None = None, arch: str | None = None
) -> dict:
    """Snapshot the selected engine, its KMD, and all relevant pack matchers."""
    index = load_descriptor_tree(descriptor_tree)
    _, ued = select_engine(index, engine)
    return provenance_for_engine(index, ued, arch)
