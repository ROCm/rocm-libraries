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
_SLOT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
#: Descriptor file-format versions the tooling reads; 1.1 adds the RFC 0021 slot members.
SUPPORTED_FILE_VERSIONS = ("1.0", "1.1")
#: UED and KDP members that exist only from file-format 1.1.
_SLOT_MEMBERS = frozenset({"slots", "slot", "shared_workspace"})
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


def slot_name(value: object, where: str) -> str:
    """A valid slot name (RFC 0021 §4): `^[a-z][a-z0-9_]*$`, never the reserved `global`."""
    if (
        not isinstance(value, str)
        or not _SLOT_NAME.fullmatch(value)
        or value == "global"
    ):
        raise ProvenanceError(
            f"{where} {value!r} is not a valid slot name (^[a-z][a-z0-9_]*$, never 'global')"
        )
    return value


def _slot_dependencies(value: object, where: str) -> list[dict]:
    """`trained_against.kmds`: a non-empty array of `{slot, id, revision}`, one per slot."""
    if not isinstance(value, list) or not value:
        raise ProvenanceError(f"{where} must be a non-empty array")
    entries = []
    seen = set()
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"slot", "id", "revision"}:
            raise ProvenanceError(
                f"{where} entries require exactly slot, id and revision"
            )
        name = slot_name(entry["slot"], f"{where} slot")
        if name in seen:
            raise ProvenanceError(f"{where} names slot {name!r} twice")
        seen.add(name)
        entries.append(
            {
                "slot": name,
                **_dependency({k: entry[k] for k in ("id", "revision")}, where),
            }
        )
    return entries


def validate_provenance(snapshot: object) -> dict:
    """Validate a recorded snapshot without consulting or inventing dependencies.

    Accepts what the loader accepts (RFC 0019 §4.1): a descriptor set (`ued`/`kmd`/`umd`,
    all or none) and/or a `selector_revision` for engines with no UED, plus an optional
    `feature_semantics_revision` (absent means 1). RFC 0021 §9 adds two slot forms: `slot`
    beside `kmd` (one slot's ranker), and `kmds: [{slot, id, revision}]` in place of `kmd`
    (a slotted engine's `predict_engine` model).
    """
    if not isinstance(snapshot, dict):
        raise ProvenanceError("trained_against must be an object")
    unknown = set(snapshot) - {
        "ued",
        "kmd",
        "kmds",
        "slot",
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
    names_descriptor_set = bool({"ued", "kmd", "kmds", "umd"} & set(snapshot))
    if not names_descriptor_set and "selector_revision" not in snapshot:
        raise ProvenanceError(
            "trained_against must name a descriptor set or a selector_revision"
        )
    if "selector_revision" in snapshot:
        revision_text = snapshot["selector_revision"]
        if not isinstance(revision_text, str) or not revision_text:
            raise ProvenanceError(
                "trained_against.selector_revision must be a non-empty string"
            )
    if "slot" in snapshot and "kmd" not in snapshot:
        raise ProvenanceError("trained_against.slot requires kmd, and excludes kmds")
    if "kmd" in snapshot and "kmds" in snapshot:
        raise ProvenanceError("trained_against names both kmd and kmds")
    if not names_descriptor_set:
        return {"selector_revision": snapshot["selector_revision"], **semantics}
    # All three or none: a partial descriptor set is unverifiable. `kmds` stands in for `kmd`.
    kmd_key = "kmds" if "kmds" in snapshot else "kmd"
    if not {"ued", kmd_key, "umd"} <= set(snapshot):
        raise ProvenanceError(
            f"trained_against requires exactly ued, {kmd_key} and umd"
        )
    if not isinstance(snapshot["umd"], list):
        raise ProvenanceError("trained_against.umd must be an array")
    matchers = [_dependency(item, "trained_against.umd") for item in snapshot["umd"]]
    ids = [item["id"] for item in matchers]
    if len(ids) != len(set(ids)):
        raise ProvenanceError("trained_against.umd has duplicate matcher identities")
    recorded = {"ued": _dependency(snapshot["ued"], "trained_against.ued")}
    if kmd_key == "kmds":
        recorded["kmds"] = _slot_dependencies(snapshot["kmds"], "trained_against.kmds")
    else:
        recorded["kmd"] = _dependency(snapshot["kmd"], "trained_against.kmd")
        if "slot" in snapshot:
            recorded["slot"] = slot_name(snapshot["slot"], "trained_against.slot")
    recorded["umd"] = sorted(matchers, key=lambda item: item["id"])
    if "selector_revision" in snapshot:
        # A descriptor-backed engine may also record its selector revision; the loader
        # checks it independently of the descriptor revisions.
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


def _check_dependency(kind: str, dependency: dict, by_id: dict, foreign=frozenset()):
    """One recorded dependency must still exist (or be another arch's) and not regress."""
    identity = dependency["id"]
    current = by_id.get(identity)
    if current is None and kind == "umd" and identity in foreign:
        return
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


def _compare_kmds(trained: dict, actual: dict) -> None:
    """The KMD half of the comparison, in the unslotted, slot-ranker and engine-level forms.

    Mirrors `slottedProvenanceError` (DescriptorLoader.hpp): a slot ranker names its own
    slot's KMD and the `slot` it was trained for (required); an engine-level model of a
    slotted engine names every slot's KMD in `kmds`; an unslotted engine satisfies neither
    slot form.
    """
    if "kmds" in trained:
        if "kmds" not in actual:
            if "slot" in actual:
                raise ProvenanceError(
                    f"trained_against names kmds, but the ranker of slot {actual['slot']!r} "
                    "must name its own slot's kmd"
                )
            raise ProvenanceError(
                "trained_against names slot provenance ('kmds'), which an engine that "
                "declares no slots cannot satisfy"
            )
        declared = {entry["slot"]: entry for entry in actual["kmds"]}
        for entry in trained["kmds"]:
            current = declared.get(entry["slot"])
            if current is None:
                raise ProvenanceError(
                    f"trained_against kmds names slot {entry['slot']!r}, which the engine "
                    "does not declare"
                )
            _check_dependency("kmds", entry, {current["id"]: current})
        named = {entry["slot"] for entry in trained["kmds"]}
        for slot in declared:
            if slot not in named:
                raise ProvenanceError(
                    f"trained_against kmds does not name slot {slot!r}"
                )
        return
    if "kmds" in actual:
        if "slot" in trained:
            raise ProvenanceError(
                "trained_against names a single slot, but a slotted engine's predict_engine "
                "model must name every slot's KMD in kmds"
            )
        raise ProvenanceError(
            "trained_against names kmd, but a slotted engine's predict_engine model must "
            "name every slot's KMD in kmds"
        )
    if "slot" in actual and "slot" not in trained:
        raise ProvenanceError(
            f"trained_against names no slot, but the ranker of slot {actual['slot']!r} "
            "must name the slot it was trained for"
        )
    if "slot" in trained:
        if "slot" not in actual:
            raise ProvenanceError(
                "trained_against names slot provenance ('slot'), which an engine that "
                "declares no slots cannot satisfy"
            )
        if trained["slot"] != actual["slot"]:
            raise ProvenanceError(
                f"trained_against names slot {trained['slot']!r}, but the ranker serves "
                f"slot {actual['slot']!r}"
            )
    _check_dependency("kmd", trained["kmd"], {actual["kmd"]["id"]: actual["kmd"]})


def compare_provenance(
    trained: object, actual: object, *, foreign_matchers=frozenset()
) -> None:
    """Existing dependencies must retain identity/major and not regress minor.

    As `DescriptorLoader.hpp`'s provenanceError decides it: a recorded matcher in
    `foreign_matchers` (owned only by other arches' packs) is ignored, and every matcher in
    `actual` -- the arch's own -- must be recorded, since the runtime disables a model that
    never saw one of them. The same matcher rule holds for slotted engines (`actual` is then
    scoped to the ranker's slot, or to every slot for an engine-level model), as in
    `umdProvenanceError`. Selector revisions are opaque and must match exactly. Slot forms
    compare as `_compare_kmds` describes.
    """
    trained = validate_provenance(trained)
    actual = validate_provenance(actual)
    recorded_revision = trained.get("selector_revision")
    # Only when `actual` has one: a snapshot built from descriptors cannot name the
    # provider's selector revision.
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
    _check_dependency("ued", trained["ued"], {actual["ued"]["id"]: actual["ued"]})
    _compare_kmds(trained, actual)
    by_id = {item["id"]: item for item in actual["umd"]}
    for dependency in trained["umd"]:
        _check_dependency("umd", dependency, by_id, foreign_matchers)
    recorded_matchers = {item["id"] for item in trained["umd"]}
    for matcher in actual["umd"]:
        if matcher["id"] not in recorded_matchers:
            raise ProvenanceError(
                f"trained_against.umd: matcher {matcher['id']} serves this architecture but "
                "the model was not trained with it, so the runtime would disable the model; "
                "retrain against the current descriptor set"
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
            # Only the kinds that carry slot members have a 1.1; a file that carries them
            # at 1.0 is the loader's refusal too (RFC 0021 §9).
            versions = SUPPORTED_FILE_VERSIONS if kind in ("ued", "kdp") else ("1.0",)
            if doc.get("version") not in versions:
                raise ProvenanceError(
                    f"{path}: unsupported file-format version {doc.get('version')!r}; "
                    f"expected one of {', '.join(versions)}"
                )
            slotted = sorted(_SLOT_MEMBERS & set(doc))
            if doc["version"] == "1.0" and slotted:
                raise ProvenanceError(
                    f"{path}: {', '.join(slotted)} require file-format version 1.1"
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


def _pack_matchers(
    index: dict, ued_id: str, arch: str | None, slot: str | None = None
) -> set[str]:
    """Matcher ids of the engine's packs that serve `arch` (every pack for None/default).

    A pack with no `arch` list serves every architecture, exactly as the loader reads it.
    With `slot`, only the packs of that slot of a slotted engine count.
    """
    matchers = set()
    for path, pack in index["kdp"].values():
        owner = descriptor_id(pack.get("engine"), f"{path}.engine")
        if owner != ued_id:
            continue
        if slot is not None and pack.get("slot") != slot:
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


def foreign_matcher_ids(
    index: dict, ued: dict, arch: str | None, slot: str | None = None
) -> frozenset[str]:
    """Matchers owned only by the engine's packs for architectures other than `arch`
    (of `slot`'s packs, for a slot ranker)."""
    ued_id = descriptor_id(ued.get("id"), "UED")
    return frozenset(
        _pack_matchers(index, ued_id, None, slot)
        - _pack_matchers(index, ued_id, arch, slot)
    )


def declared_slots(ued: dict) -> list[dict] | None:
    """The UED's `slots` entries, or None for an unslotted engine."""
    slots = ued.get("slots")
    if slots is None:
        return None
    if not isinstance(slots, list) or not slots:
        raise ProvenanceError("UED slots must be a non-empty array")
    names = []
    for entry in slots:
        if not isinstance(entry, dict):
            raise ProvenanceError("UED slots entries must be objects")
        names.append(slot_name(entry.get("name"), "UED slots[].name"))
    if len(names) != len(set(names)):
        raise ProvenanceError("UED slots names must be unique")
    return slots


def provenance_for_engine(
    index: dict, ued: dict, arch: str | None = None, slot: str | None = None
) -> dict:
    """The snapshot an engine-bound model is compared against.

    An unslotted engine has one form. A slotted engine has two: a slot ranker's
    (`slot` given: that slot's KMD and packs only) and the engine-level `kmds` form
    (`slot` None: every slot's KMD, every pack).
    """

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
    slots = declared_slots(ued)
    if slots is None:
        if slot is not None:
            raise ProvenanceError(
                f"engine {ued.get('name', ued_id)!r} declares no slots, so it has no "
                f"slot {slot!r}"
            )
        matchers = _pack_matchers(index, ued_id, arch)
        return validate_provenance(
            {
                "ued": ued_dependency,
                "kmd": resolve("kmd", ued.get("metadata")),
                "umd": [resolve("umd", identity) for identity in sorted(matchers)],
            }
        )
    matchers = _pack_matchers(index, ued_id, arch, slot)
    umd = [resolve("umd", identity) for identity in sorted(matchers)]
    if slot is not None:
        entry = next((item for item in slots if item["name"] == slot), None)
        if entry is None:
            raise ProvenanceError(
                f"engine {ued.get('name', ued_id)!r} declares no slot {slot!r}"
            )
        return validate_provenance(
            {
                "ued": ued_dependency,
                "kmd": resolve("kmd", entry.get("metadata")),
                "slot": slot,
                "umd": umd,
            }
        )
    return validate_provenance(
        {
            "ued": ued_dependency,
            "kmds": [
                {"slot": item["name"], **resolve("kmd", item.get("metadata"))}
                for item in slots
            ],
            "umd": umd,
        }
    )


def snapshot_provenance(
    descriptor_tree: Path,
    engine: str | None = None,
    arch: str | None = None,
    slot: str | None = None,
) -> dict:
    """Snapshot the selected engine, its KMD, and all relevant pack matchers.

    For a slotted engine, `slot` snapshots one slot's ranker (that slot's KMD and
    packs); without it the snapshot is the engine-level one (every slot's KMD, every pack).
    """
    index = load_descriptor_tree(descriptor_tree)
    _, ued = select_engine(index, engine)
    return provenance_for_engine(index, ued, arch, slot)
