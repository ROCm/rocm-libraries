# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Slotted engines (RFC 0021) at pack time.

The loader (`DescriptorLoader.hpp`, `UhdParser.hpp`) is the authority; the packer
mirrors the statically decidable rules so a bad slotted bundle fails the pack,
not the load. `fixtures/slotted/` is a loader-valid slotted engine: three slots
(`prep`, `dkdv`, `dq`) with a KMD and a dispatch descriptor each, metric-less
rankers on two, an engine estimate naming every slot's KMD, and a shared workspace
token. Each rule below breaks one thing in a copy of it.
"""

from __future__ import annotations

import copy
import hashlib
import gzip
import json
import shutil
import sys
from pathlib import Path

import pytest

from hkp_pack import desk_check, descriptor_context
from hkp_pack.descriptors import load_flat_input
from hkp_pack.errors import HkpPackError
from pack_helpers import GENERIC_TARGETS, run_pipeline

pytestmark = pytest.mark.quick

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "slotted"
MULTI_LAUNCH_FIXTURE = FIXTURE.parent / "slotted_multi_launch"
ENGINE = "hipkernel:SlottedFixture"
UED_ID = "d5a00000-0000-4000-8000-000000000001"
KMD_IDS = {
    "prep": "d5a00000-0000-4000-8000-000000000002",
    "dkdv": "d5a00000-0000-4000-8000-000000000003",
    "dq": "d5a00000-0000-4000-8000-000000000004",
}
DKDV_KERNELS = {
    "d5a00000-0000-4000-8000-000000000021",
    "d5a00000-0000-4000-8000-000000000022",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, doc: dict) -> None:
    # Bytes, not text mode: a Windows text-mode write turns every "\n" into "\r\n".
    path.write_bytes((json.dumps(doc, indent=2) + "\n").encode("utf-8"))


def _files(folder: Path) -> dict:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def _root(tmp_path: Path, **edits) -> Path:
    """A copy of the slotted fixture; `edits` maps a file stem to a function
    editing that document in place (a stem such as `fixture.ued`)."""
    root = tmp_path / "src"
    shutil.copytree(FIXTURE, root)
    for stem, edit in edits.items():
        path = root / f"{stem}.json"
        doc = _read(path)
        edit(doc)
        _write(path, doc)
    return root


def _load(root: Path):
    return load_flat_input(root, GENERIC_TARGETS, log=lambda *_: None)


def _pack(root: Path, tmp_path: Path, rocm_kpack_dir, arch="gfx942"):
    run_pipeline(
        source_root=root,
        arches=[arch],
        out_root=tmp_path / "out",
        hipcc="hipcc-not-invoked",
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
        source_label="slotted-test",
    )
    return tmp_path / "out" / arch


# --- A valid slotted engine packs, and everything it needs ships ------------------
def test_slotted_engine_packs_with_every_slot_kmd_and_ranker(tmp_path, rocm_kpack_dir):
    root = _root(tmp_path)
    # Unreferenced generics are pruned as ever: an unreachable slot KMD or ranker
    # must show up as absent from the shard, not be hidden by a copy-everything pack.
    _write(
        root / "orphan.kmd.json",
        {
            "version": "1.0",
            "id": "d5a00000-0000-4000-8000-0000000000f1",
            "name": "unreferenced",
            "fields": [{"name": "dtype", "type": "string"}],
        },
    )
    _write(
        root / "orphan.uhd.json",
        {
            "version": "1.0",
            "id": "d5a00000-0000-4000-8000-0000000000f2",
            "name": "unreferenced",
            "adapter": "static_order",
            "static_order": {},
        },
    )

    shard = _files(_pack(root, tmp_path, rocm_kpack_dir))

    generics = {
        name
        for name in shard
        if name.endswith(
            (".kmd.json", ".uhd.json", ".ued.json", ".umd.json", ".udd.json")
        )
    }
    assert generics == {
        "prep.kmd.json",
        "dkdv.kmd.json",
        "dq.kmd.json",
        "dkdv.uhd.json",
        "dq.uhd.json",
        "predict_engine.uhd.json",
        "fixture.ued.json",
        "kernel_dtype_matches_graph.umd.json",
        "prep.udd.json",
        "dkdv.udd.json",
        "dq.udd.json",
    }
    for name in generics:
        assert shard[name] == (root / name).read_bytes()
    kdps = {
        name: json.loads(shard[name]) for name in shard if name.endswith(".kdp.json")
    }
    assert {name: doc["slot"] for name, doc in kdps.items()} == {
        "prep.kdp.json": "prep",
        "dkdv.kdp.json": "dkdv",
        "dq.kdp.json": "dq",
    }
    assert {doc["version"] for doc in kdps.values()} == {"1.1"}
    # One dispatch descriptor per slot (RFC 0021 §4): packs of different slots may
    # launch different sets, so none shares a dispatch symbol.
    assert len({doc["dispatch"] for doc in kdps.values()}) == 3


def test_desk_check_resolves_each_pack_to_its_slot_kmd(tmp_path, rocm_kpack_dir):
    shard = _pack(_root(tmp_path), tmp_path, rocm_kpack_dir)

    for tree in (FIXTURE, shard):
        bundles = descriptor_context.resolve_bundles(
            descriptor_context.Index(str(tree))
        )
        resolved = {
            Path(bundle.kdp_path).name: (bundle.kdp_doc["slot"], bundle.kmd["id"])
            for bundle in bundles
        }
        assert resolved == {
            f"{slot}.kdp.json": (slot, kmd) for slot, kmd in KMD_IDS.items()
        }
        assert {bundle.engine["id"] for bundle in bundles} == {UED_ID}


def test_a_slot_pack_with_sources_reads_against_its_slot_kmd_launch_by_launch():
    # The multi-launch slot's pack resolves to its slot's KMD, and the desk check
    # reads each of its launches rather than a `kernel_source` it does not carry.
    bundles = descriptor_context.resolve_bundles(
        descriptor_context.Index(str(MULTI_LAUNCH_FIXTURE))
    )
    by_slot = {bundle.kdp_doc["slot"]: bundle for bundle in bundles}

    assert set(by_slot) == {"prep", "dkdv", "dq"}
    assert by_slot["dkdv"].kmd["name"] == "dkdv variant fields"
    assert by_slot["dkdv"].kmd["id"] != by_slot["prep"].kmd["id"]
    (entry,) = by_slot["dkdv"].entries
    assert set(desk_check._launch_sources(entry.ukd)) == {"decode", "merge"}
    (single,) = by_slot["prep"].entries
    assert set(desk_check._launch_sources(single.ukd)) == {None}


def test_desk_check_refuses_a_pack_naming_an_undeclared_slot(tmp_path):
    root = _root(tmp_path, **{"dq.kdp": lambda doc: doc.update(slot="nope")})

    with pytest.raises(
        HkpPackError, match=r"names slot 'nope', which engine .* does not"
    ):
        descriptor_context.resolve_bundles(descriptor_context.Index(str(root)))


def test_compiler_completes_each_pack_against_its_slot_kmd(tmp_path, rocm_kpack_dir):
    # The dq kernels fill `block_n`, which dq's KMD declares and dkdv's does not.
    # Naming dkdv instead is a declared slot, so validation passes and the
    # compiler's completion is what refuses it.
    root = _root(tmp_path, **{"dq.kdp": lambda doc: doc.update(slot="dkdv")})
    _load(root)

    with pytest.raises(HkpPackError, match="undeclared metadata"):
        _pack(root, tmp_path, rocm_kpack_dir)


def test_a_standalone_kernel_may_serve_two_packs_of_one_slot(tmp_path):
    root = _shared_kernel_root(tmp_path, ["dkdv", "dkdv"])

    assert len(_load(root).kdps()) == 5


# --- Each rule, broken once ------------------------------------------------------
def _ued(edit):
    return {"fixture.ued": edit}


def _slots(edit):
    return _ued(lambda doc: edit(doc["slots"]))


def _rename(index, name):
    return _slots(lambda slots: slots[index].update(name=name))


def _beside_slots(member, value):
    return _ued(lambda doc: doc.update({member: value}))


def _unslotted(doc):
    """The same engine without `slots`: its first slot's KMD becomes the engine's."""
    del doc["slots"]
    doc["metadata"] = KMD_IDS["prep"]


def _without(key):
    return lambda doc: doc.pop(key)


def _ranker_trained(**changes):
    def edit(doc):
        trained = doc["trained_against"]
        for key, value in changes.items():
            if value is None:
                trained.pop(key, None)
            else:
                trained[key] = value

    return edit


def _estimate_kmds(edit):
    return lambda doc: edit(doc["trained_against"]["kmds"])


def _shared_kernel_root(tmp_path, slots):
    """The fixture plus, for each named slot, one more pack of that slot listing
    the same standalone kernel, whose metadata fits every slot's KMD."""
    kernel = "d5a00000-0000-4000-8000-0000000000e1"
    root = _root(tmp_path)
    _write(
        root / "shared.ukd.json",
        {
            "version": "1.0",
            "id": kernel,
            "name": "shared",
            "kernel_source": {
                "kind": "embedded_source",
                "source_file": "Fixture.cpp",
                "entry_point": "Fixture",
            },
            "metadata": {"dtype": "FLOAT"},
            "priority": 0,
        },
    )
    for index, slot in enumerate(slots):
        pack = _read(root / f"{slot}.kdp.json")
        pack.update(id=f"d5a00000-0000-4000-8000-0000000000e{index + 2}")
        pack.update(name=f"hipkernel:shared_{index}", kernelDescriptors=[kernel])
        _write(root / f"shared_{index}.kdp.json", pack)
    return root


def _estimate_one_kmd(doc):
    trained = doc["trained_against"]
    entry = trained.pop("kmds")[0]
    entry.pop("slot")
    trained["kmd"] = entry


NEGATIVE_CASES = {
    # --- UED shape (RFC 0021 §4) ---
    "malformed_slot_name": (
        _rename(1, "Dkdv"),
        r"slot name 'Dkdv' in UED fixture.ued.json must match",
    ),
    "slot_name_global": (
        _rename(1, "global"),
        r"slot name 'global' in UED fixture.ued.json is reserved",
    ),
    "duplicate_slot": (
        _rename(2, "dkdv"),
        r"slot name 'dkdv' is declared more than once in UED fixture.ued.json",
    ),
    "empty_slots": (
        _ued(lambda doc: doc.update(slots=[])),
        r"key 'slots' in UED fixture.ued.json must be a non-empty array",
    ),
    "slot_without_metadata": (
        _slots(lambda slots: slots[0].pop("metadata")),
        r"UED fixture.ued.json slot 'prep' missing required field 'metadata'",
    ),
    "slot_unknown_member": (
        _slots(lambda slots: slots[0].update(heuristic="x")),
        r"slots\[0\] in UED fixture.ued.json has unknown fields",
    ),
    "slot_references_unknown_kmd": (
        _slots(
            lambda slots: slots[0].update(
                metadata="d5a00000-0000-4000-8000-0000000000aa"
            )
        ),
        r"references unknown KMD descriptor Id",
    ),
    "engine_metadata_beside_slots": (
        _beside_slots("metadata", KMD_IDS["prep"]),
        r"key 'metadata' in UED fixture.ued.json must not appear beside 'slots'",
    ),
    "engine_knobs_beside_slots": (
        _beside_slots("knobs", ["block_m"]),
        r"key 'knobs' in UED fixture.ued.json must not appear beside 'slots'",
    ),
    "engine_ranker_beside_slots": (
        _beside_slots(
            "sort_kernel_catalog", {"default": "d5a00000-0000-4000-8000-000000000010"}
        ),
        r"key 'sort_kernel_catalog' in UED fixture.ued.json must not appear beside",
    ),
    "engine_candidate_predictor_beside_slots": (
        _beside_slots(
            "predict_applicable_kernels",
            {"default": "d5a00000-0000-4000-8000-000000000010"},
        ),
        r"key 'predict_applicable_kernels' in UED fixture.ued.json must not appear",
    ),
    "ued_1_0_with_slots": (
        _ued(lambda doc: doc.update(version="1.0")),
        r"UED fixture.ued.json declares version 1.0 but uses slot member 'slots'",
    ),
    "kdp_1_0_with_slot": (
        {"prep.kdp": lambda doc: doc.update(version="1.0")},
        r"KDP prep.kdp.json declares version 1.0 but uses slot member 'slot'",
    ),
    "ued_1_2_with_slots": (
        _ued(lambda doc: doc.update(version="1.2")),
        r"UED fixture.ued.json declares version 1.2 but uses slot member 'slots'; "
        r"this packer supports slot members only at version 1.1",
    ),
    "ued_2_1_with_slots": (
        _ued(lambda doc: doc.update(version="2.1")),
        r"UED fixture.ued.json declares version 2.1 but uses slot member 'slots'; "
        r"this packer supports slot members only at version 1.1",
    ),
    "kdp_1_2_with_slot": (
        {"prep.kdp": lambda doc: doc.update(version="1.2")},
        r"KDP prep.kdp.json declares version 1.2 but uses slot member 'slot'; "
        r"this packer supports slot members only at version 1.1",
    ),
    "ued_1_0_shared_workspace": (
        _ued(lambda doc: (_unslotted(doc), doc.update(version="1.0"))),
        r"declares version 1.0 but uses slot member 'shared_workspace'",
    ),
    # --- shared_workspace (RFC 0021 §6; structure only) ---
    "shared_workspace_without_slots": (
        _ued(_unslotted),
        r"key 'shared_workspace' in UED fixture.ued.json requires 'slots'",
    ),
    "shared_workspace_without_native_graph_match": (
        _ued(_without("graph_match")),
        r"key 'shared_workspace' in UED fixture.ued.json requires a native graph_match",
    ),
    "shared_workspace_empty_token": (
        _ued(lambda doc: doc.update(shared_workspace="")),
        r"shared_workspace must be a nonempty string",
    ),
    # --- KDP.slot ---
    "kdp_without_slot": (
        {"dq.kdp": _without("slot")},
        r"KDP dq.kdp.json declares no slot, which every pack of slotted engine "
        r"'hipkernel:SlottedFixture' must",
    ),
    "kdp_names_undeclared_slot": (
        {"dq.kdp": lambda doc: doc.update(slot="nope")},
        r"KDP dq.kdp.json names slot 'nope', which engine 'hipkernel:SlottedFixture' "
        r"does not declare",
    ),
    "kdp_malformed_slot": (
        {"dq.kdp": lambda doc: doc.update(slot="DQ")},
        r"slot name 'DQ' in KDP dq.kdp.json must match",
    ),
    "kdp_slot_global": (
        {"dq.kdp": lambda doc: doc.update(slot="global")},
        r"slot name 'global' in KDP dq.kdp.json is reserved",
    ),
    "kdp_slot_not_a_string": (
        {"dq.kdp": lambda doc: doc.update(slot=7)},
        r"KDP dq.kdp.json.slot must be a nonempty string",
    ),
    "kdp_slot_on_unslotted_engine": (
        _ued(
            lambda doc: (
                _unslotted(doc),
                doc.pop("shared_workspace"),
                doc.pop("predict_engine"),
            )
        ),
        r"KDP \w+.kdp.json declares slot '\w+', but engine "
        r"'hipkernel:SlottedFixture' declares no slots",
    ),
    # --- per-slot knobs and rankers ---
    "knob_not_in_slot_kmd": (
        _slots(lambda slots: slots[1].update(knobs=["block_n"])),
        r"UED fixture.ued.json slot 'dkdv' exposes knob 'block_n', which KMD "
        r"dkdv.kmd.json does not declare",
    ),
    "slot_knob_listed_twice": (
        _slots(lambda slots: slots[1].update(knobs=["block_m", "block_m"])),
        r"slot 'dkdv'.knobs lists 'block_m' twice",
    ),
    "slot_ranker_map_empty": (
        _slots(lambda slots: slots[1].update(sort_kernel_catalog={})),
        r"slot 'dkdv'.sort_kernel_catalog must be a nonempty arch-to-UUID map",
    ),
    "slot_ranker_declares_metric": (
        {
            "dkdv.uhd": lambda doc: doc.update(
                score={"metric": "tflops", "calibrated": True}
            )
        },
        r"slot 'dkdv'.sort_kernel_catalog.default names UHD .* declares score.metric "
        r"'tflops'; a slot ranker must be metric-less",
    ),
    "slot_ranker_reads_ws_token": (
        {"dkdv.uhd": lambda doc: doc.update(features_signature=["$ws.bytes"])},
        r"dkdv.uhd.json: feature signature reads '\$ws.bytes', a shared-workspace "
        r"token a slotted engine reserves",
    ),
    "slot_ranker_reads_ws_token_in_expression": (
        {
            "dkdv.uhd": lambda doc: doc.update(
                features_signature=[
                    "$kernel.block_m",
                    {"mul": ["$kernel.block_m", {"max": [1, "$ws.scratch_offset"]}]},
                ]
            )
        },
        r"reads '\$ws.scratch_offset', a shared-workspace token",
    ),
    "slot_ranker_reads_two_ws_tokens": (
        {
            "dkdv.uhd": lambda doc: doc.update(
                features_signature=["$ws.scratch_offset", "$ws.bytes"]
            )
        },
        r"reads '\$ws.bytes', a shared-workspace token",
    ),
    # --- slot / kmds provenance (RFC 0021 §9) ---
    "slot_ranker_selector_only": (
        {
            "dkdv.uhd": lambda doc: doc.update(
                trained_against={"selector_revision": "r1"}
            )
        },
        r"dkdv.uhd.json: trained_against records no descriptor set, which a model "
        r"bound to a UED must",
    ),
    "estimate_selector_only": (
        {
            "predict_engine.uhd": lambda doc: doc.update(
                trained_against={"selector_revision": "r1"}
            )
        },
        r"predict_engine.uhd.json: trained_against records no descriptor set, which "
        r"a model bound to a UED must",
    ),
    "unslotted_engine_estimate_names_kmds": (
        _ued(lambda doc: (_unslotted(doc), doc.pop("shared_workspace"))),
        r"UED fixture.ued.json predict_engine UHD predict_engine.uhd.json: "
        r"trained_against names slot provenance \('slot' or 'kmds'\), which an "
        r"engine that declares no slots cannot satisfy",
    ),
    "unslotted_engine_ranker_names_slot": (
        _ued(
            lambda doc: (
                _unslotted(doc),
                doc.pop("shared_workspace"),
                doc.pop("predict_engine"),
                doc.update(
                    sort_kernel_catalog={
                        "default": "d5a00000-0000-4000-8000-000000000010"
                    }
                ),
            )
        ),
        r"UED fixture.ued.json sort_kernel_catalog UHD dkdv.uhd.json: "
        r"trained_against names slot provenance",
    ),
    "ranker_names_another_slot": (
        {"dkdv.uhd": _ranker_trained(slot="dq")},
        r"trained_against names slot 'dq', but the ranker serves slot 'dkdv'",
    ),
    "ranker_names_no_slot": (
        {"dkdv.uhd": _ranker_trained(slot=None)},
        r"trained_against names no slot, but the ranker of slot 'dkdv' must name",
    ),
    "ranker_names_another_slots_kmd": (
        {
            "dkdv.uhd": lambda doc: doc["trained_against"]["kmd"].update(
                id=KMD_IDS["dq"]
            )
        },
        r"trained_against names a KMD other than the one of slot 'dkdv'",
    ),
    "ranker_names_kmds": (
        {
            "dkdv.uhd": lambda doc: (
                doc["trained_against"].update(
                    kmds=[{"slot": "dkdv", **doc["trained_against"].pop("kmd")}]
                ),
                doc["trained_against"].pop("slot"),
            )
        },
        r"trained_against names kmds, but the ranker of slot 'dkdv' must name its own",
    ),
    "estimate_misses_a_slot": (
        {"predict_engine.uhd": _estimate_kmds(lambda kmds: kmds.pop())},
        r"trained_against kmds does not name slot 'dq'",
    ),
    "estimate_names_undeclared_slot": (
        {
            "predict_engine.uhd": _estimate_kmds(
                lambda kmds: kmds[0].update(slot="nope")
            )
        },
        r"trained_against kmds names slot 'nope', which the engine does not declare",
    ),
    "estimate_names_another_slots_kmd": (
        {
            "predict_engine.uhd": _estimate_kmds(
                lambda kmds: kmds[0].update(id=KMD_IDS["dq"])
            )
        },
        r"trained_against kmds names a KMD other than the one of slot 'prep'",
    ),
    "estimate_names_one_kmd": (
        {"predict_engine.uhd": _estimate_one_kmd},
        r"trained_against names kmd, but a slotted engine's predict_engine model must "
        r"name every slot's KMD in kmds",
    ),
    "estimate_names_a_slot": (
        {
            "predict_engine.uhd": lambda doc: (
                _estimate_one_kmd(doc),
                doc["trained_against"].update(slot="prep"),
            )
        },
        r"trained_against names a single slot, but a slotted engine's predict_engine "
        r"model must name every slot's KMD in kmds",
    ),
    "estimate_repeats_a_slot": (
        {
            "predict_engine.uhd": _estimate_kmds(
                lambda kmds: kmds.append(copy.deepcopy(kmds[0]))
            )
        },
        r"kmds names slot 'prep' twice",
    ),
    "uhd_1_0_with_slot": (
        {"dkdv.uhd": lambda doc: doc.update(version="1.0")},
        r"uses slot provenance \('slot' or 'kmds'\), which requires UHD version 1.1",
    ),
    "uhd_1_0_with_kmds": (
        {"predict_engine.uhd": lambda doc: doc.update(version="1.0")},
        r"uses slot provenance \('slot' or 'kmds'\), which requires UHD version 1.1",
    ),
    "uhd_slot_not_a_name": (
        {"dkdv.uhd": _ranker_trained(slot="Dkdv")},
        r"slot name 'Dkdv' in .*trained_against.slot must match",
    ),
    "uhd_kmds_global": (
        {
            "predict_engine.uhd": _estimate_kmds(
                lambda kmds: kmds[0].update(slot="global")
            )
        },
        r"slot name 'global' in .*trained_against.kmds is reserved",
    ),
    "uhd_kmd_and_kmds": (
        {
            "predict_engine.uhd": lambda doc: doc["trained_against"].update(
                kmd={"id": KMD_IDS["prep"], "revision": "1.0"}
            )
        },
        r"trained_against names both kmd and kmds",
    ),
    "uhd_kmds_id_not_a_uuid": (
        {"predict_engine.uhd": _estimate_kmds(lambda kmds: kmds[0].update(id="nope"))},
        r"trained_against.kmds requires a UUID, got 'nope'",
    ),
    "uhd_kmds_revision_malformed": (
        {
            "predict_engine.uhd": _estimate_kmds(
                lambda kmds: kmds[0].update(revision="x")
            )
        },
        r"trained_against.kmds revision has invalid version 'x'",
    ),
    "uhd_kmds_unknown_member": (
        {"predict_engine.uhd": _estimate_kmds(lambda kmds: kmds[0].update(extra=1))},
        r"trained_against.kmds has unknown fields \['extra'\]",
    ),
    "uhd_empty_kmds": (
        {"predict_engine.uhd": _ranker_trained(kmds=[])},
        r"trained_against.kmds must be a non-empty array",
    ),
}


@pytest.mark.parametrize("name", sorted(NEGATIVE_CASES))
def test_the_packer_refuses_a_slotted_bundle_the_loader_would_drop(tmp_path, name):
    edits, message = NEGATIVE_CASES[name]
    root = _root(tmp_path, **edits)

    with pytest.raises(HkpPackError, match=message):
        _load(root)


def test_a_kernel_may_not_serve_two_slots(tmp_path):
    root = _shared_kernel_root(tmp_path, ["dkdv", "dq"])

    with pytest.raises(
        HkpPackError,
        match=r"which already serves slot 'dkdv' of engine 'hipkernel:SlottedFixture'; "
        r"a kernel serves at most one slot",
    ):
        _load(root)


def test_a_kernel_may_serve_different_slots_of_two_engines(tmp_path):
    # The one-slot rule is per engine: a second engine's pack lists the kernel the
    # first engine's `dkdv` pack lists, under its own slot `dq`.
    root = _shared_kernel_root(tmp_path, ["dkdv"])
    other = "d5a00000-0000-4000-8000-0000000000f3"
    ued = _read(root / "fixture.ued.json")
    ued.update(
        id=other,
        name="hipkernel:OtherSlotted",
        slots=[slot for slot in ued["slots"] if slot["name"] != "prep"],
    )
    ued.pop("predict_engine")
    ued.pop("shared_workspace")
    ued.pop("graph_match")
    for slot in ued["slots"]:
        slot.pop("sort_kernel_catalog")
    _write(root / "other.ued.json", ued)
    pack = _read(root / "shared_0.kdp.json")
    pack.update(
        id="d5a00000-0000-4000-8000-0000000000f4",
        name="hipkernel:other_shared",
        engine=other,
        slot="dq",
    )
    _write(root / "other_shared.kdp.json", pack)

    assert len(_load(root).kdps()) == 5


def test_a_pack_of_a_slotted_engine_without_a_slot_is_refused_by_the_desk_check(
    tmp_path,
):
    root = _root(tmp_path, **{"dq.kdp": _without("slot")})

    with pytest.raises(HkpPackError, match=r"declares no slot, which every pack of"):
        descriptor_context.resolve_bundles(descriptor_context.Index(str(root)))


def test_the_unedited_slotted_fixture_loads(tmp_path):
    # The control for every case above: each fails for its own rule only if the
    # fixture it edits loads.
    assert len(_load(_root(tmp_path)).kdps()) == 3


# --- Unslotted output is unchanged -----------------------------------------------
UNSLOTTED_MANIFEST = (
    Path(__file__).resolve().parent / "fixtures" / "unslotted_pack.json"
)
_WINDOWS = sys.platform == "win32"
_EMPTY_ARCH = Path(__file__).resolve().parent / "fixtures" / "empty_arch"


def _unslotted_root(tmp_path: Path) -> Path:
    """The `empty_arch` engine plus one wildcard pack of an embedded kernel."""
    root = tmp_path / "unslotted"
    shutil.copytree(_EMPTY_ARCH, root)
    if _WINDOWS:
        # The golden digests are over LF bytes; a Windows checkout may have converted the fixtures.
        for fixture in root.rglob("*.json"):
            fixture.write_bytes(fixture.read_bytes().replace(b"\r\n", b"\n"))
    _write(
        root / "solo.kdp.json",
        {
            "version": "1.0",
            "id": "6a4a8f6e-6c1a-5b2b-9f3b-0d6f0f1f6a01",
            "name": "test_fixture:solo_pack",
            "arch": [],
            "matchers": ["umd-solo"],
            "engine": "7dfa7b97-7409-5013-8bc5-a644a26ab88d",
            "dispatch": "udd-solo",
            "kernelDescriptors": [
                {
                    "version": "1.0",
                    "id": "6a4a8f6e-6c1a-5b2b-9f3b-0d6f0f1f6a02",
                    "name": "solo.k",
                    "kernel_source": {
                        "kind": "embedded_source",
                        "source_file": "kernels/PointwiseAdd.cpp",
                        "entry_point": "PointwiseAdd",
                    },
                    "metadata": {"dtype": "FLOAT", "block_size": 64},
                    "priority": 0,
                }
            ],
        },
    )
    return root


def _content_digest(name: str, data: bytes) -> str:
    """The digest of a packed file's content. A gzip file is digested decompressed:
    its deflate stream is the interpreter's zlib's to choose. The packer rewrites a
    KDP in text mode, which ends its line with CRLF on Windows only; the digests
    are over LF bytes."""
    if name.endswith(".gz"):
        data = gzip.decompress(data)
    if _WINDOWS:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def test_an_unslotted_bundle_packs_to_the_same_bytes_as_before_slots(
    tmp_path, rocm_kpack_dir
):
    """The packed tree of an unslotted bundle, file by file, against digests taken
    from the packer as it was before slotted packing: slots add no byte to a
    bundle that has none, and its UED, KDP and UHD stay at the formats they had."""
    shard = _files(_pack(_unslotted_root(tmp_path), tmp_path, rocm_kpack_dir))

    digests = {name: _content_digest(name, data) for name, data in shard.items()}
    assert digests == _read(UNSLOTTED_MANIFEST)
    assert {
        json.loads(shard[name])["version"]
        for name in shard
        if name.endswith((".ued.json", ".kdp.json", ".uhd.json"))
    } == {"1.0"}
