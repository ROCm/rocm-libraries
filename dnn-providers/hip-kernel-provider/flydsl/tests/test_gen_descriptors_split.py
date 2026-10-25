# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The slotted emitter of ``gen_descriptors.py``: the SDPA backward split engine.

``hipkernel:flydsl_sdpa_bwd_split`` has no instance table of its own. Its slots are the
content ops ``sdpa_bwd_prep``, ``sdpa_bwd_dkdv`` and ``sdpa_bwd_dq``; each op keeps its
manifest, digest and ``--check``, and the engine's UED 1.1 sits in the descriptor-only
folder ``sdpa_bwd_split/``. These tests run the emitter on synthetic manifest entries (no
objects, no content tree) and hold the output to RFC 0021 and to the packer's own
authoring validator.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
import gen_descriptors as gd  # noqa: E402
from generators import _manifest  # noqa: E402
from generators._instances import instances_for  # noqa: E402
from generators._ops import OP_KERNEL_DIRS  # noqa: E402
from hkp_pack.generic_targets import DEFAULT_TABLE_PATH, GenericTargets  # noqa: E402

ARCH = "gfx950"
SLOT_OPS = ("sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq")
SLOTS = ("prep", "dkdv", "dq")
REDUCE = "fmha_bwd_dbias_reduce_gfx950_kernel_1"
# Members a slotted UED declares per slot, never beside `slots` (RFC 0021 section 4).
FORBIDDEN_UED_MEMBERS = (
    "metadata",
    "sort_kernel_catalog",
    "knobs",
    "predict_applicable_kernels",
)


def _entry(
    op: str,
    name: str,
    priority: int = 100,
    dtype: str = "bf16",
    waves_per_eu: int = 1,
    **knobs: Any,
) -> gd.KernelEntry:
    record = {
        "name": name,
        "file": f"{name}.hsaco",
        "symbol": f"sym_{name}",
        "dtype": dtype,
        "priority": priority,
        "knobs": {
            "head_dim": 64,
            "block_m": 64,
            "waves_per_eu": waves_per_eu,
            **knobs,
        },
    }
    return gd.KernelEntry(op=op, path=Path("unused"), data=b"", record=record)


def _rows() -> list[gd.KernelEntry]:
    """Two rows per slot; the dq rows carry the combined dBias reduce symbol."""
    rows = []
    for op in SLOT_OPS:
        extra = {"reduce_symbol": ""} if op == "sdpa_bwd_dq" else {}
        rows.append(_entry(op, f"{op}_a", dtype="bf16", **extra))
        combined = {"reduce_symbol": REDUCE} if op == "sdpa_bwd_dq" else {}
        rows.append(_entry(op, f"{op}_b", dtype="f16", waves_per_eu=2, **combined))
    return rows


def _split(rows: list[gd.KernelEntry] | None = None) -> dict:
    return gd.build_documents(_rows() if rows is None else rows, ARCH)


def _ued(folders: dict) -> dict:
    (document,) = folders[Path("sdpa_bwd_split")].values()
    return document


def _kdp(folders: dict, op: str) -> dict:
    return folders[Path(op) / ARCH][f"flydsl_{op}.kdp.json"]


def test_the_engine_is_one_ued_and_each_slot_is_its_op() -> None:
    folders = _split()
    assert sorted(map(str, folders)) == sorted(
        [
            "sdpa_bwd_split",
            *SLOT_OPS,
            *(f"{op}/{ARCH}" for op in SLOT_OPS),
        ]
    )
    ued = _ued(folders)
    assert ued["name"] == "hipkernel:flydsl_sdpa_bwd_split"
    assert [slot["name"] for slot in ued["slots"]] == list(SLOTS)
    for op, slot in zip(SLOT_OPS, SLOTS):
        docs = folders[Path(op)]
        # No UED, ranker or UHD beside a slot's KMD: the engine owns those.
        assert sorted(docs) == [
            f"flydsl_{op}.kmd.json",
            f"flydsl_{op}.udd.json",
            f"flydsl_{op}_kernel_match.umd.json",
        ]
        kmd = docs[f"flydsl_{op}.kmd.json"]
        assert (
            next(s for s in ued["slots"] if s["name"] == slot)["metadata"] == kmd["id"]
        )
        symbols = f"hipkernel.flydsl_sdpa_bwd_split.{slot}"
        assert docs[f"flydsl_{op}.udd.json"]["dispatch_symbol"] == f"{symbols}.dispatch"
        umd = docs[f"flydsl_{op}_kernel_match.umd.json"]
        assert (umd["scope"], umd["match_symbol"]) == (
            "kernel",
            f"{symbols}.kernel_match",
        )
        kdp = _kdp(folders, op)
        assert (kdp["version"], kdp["slot"]) == ("1.1", slot)
        assert kdp["engine"] == ued["id"]
        assert kdp["dispatch"] == docs[f"flydsl_{op}.udd.json"]["id"]
        assert kdp["matchers"] == [umd["id"]]
        (consumer,) = kdp["provenance"]["specialization_contract"]["consumers"]
        assert (consumer["engine_id"], consumer["kmd_id"]) == (ued["id"], kmd["id"])
        assert {k["metadata"]["dtype"] for k in kdp["kernelDescriptors"]} == {
            "bf16",
            "f16",
        }


def test_the_ued_is_slotted_1_1_and_carries_no_engine_wide_selection() -> None:
    ued = _ued(_split())
    assert ued["version"] == "1.1"
    assert ued["sdk_version"] == "1.2.0"
    assert ued["shared_workspace"] == "ws.bytes"
    assert ued["graph_match"] == {
        "native": "hipkernel.flydsl_sdpa_bwd_split.graph_match"
    }
    for member in FORBIDDEN_UED_MEMBERS:
        assert member not in ued
    # A slotted engine's plans cannot be saved, so the note is not authored.
    assert "behavior_notes" not in ued
    # No ranker at all: the runtime orders a slot by priority, then id.
    assert not any("sort_kernel_catalog" in slot for slot in ued["slots"])


def test_a_slots_knobs_are_its_own_tuning_fields() -> None:
    folders = _split()
    by_slot = {slot["name"]: slot["knobs"] for slot in _ued(folders)["slots"]}
    # head_dim is graph-determined, so only the schedule fields are knobs.
    assert by_slot == {
        "prep": ["block_m", "waves_per_eu"],
        "dkdv": ["block_m", "waves_per_eu"],
        "dq": ["block_m", "waves_per_eu"],
    }
    for op, slot in zip(SLOT_OPS, SLOTS):
        fields = [
            f["name"] for f in folders[Path(op)][f"flydsl_{op}.kmd.json"]["fields"]
        ]
        assert set(by_slot[slot]) <= set(fields)
        assert not set(by_slot[slot]) & gd.GRAPH_DETERMINED_FIELDS
        assert set(by_slot[slot]) == set(fields) & gd.TUNING_FIELDS
    # reduce_symbol is graph-determined: a dq KMD field, never a knob.
    assert "reduce_symbol" in {
        f["name"]
        for f in folders[Path("sdpa_bwd_dq")]["flydsl_sdpa_bwd_dq.kmd.json"]["fields"]
    }
    assert "reduce_symbol" not in by_slot["dq"]


def test_a_slot_with_a_field_no_set_classifies_is_refused() -> None:
    rows = [
        _entry("sdpa_bwd_dq", name, unclassified_knob=1, reduce_symbol="")
        for name in ("dq_x", "dq_y")
    ]
    rows += [r for r in _rows() if r.op != "sdpa_bwd_dq"]
    with pytest.raises(gd.HkpPackError, match="unclassified_knob"):
        _split(rows)


def test_combined_dq_rows_keep_the_reduce_symbol_as_metadata() -> None:
    kernels = {
        k["name"]: k["metadata"]
        for k in _kdp(_split(), "sdpa_bwd_dq")["kernelDescriptors"]
    }
    assert kernels["sdpa_bwd_dq_a"]["reduce_symbol"] == ""
    assert kernels["sdpa_bwd_dq_b"]["reduce_symbol"] == REDUCE


def test_no_kernel_id_is_shared_across_slots() -> None:
    folders = _split()
    ids = [
        identity
        for op in SLOT_OPS
        for kdp in [_kdp(folders, op)]
        for identity in [kdp["id"], *(k["id"] for k in kdp["kernelDescriptors"])]
    ]
    assert len(ids) == len(set(ids)) == 3 * 3
    # The same instance names in every slot still get distinct ids.
    rows = [
        (
            _entry(op, "same", reduce_symbol="")
            if op.endswith("dq")
            else _entry(op, "same")
        )
        for op in SLOT_OPS
    ]
    folders = _split(rows)
    assert (
        len({_kdp(folders, op)["kernelDescriptors"][0]["id"] for op in SLOT_OPS}) == 3
    )


def test_an_id_reused_across_slots_is_refused() -> None:
    folders = _split()
    packs = {slot: _kdp(folders, op) for op, slot in zip(SLOT_OPS, SLOTS)}
    packs["dq"]["kernelDescriptors"][0]["id"] = packs["prep"]["kernelDescriptors"][0][
        "id"
    ]
    with pytest.raises(gd.HkpPackError, match="slot 'prep' and slot 'dq'"):
        gd._require_disjoint_ids(packs)


def test_a_duplicate_metadata_tuple_in_a_slot_is_an_error_not_a_drop() -> None:
    rows = _rows() + [_entry("sdpa_bwd_dkdv", "dkdv_dup", dtype="bf16")]
    # dkdv_dup carries the first dkdv row's metadata ({dtype bf16, head_dim, block_m}).
    with pytest.raises(gd.HkpPackError, match="same metadata"):
        _split(rows)
    # The same tuple in two different slots is fine: injectivity is per slot.
    _split()


def test_a_partial_split_is_refused() -> None:
    rows = [r for r in _rows() if r.op != "sdpa_bwd_dkdv"]
    with pytest.raises(gd.HkpPackError, match=r"no content for \['sdpa_bwd_dkdv'\]"):
        _split(rows)


def test_an_unslotted_op_keeps_its_1_0_engine() -> None:
    folders = gd.build_documents([_entry("sdpa_rdna", "r0")], ARCH)
    assert Path("sdpa_bwd_split") not in folders
    docs = folders[Path("sdpa_rdna")]
    ued = docs["flydsl_sdpa_rdna.ued.json"]
    assert ued["version"] == "1.0" and "slots" not in ued and "metadata" in ued
    kdp = folders[Path("sdpa_rdna") / ARCH]["flydsl_sdpa_rdna.kdp.json"]
    assert kdp["version"] == "1.0" and "slot" not in kdp


def test_priorities_pass_through_so_the_policy_default_is_the_reference() -> None:
    # The runtime reference is the top-priority kernel, ties by id. The rows' priority
    # (set by the instance table: a non-flaky row without explicit perf knobs outranks
    # the others) must reach the pack unchanged for that row to be the reference.
    rows = [
        _entry("sdpa_bwd_dkdv", "flaky", priority=100, waves_per_eu=1),
        _entry("sdpa_bwd_dkdv", "explicit_knob", priority=100, waves_per_eu=2),
        _entry("sdpa_bwd_dkdv", "default", priority=101, waves_per_eu=3),
    ]
    rows += [r for r in _rows() if r.op != "sdpa_bwd_dkdv"]
    kernels = _kdp(_split(rows), "sdpa_bwd_dkdv")["kernelDescriptors"]
    assert {k["name"]: k["priority"] for k in kernels} == {
        "flaky": 100,
        "explicit_knob": 100,
        "default": 101,
    }
    reference = min(kernels, key=lambda k: (-k["priority"], k["id"]))
    assert reference["name"] == "default"


def test_the_packers_authoring_validator_accepts_the_output(tmp_path: Path) -> None:
    folders = _split()
    gd._write(tmp_path, folders)
    assert gd._check(tmp_path, folders) == []
    flat = gd.load_flat_input(
        tmp_path, GenericTargets.load(DEFAULT_TABLE_PATH), log=lambda *a, **k: None
    )
    assert flat is not None


def test_check_covers_the_descriptor_only_ued_folder(tmp_path: Path) -> None:
    folders = _split()
    gd._write(tmp_path, folders)
    ued_path = tmp_path / "sdpa_bwd_split" / "flydsl_sdpa_bwd_split.ued.json"

    ued = json.loads(ued_path.read_text(encoding="utf-8"))
    ued["slots"][0]["knobs"].append("head_dim")
    ued_path.write_text(json.dumps(ued, indent=2) + "\n", encoding="utf-8")
    (problems,) = gd._check(tmp_path, folders)
    assert "flydsl_sdpa_bwd_split.ued.json: differs" in problems

    ued_path.unlink()
    (problems,) = gd._check(tmp_path, folders)
    assert "flydsl_sdpa_bwd_split.ued.json: missing" in problems

    ued_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "sdpa_bwd_split" / "stale.ued.json").write_text(
        "{}\n", encoding="utf-8"
    )
    assert any(
        "stale.ued.json: names no object" in p for p in gd._check(tmp_path, folders)
    )


# ---------------------------------------------------------------------------
# Other ops' inputs digests are not moved by the split ops
# ---------------------------------------------------------------------------


def _arch_for(op: str) -> str | None:
    for arch in ("gfx11-generic", "gfx12-generic", "gfx950", "gfx942", "gfx90a"):
        try:
            if instances_for(op, arch):
                return arch
        except KeyError:
            continue
    return None


def test_a_slot_ops_inputs_move_no_other_ops_digest(tmp_path: Path) -> None:
    # The emitter (gen_descriptors.py) is no input to any op's objects, and a slot op's
    # own generator module reaches only that op's digest. Editing both leaves every
    # other op current, so shipping the split ops restales nothing else.
    root = tmp_path / "flydsl"
    ignore = shutil.ignore_patterns("__pycache__")
    shutil.copytree(FLYDSL / "generators", root / "generators", ignore=ignore)
    shutil.copytree(FLYDSL / "kernels_src", root / "kernels_src", ignore=ignore)
    shutil.copy(FLYDSL / "gen_descriptors.py", root / "gen_descriptors.py")

    targets = {op: _arch_for(op) for op in OP_KERNEL_DIRS}
    targets = {op: arch for op, arch in targets.items() if arch}
    assert set(SLOT_OPS) <= set(targets)

    def digests() -> dict[str, str]:
        return {
            op: _manifest.inputs_digest(op, arch, root) for op, arch in targets.items()
        }

    before = digests()
    for path in (
        root / "gen_descriptors.py",
        root / "generators" / "gen_sdpa_bwd_dq.py",
    ):
        path.write_bytes(path.read_bytes() + b"\n# edited\n")
    after = digests()
    assert [op for op in before if before[op] != after[op]] == ["sdpa_bwd_dq"]
