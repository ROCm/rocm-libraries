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


GFX11 = "gfx11-generic"
GFX11_REDUCE = "fmha_bwd_dbias_reduce_gfx11_kernel_0"


def _tailzero(arch: str, dtype: str = "bf16", file: str = "tz.hsaco") -> dict:
    return {
        "hash": "a3287842cb25",
        "spec": f"tailzero,{dtype}" + ("" if arch == ARCH else f";arch={arch}"),
        "file": file,
        "symbol": f"fmha_bwd_tailzero_{arch[:6].rstrip('-')}_kernel_0",
        "sha256": "1" * 64,
        "path": Path("unused"),
    }


def _gfx11_rows() -> list[gd.KernelEntry]:
    """gfx11 rows: every slot has a plain row, and the dq op has a bias row whose reduce is a companion object."""
    rows = [_entry(op, f"{op}_a", grid="g", has_bias=0) for op in SLOT_OPS]
    companion = {
        "hash": "aa5ed4007ca3",
        "spec": f"dbias_reduce,bf16,32,plain,bk,0,0,tail,0,1;arch={GFX11}",
        "file": "aa5ed4007ca3.hsaco",
        "symbol": GFX11_REDUCE,
        "sha256": "0" * 64,
        "path": Path("unused"),
    }
    rows[2] = _entry(
        "sdpa_bwd_dq", "sdpa_bwd_dq_a", grid="g", has_bias=0, reduce_symbol=""
    )
    rows.append(
        gd.KernelEntry(
            op="sdpa_bwd_dq",
            path=Path("unused"),
            data=b"",
            record=_entry(
                "sdpa_bwd_dq",
                "sdpa_bwd_dq_bias",
                grid="g",
                has_bias=1,
                reduce_symbol=GFX11_REDUCE,
            ).record,
            companions={"dbias_reduce": companion, "tailzero": _tailzero(GFX11)},
        )
    )
    return rows


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
        # The dQ and dK/dV ops also carry the dispatches of their multi-launch rows (`dq_reduce`, `dq_u2`, `dkdv_u2`), shared by every arch.
        extra = [
            f"flydsl_{op}_{suffix}.udd.json" for suffix in gd.EXTRA_PACKS.get(op, {})
        ]
        assert sorted(docs) == sorted(
            [
                f"flydsl_{op}.kmd.json",
                f"flydsl_{op}.udd.json",
                f"flydsl_{op}_kernel_match.umd.json",
                *extra,
            ]
        )
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
    assert ued["sdk_version"] == gd.SDK_VERSION_BY_OP["sdpa_bwd_split"]
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
    # head_dim is a graph fact but also an alternative of dkdv and dq, whose tiles overlap;
    # prep has no tile knob, and every slot's schedule fields are knobs.
    assert by_slot == {
        "prep": ["block_m", "waves_per_eu"],
        "dkdv": ["head_dim", "block_m", "waves_per_eu"],
        "dq": ["head_dim", "block_m", "waves_per_eu"],
    }
    for op, slot in zip(SLOT_OPS, SLOTS):
        fields = [
            f["name"] for f in folders[Path(op)][f"flydsl_{op}.kmd.json"]["fields"]
        ]
        assert set(by_slot[slot]) <= set(fields)
        alternatives = gd.ALTERNATIVE_KNOBS[op]
        assert not set(by_slot[slot]) & gd.GRAPH_DETERMINED_FIELDS - alternatives
        assert set(by_slot[slot]) == set(fields) & (gd.TUNING_FIELDS | alternatives)
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
    with_reduce = {slot: [kdp] for slot, kdp in packs.items()}
    packs["dq"]["kernelDescriptors"][0]["id"] = packs["prep"]["kernelDescriptors"][0][
        "id"
    ]
    with pytest.raises(gd.HkpPackError, match="slot 'prep' and slot 'dq'"):
        gd._require_disjoint_ids(with_reduce)


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


def test_a_dq_row_with_a_companion_ships_in_its_own_pack_on_the_reduce_dispatch() -> (
    None
):
    folders = gd.build_documents(_gfx11_rows(), GFX11)
    pack = folders[Path("sdpa_bwd_dq") / GFX11]
    assert sorted(pack) == ["flydsl_sdpa_bwd_dq.kdp.json", f"{GFX11}.reduce.kdp.json"]
    plain, reduce = (
        pack["flydsl_sdpa_bwd_dq.kdp.json"],
        pack[f"{GFX11}.reduce.kdp.json"],
    )
    shared = folders[Path("sdpa_bwd_dq")]
    udd = shared["flydsl_sdpa_bwd_dq_reduce.udd.json"]
    assert (
        udd["dispatch_symbol"] == "hipkernel.flydsl_sdpa_bwd_split.dq_reduce.dispatch"
    )
    assert plain["dispatch"] == shared["flydsl_sdpa_bwd_dq.udd.json"]["id"]
    assert reduce["dispatch"] == udd["id"]
    assert (reduce["slot"], reduce["version"]) == ("dq", "1.1")
    assert reduce["id"] != plain["id"]
    (single,) = [k for k in plain["kernelDescriptors"]]
    assert "kernel_source" in single and "sources" not in single
    (multi,) = reduce["kernelDescriptors"]
    assert "kernel_source" not in multi
    assert list(multi["sources"]) == ["dq", "dbias_reduce", "tailzero"]
    assert multi["sources"]["dq"] == {
        "kind": "hsaco",
        "file": "sdpa_bwd_dq_bias.hsaco",
        "symbol": "sym_sdpa_bwd_dq_bias",
    }
    assert multi["sources"]["dbias_reduce"] == {
        "kind": "hsaco",
        "file": "aa5ed4007ca3.hsaco",
        "symbol": GFX11_REDUCE,
    }
    assert (
        multi["metadata"]["reduce_symbol"] == multi["sources"]["dbias_reduce"]["symbol"]
    )


def test_a_reduce_symbol_that_is_not_its_companions_is_refused() -> None:
    rows = _gfx11_rows()
    rows[-1].record["knobs"]["reduce_symbol"] = "another_kernel"
    with pytest.raises(gd.HkpPackError, match="reduce_symbol"):
        gd.build_documents(rows, GFX11)


def test_a_gfx11_pack_with_a_reduce_row_passes_the_authoring_validator(
    tmp_path: Path,
) -> None:
    folders = gd.build_documents(_gfx11_rows(), GFX11)
    gd._write(tmp_path, folders)
    assert gd._check(tmp_path, folders) == []
    gd.load_flat_input(
        tmp_path, GenericTargets.load(DEFAULT_TABLE_PATH), log=lambda *a, **k: None
    )


def test_every_arch_run_of_the_dq_op_authors_the_shared_reduce_dispatch() -> None:
    gfx950 = _split()[Path("sdpa_bwd_dq")]
    gfx11 = gd.build_documents(_gfx11_rows(), GFX11)[Path("sdpa_bwd_dq")]
    name = "flydsl_sdpa_bwd_dq_reduce.udd.json"
    assert gfx950[name] == gfx11[name]


def _with_companions(row: gd.KernelEntry, companions: dict) -> gd.KernelEntry:
    return row._replace(companions=companions)


def test_a_dq_row_with_only_a_tailzero_companion_is_dq_then_tailzero_on_the_plain_dispatch() -> (
    None
):
    """Every gfx950 dQ row and every gfx11 dQ row without a standalone reduce: ``{dq, tailzero}`` on the plain dQ dispatch."""
    rows = _rows()
    rows[5] = _with_companions(rows[5], {"tailzero": _tailzero(ARCH)})
    folders = _split(rows)
    pack = folders[Path("sdpa_bwd_dq") / ARCH]
    assert sorted(pack) == ["flydsl_sdpa_bwd_dq.kdp.json"]
    plain = pack["flydsl_sdpa_bwd_dq.kdp.json"]
    assert (
        plain["dispatch"]
        == folders[Path("sdpa_bwd_dq")]["flydsl_sdpa_bwd_dq.udd.json"]["id"]
    )
    (tailed,) = [k for k in plain["kernelDescriptors"] if "sources" in k]
    assert list(tailed["sources"]) == ["dq", "tailzero"]
    assert tailed["sources"]["tailzero"]["file"] == "tz.hsaco"
    # The row's reduce symbol stays a symbol of its own object.
    assert tailed["metadata"]["reduce_symbol"] == REDUCE


def test_a_merged_row_launches_merged_then_tailzero_across_folders() -> None:
    row = _entry("sdpa_bwd_merged", "sdpa_bwd_merged_a", gqa_pack=0, block_q=-1)
    crossing = _tailzero(ARCH, file="../../sdpa_bwd_dq/gfx950/tz.hsaco")
    folders = gd.build_documents([_with_companions(row, {"tailzero": crossing})], ARCH)
    (kernel,) = folders[Path("sdpa_bwd_merged") / ARCH][
        "flydsl_sdpa_bwd_merged.kdp.json"
    ]["kernelDescriptors"]
    assert list(kernel["sources"]) == ["merged", "tailzero"]
    assert kernel["sources"]["tailzero"]["file"] == "../../sdpa_bwd_dq/gfx950/tz.hsaco"
    assert "kernel_source" not in kernel


@pytest.mark.parametrize("block_q", [None, 0, -1, 65, 128])
@pytest.mark.parametrize("op", ["sdpa_bwd_dkdv", "sdpa_bwd_merged"])
def test_a_packed_row_without_a_valid_block_q_is_refused(op: str, block_q) -> None:
    knobs = {} if block_q is None else {"block_q": block_q}
    rows = [_entry(op, f"{op}_packed", gqa_pack=1, **knobs)]
    with pytest.raises(gd.HkpPackError, match="block_q"):
        gd.build_documents(rows, ARCH)


@pytest.mark.parametrize("block_q", [1, 32, 64])
def test_a_packed_row_with_a_block_q_up_to_the_bound_carries_it(block_q: int) -> None:
    rows = _rows()
    rows[2] = _entry(
        "sdpa_bwd_dkdv", "sdpa_bwd_dkdv_packed", gqa_pack=1, block_q=block_q
    )
    rows[3] = _entry(
        "sdpa_bwd_dkdv", "sdpa_bwd_dkdv_plain", gqa_pack=0, block_q=-1, dtype="f16"
    )
    folders = _split(rows)
    kmd = folders[Path("sdpa_bwd_dkdv")]["flydsl_sdpa_bwd_dkdv.kmd.json"]
    assert {"name": "block_q", "type": "int"} in kmd["fields"]
    packed = next(
        k
        for k in _kdp(folders, "sdpa_bwd_dkdv")["kernelDescriptors"]
        if k["metadata"].get("gqa_pack") == 1
    )
    assert packed["metadata"]["block_q"] == block_q


def test_an_unpacked_row_keeps_its_own_block_q() -> None:
    rows = _rows()
    rows[2] = _entry("sdpa_bwd_dkdv", "sdpa_bwd_dkdv_plain", gqa_pack=0, block_q=-1)
    rows[3] = _entry(
        "sdpa_bwd_dkdv", "sdpa_bwd_dkdv_plain2", gqa_pack=0, block_q=64, dtype="f16"
    )
    _split(rows)


def test_the_bound_is_the_engines_header_definition() -> None:
    assert gd.pack_max_block_q() == 64


# ---------------------------------------------------------------------------
# The value-head split (u2), the FLOAT bias and the resolved bias stage
# ---------------------------------------------------------------------------

DK_SPEC = "dkdv,f16,64,plain,bk,0,0,tail,0,1;u2=dk"
DV_SPEC = "dkdv,f16,64,plain,bk,0,0,tail,0,1;u2=dv"


def _kernel(op: str, name: str, spec: str, **knobs: Any) -> gd.KernelEntry:
    """A dK/dV or dQ row of the manifest with a spec, an object and the feature knobs the build writes."""
    row = _entry(
        op,
        name,
        dtype="f16",
        **{"bias_f32": -1, "u2": "auto", "block_threads": 256, **knobs},
    )
    row.record.update({"spec": spec, "sha256": name[0] * 64})
    return row


def _u2_rows(**dv_knobs: Any) -> list[gd.KernelEntry]:
    dk = _kernel("sdpa_bwd_dkdv", "dk", DK_SPEC, block_kv=64, u2="dk")
    dv = _kernel(
        "sdpa_bwd_dkdv", "dv", DV_SPEC, **{"block_kv": 64, "u2": "dv", **dv_knobs}
    )
    plain = _kernel("sdpa_bwd_dkdv", "plain", DK_SPEC.split(";")[0], block_kv=64)
    return [plain, dk, dv]


def test_a_dv_only_row_is_attached_to_its_dk_only_row_and_is_no_kernel() -> None:
    attached = gd.attach_dvonly(_u2_rows())
    assert [e.record["name"] for e in attached] == ["plain", "dk"]
    plain, dk = attached
    assert plain.companions is None
    dvonly = dk.companions[gd.DVONLY_ROLE]
    assert (dvonly["file"], dvonly["symbol"], dvonly["sha256"]) == (
        "dv.hsaco",
        "sym_dv",
        "d" * 64,
    )


def test_a_dk_only_row_ships_in_the_u2_pack_as_dkdv_then_dvonly() -> None:
    rows = [r for r in _rows() if r.op != "sdpa_bwd_dkdv"] + gd.attach_dvonly(
        _u2_rows()
    )
    folders = _split(rows)
    pack = folders[Path("sdpa_bwd_dkdv") / ARCH]
    assert sorted(pack) == ["flydsl_sdpa_bwd_dkdv.kdp.json", f"{ARCH}.u2.kdp.json"]
    shared = folders[Path("sdpa_bwd_dkdv")]
    udd = shared["flydsl_sdpa_bwd_dkdv_u2.udd.json"]
    assert udd["dispatch_symbol"] == "hipkernel.flydsl_sdpa_bwd_split.dkdv_u2.dispatch"
    plain, u2 = pack["flydsl_sdpa_bwd_dkdv.kdp.json"], pack[f"{ARCH}.u2.kdp.json"]
    assert plain["dispatch"] == shared["flydsl_sdpa_bwd_dkdv.udd.json"]["id"]
    assert u2["dispatch"] == udd["id"] and (u2["slot"], u2["version"]) == (
        "dkdv",
        "1.1",
    )
    (single,) = plain["kernelDescriptors"]
    assert "sources" not in single and single["metadata"]["u2_role"] == "none"
    (multi,) = u2["kernelDescriptors"]
    assert list(multi["sources"]) == ["dkdv", "dvonly"]
    assert multi["sources"]["dkdv"]["file"] == "dk.hsaco"
    assert multi["sources"]["dvonly"] == {
        "kind": "hsaco",
        "file": "dv.hsaco",
        "symbol": "sym_dv",
    }
    assert multi["metadata"]["u2_role"] == "dk"


def test_the_dkdv_schema_is_one_for_u2_and_other_rows_and_names_no_raw_u2() -> None:
    rows = [r for r in _rows() if r.op != "sdpa_bwd_dkdv"] + gd.attach_dvonly(
        _u2_rows()
    )
    kmd = _split(rows)[Path("sdpa_bwd_dkdv")]["flydsl_sdpa_bwd_dkdv.kmd.json"]
    fields = {f["name"]: f["type"] for f in kmd["fields"]}
    assert fields["u2_role"] == "string" and fields["bias_f32"] == "int"
    assert "u2" not in fields
    kernels = [
        k
        for name, doc in _split(rows)[Path("sdpa_bwd_dkdv") / ARCH].items()
        for k in doc["kernelDescriptors"]
    ]
    assert {k["metadata"]["bias_f32"] for k in kernels} == {0}
    assert {k["metadata"]["u2_role"] for k in kernels} == {"none", "dk"}


@pytest.mark.parametrize(
    ("dv_knobs", "field"),
    [({"block_kv": 32}, "block_kv"), ({"block_threads": 512}, "block_threads")],
)
def test_a_pair_that_launches_over_two_workgroup_shapes_is_refused(
    dv_knobs, field
) -> None:
    with pytest.raises(ValueError, match=field):
        gd.attach_dvonly(_u2_rows(**dv_knobs))


def test_a_dk_only_row_without_a_dv_only_row_and_the_reverse_are_refused() -> None:
    rows = _u2_rows()
    with pytest.raises(ValueError, match="no dV-only row"):
        gd.attach_dvonly(rows[:2])
    with pytest.raises(ValueError, match="no dK-only row"):
        gd.attach_dvonly([rows[0], rows[2]])


def test_a_dv_only_row_pairs_only_with_the_spec_equal_apart_from_u2() -> None:
    other = _kernel(
        "sdpa_bwd_dkdv", "dk2", DK_SPEC.replace(",64,", ",128,"), block_kv=64, u2="dk"
    )
    with pytest.raises(ValueError, match="no dV-only row"):
        gd.attach_dvonly([*_u2_rows(), other])


def test_a_dv_only_partner_is_held_to_the_rne_rule_with_its_dk_only_row() -> None:
    dk = _kernel(
        "sdpa_bwd_dkdv",
        "dk",
        f"dkdv,bf16,64,plain,bk,0,1,g1,0,1;arch={GFX11};rne=1;u2=dk",
        block_kv=64,
        u2="dk",
    )
    dv = _kernel(
        "sdpa_bwd_dkdv",
        "dv",
        f"dkdv,bf16,64,plain,bk,0,1,g1,0,1;arch={GFX11};u2=dv",
        block_kv=64,
        u2="dv",
    )
    # The pair is refused first for what it is: the specs differ in rne as well as u2.
    with pytest.raises(ValueError, match="no dV-only row"):
        gd.attach_dvonly([dk, dv])
    dv = _kernel(
        "sdpa_bwd_dkdv",
        "dv",
        f"dkdv,bf16,64,plain,bk,0,1,g1,0,1;arch={GFX11};rne=1;u2=dv",
        block_kv=64,
        u2="dv",
    )
    (paired,) = gd.attach_dvonly([dk, dv])
    gd.require_rne_pins([paired])
    paired.companions[gd.DVONLY_ROLE][
        "spec"
    ] = f"dkdv,bf16,64,plain,bk,0,1,g1,0,1;arch={GFX11};u2=dv"
    with pytest.raises(ValueError, match="rne"):
        gd.require_rne_pins([paired])


def _dq(name: str, dtype: str, tile: int, stage: str, **knobs: Any) -> gd.KernelEntry:
    return _entry(
        "sdpa_bwd_dq",
        name,
        dtype=dtype,
        head_dim=tile,
        has_bias=int(stage != "na"),
        bias_f32=knobs.pop("bias_f32", -1),
        u2="auto",
        bias_stage_resolved=stage,
        reduce_symbol="",
    )


def _stages(rows: list[gd.KernelEntry]) -> dict[str, str]:
    limits = gd._lds_tile_limits(rows)
    return {
        r.record["name"]: gd.kmd_knobs(r, limits)["bias_stage_resolved"] for r in rows
    }


def test_the_resolved_stage_splits_direct_by_whether_a_staged_row_serves_the_tile() -> (
    None
):
    """``direct`` where a staged row exists for the dtype at that tile; ``direct_only`` above the largest staged tile, for a FLOAT bias and for fp32."""
    rows = [
        _dq("lds64", "bf16", 64, "lds"),
        _dq("lds224", "bf16", 224, "lds"),
        _dq("explicit128", "bf16", 128, "direct"),
        _dq("tile256", "bf16", 256, "direct"),
        _dq("f32bias64", "bf16", 64, "direct", bias_f32=1),
        _dq("fp32", "fp32", 64, "direct"),
        _dq("plain", "bf16", 64, "na"),
        _dq("f16explicit", "f16", 64, "direct"),
    ]
    assert _stages(rows) == {
        "lds64": "lds",
        "lds224": "lds",
        "explicit128": "direct",
        "tile256": "direct_only",
        "f32bias64": "direct_only",
        "fp32": "direct_only",
        "plain": "na",
        # f16 has no staged row in this catalogue, so none of its direct rows has a staged alternative.
        "f16explicit": "direct_only",
    }


def test_a_float_bias_row_carries_bias_f32_1_and_every_other_row_0() -> None:
    float_row = _dq("f32bias64", "bf16", 64, "direct", bias_f32=1)
    other = _dq("lds64", "bf16", 64, "lds")
    limits = gd._lds_tile_limits([other])
    assert gd.kmd_knobs(float_row, limits)["bias_f32"] == 1
    assert gd.kmd_knobs(other, limits)["bias_f32"] == 0


HV_TAILZERO = {
    **_tailzero(ARCH),
    "spec": "tailzero_hv,bf16",
    "symbol": "fmha_bwd_tailzero_hv_gfx950_kernel_0",
    "file": "tzhv.hsaco",
}


def _dq_row(name: str, u2: str, tailzero: dict, **knobs: Any) -> gd.KernelEntry:
    """A dQ row of one schema (head_dim, reduce_symbol and the feature knobs): ``u2=dq`` for a V-head row, ``auto`` for a standard one."""
    spec = "dq,f16,64,plain,bk,0,0,tail,0,1" + (";u2=dq" if u2 == "dq" else "")
    row = _kernel("sdpa_bwd_dq", name, spec, u2=u2, reduce_symbol="", **knobs)
    return row._replace(companions={"tailzero": tailzero})


def _hv_dq(name: str = "hv", tailzero: dict = HV_TAILZERO) -> gd.KernelEntry:
    return _dq_row(name, "dq", tailzero)


def test_a_v_head_dq_row_ships_in_the_dq_u2_pack_as_dq_then_the_v_head_tailzero() -> (
    None
):
    rows = [r for r in _rows() if r.op != "sdpa_bwd_dq"] + [
        _dq_row("std", "auto", _tailzero(ARCH)),
        _hv_dq(),
    ]
    folders = _split(rows)
    pack = folders[Path("sdpa_bwd_dq") / ARCH]
    assert sorted(pack) == ["flydsl_sdpa_bwd_dq.kdp.json", f"{ARCH}.dq_u2.kdp.json"]
    shared = folders[Path("sdpa_bwd_dq")]
    udd = shared["flydsl_sdpa_bwd_dq_u2.udd.json"]
    assert udd["dispatch_symbol"] == "hipkernel.flydsl_sdpa_bwd_split.dq_u2.dispatch"
    plain, u2 = pack["flydsl_sdpa_bwd_dq.kdp.json"], pack[f"{ARCH}.dq_u2.kdp.json"]
    assert plain["dispatch"] == shared["flydsl_sdpa_bwd_dq.udd.json"]["id"]
    assert u2["dispatch"] == udd["id"] != plain["dispatch"]
    assert (u2["slot"], u2["version"]) == ("dq", "1.1")
    (single,) = plain["kernelDescriptors"]
    assert list(single["sources"]) == ["dq", "tailzero"]
    assert single["sources"]["tailzero"]["file"] == "tz.hsaco"
    assert single["metadata"]["u2_role"] == "none"
    (multi,) = u2["kernelDescriptors"]
    assert list(multi["sources"]) == ["dq", "tailzero"]
    assert multi["sources"]["dq"]["file"] == "hv.hsaco"
    assert multi["sources"]["tailzero"]["file"] == "tzhv.hsaco"
    assert multi["metadata"]["u2_role"] == "dq"


def test_the_dq_op_ships_its_reduce_and_u2_packs_side_by_side() -> None:
    """gfx11: a dB-storing row (``reduce``) and a V-head row (``u2``) are two packs on two dispatches, with distinct files, ids and dispatches."""
    hv_tailzero = {**_tailzero(GFX11), "spec": f"tailzero_hv,bf16;arch={GFX11}"}
    others = [r for r in _gfx11_rows() if r.op != "sdpa_bwd_dq"]
    bias = _dq_row("bias", "auto", _tailzero(GFX11), grid="g", has_bias=1)
    bias.record["knobs"]["reduce_symbol"] = GFX11_REDUCE
    companion = {
        "hash": "aa5ed4007ca3",
        "spec": f"dbias_reduce,bf16,32,plain,bk,0,0,tail,0,1;arch={GFX11}",
        "file": "aa5ed4007ca3.hsaco",
        "symbol": GFX11_REDUCE,
        "sha256": "0" * 64,
        "path": Path("unused"),
    }
    bias = bias._replace(companions={**bias.companions, "dbias_reduce": companion})
    hv = _dq_row("hv11", "dq", hv_tailzero, grid="g", has_bias=0)
    rows = [*others, bias, hv]
    pack = gd.build_documents(rows, GFX11)[Path("sdpa_bwd_dq") / GFX11]
    assert sorted(pack) == [
        "flydsl_sdpa_bwd_dq.kdp.json",
        f"{GFX11}.dq_u2.kdp.json",
        f"{GFX11}.reduce.kdp.json",
    ]
    assert len({kdp["id"] for kdp in pack.values()}) == 3
    assert len({kdp["dispatch"] for kdp in pack.values()}) == 3
    (reduce,) = pack[f"{GFX11}.reduce.kdp.json"]["kernelDescriptors"]
    assert list(reduce["sources"]) == ["dq", "dbias_reduce", "tailzero"]
    (hv,) = pack[f"{GFX11}.dq_u2.kdp.json"]["kernelDescriptors"]
    assert list(hv["sources"]) == ["dq", "tailzero"]


def test_every_arch_run_of_the_dq_and_dkdv_ops_authors_the_shared_u2_dispatches() -> (
    None
):
    gfx950 = _split()
    gfx11 = gd.build_documents(_gfx11_rows(), GFX11)
    for op, name in (
        ("sdpa_bwd_dq", "flydsl_sdpa_bwd_dq_u2.udd.json"),
        ("sdpa_bwd_dkdv", "flydsl_sdpa_bwd_dkdv_u2.udd.json"),
    ):
        assert gfx950[Path(op)][name] == gfx11[Path(op)][name]


def test_a_u2_dispatch_name_is_the_slot_then_u2_and_no_two_ops_share_a_pack_file() -> (
    None
):
    files = [
        f"{stem}.kdp.json"
        for op, packs in gd.EXTRA_PACKS.items()
        if op in SLOT_OPS
        for stem in packs.values()
    ]
    assert sorted(files) == ["dq_u2.kdp.json", "reduce.kdp.json", "u2.kdp.json"]
    folders = _split()
    for op, slot in (("sdpa_bwd_dq", "dq"), ("sdpa_bwd_dkdv", "dkdv")):
        udd = folders[Path(op)][f"flydsl_{op}_u2.udd.json"]
        assert (
            udd["dispatch_symbol"]
            == f"hipkernel.flydsl_sdpa_bwd_split.{slot}_u2.dispatch"
        )


def test_a_pack_that_carries_a_u2_dq_row_passes_the_authoring_validator(
    tmp_path: Path,
) -> None:
    rows = [r for r in _rows() if r.op != "sdpa_bwd_dq"] + [_hv_dq()]
    folders = _split(rows)
    gd._write(tmp_path, folders)
    assert gd._check(tmp_path, folders) == []
