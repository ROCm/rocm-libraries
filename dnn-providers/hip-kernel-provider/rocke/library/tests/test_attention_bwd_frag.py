# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests of the backward fragment helpers.

* Lane-map facts per atom (``a_contig``, ``b_contig``, ``affine``, ``c_is_aT``).
* Relabel and K-permutation proofs on every backward atom of every arch: the
  emission program, replayed on coordinates, reproduces the consumer A map,
  and the K-permuted B operand pairs every A value with the B value of the same
  logical K. Synthetic mismatches raise.
* No 128-bit transpose read anywhere in the backward (IR of the emitted helpers
  and a source scan of every backward module).
* Transpose-read plans: every catalog operand takes the catalog plan (the
  legacy model); the K-permuted B of a 16x16x32 consumer takes the stacked
  plan (two 16x16x16 B reads 16 K rows apart); near misses are refused; the
  loader's emitted addresses are the proved plan's, lane for lane.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import pytest
from rocke.core.arch.target import LayoutMap
from rocke.core.ir import F16, F32, IRBuilder, PtrType, VectorType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common import _attention_bwd_frag as F
from kernels.common._attention_bwd_caps import BWD_ARCHES, bwd_atoms

DTYPES = ("fp16", "bf16")
FLAVORS = ("llvm20", "llvm22", "llvm23")
_LIB = Path(__file__).resolve().parents[1]


def _pairs():
    for arch in BWD_ARCHES:
        for dt in DTYPES:
            ops = bwd_atoms(arch, dt)
            for a_op in ops:
                yield pytest.param(arch, dt, ops[0], a_op, id=f"{arch}-{dt}-k{a_op.k}")


# --- facts ------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("arch", BWD_ARCHES)
def test_lane_map_facts(arch, dtype):
    for op in bwd_atoms(arch, dtype):
        f = F.atom_facts(op)
        assert f.a_contig and f.b_contig and f.affine and f.ab_same_k, f
        expect_at = op.k == 16 and arch != "gfx1151"
        assert f.c_is_aT is expect_at, (op.op_id, f)


def test_int_builder_evaluates_every_lane_and_slot():
    op = bwd_atoms("gfx942", "fp16")[0]
    ev = F.eval_layout(op.c_layout())
    assert len(ev) == 64 and all(len(s) == 4 for s in ev)
    cells = {c for lane in ev for c in lane}
    assert cells == {(r, c) for r in range(16) for c in range(16)}


# --- relabel and K-permutation proofs (all atoms) ---------------------------------


@pytest.mark.parametrize("arch, dtype, c_op, a_op", list(_pairs()))
def test_relabel_proof(arch, dtype, c_op, a_op):
    plan = F.plan_c_to_a(c_op, a_op)
    assert plan.c_tiles == a_op.k // 16
    expected_kind = "xlane16" if arch == "gfx1151" else "direct"
    assert plan.kind == expected_kind
    assert plan.needs_k_perm is (a_op.k == 32)
    assert sorted(plan.k_perm) == list(range(a_op.k))
    a = F.eval_layout(a_op.a_layout())
    got = F.simulate_relabel(plan, c_op, a_op)
    for lane in range(a_op.wave_size):
        for s in range(a_op.a_frag_len):
            row, k = a[lane][s]
            assert got[lane][s] == (row, plan.k_perm[k])


@pytest.mark.parametrize("arch, dtype, c_op, a_op", list(_pairs()))
def test_k_permutation_pairs_a_and_b(arch, dtype, c_op, a_op):
    """Every MMA product pairs A and B values of the same logical K."""
    plan = F.plan_c_to_a(c_op, a_op)
    if not plan.needs_k_perm:
        with pytest.raises(ValueError):
            F.kperm_b_coords(dataclasses.replace(plan, kind="xlane16"), c_op, a_op)
        return
    coords = F.kperm_b_coords(plan, c_op, a_op)
    a = F.eval_layout(a_op.a_layout())
    bl = F.eval_layout(a_op.b_layout())
    a_logical, b_logical = {}, {}
    for lane in range(a_op.wave_size):
        b_int = coords.int_coords(lane)
        for s in range(a_op.a_frag_len):
            kc = a[lane][s][1]
            a_logical.setdefault(kc, set()).add(plan.k_perm[kc])
            kb = bl[lane][s][0]
            b_logical.setdefault(kb, set()).add(b_int[s][1])
            assert b_int[s][0] == bl[lane][s][1], "B column must stay put"
    assert a_logical == b_logical
    assert all(len(v) == 1 for v in a_logical.values())
    # the loaded K values per lane are runs of 4 at 4*(lane//16) and 16 + that
    assert coords.deltas == tuple((0, d) for d in (0, 1, 2, 3, 16, 17, 18, 19))


# --- synthetic mismatches raise ----------------------------------------------------


def _perturbed(layout: LayoutMap, bad_lane: int) -> LayoutMap:
    fn = layout.fn

    def swapped(b, lane, slot):
        if isinstance(lane, int) and lane == bad_lane and slot in (0, 1):
            return fn(b, lane, 1 - slot)
        return fn(b, lane, slot)

    return dataclasses.replace(layout, fn=swapped)


@pytest.mark.parametrize("arch", BWD_ARCHES)
def test_mismatched_c_map_has_no_relabel(arch):
    op = bwd_atoms(arch, "fp16")[0]
    bad = dataclasses.replace(op, _c_layout=_perturbed(op.c_layout(), 5))
    with pytest.raises(ValueError, match="no register C-to-A relabel"):
        F.plan_c_to_a(bad, op)


def test_mismatched_b_k_map_refuses_the_permuted_load():
    ops = bwd_atoms("gfx950", "fp16")
    c_op, a_op = ops[0], ops[1]
    plan = F.plan_c_to_a(c_op, a_op)
    bad = dataclasses.replace(a_op, _b_layout=_perturbed(a_op.b_layout(), 3))
    assert not F.atom_facts(bad).ab_same_k
    with pytest.raises(ValueError, match="K maps differ"):
        F.kperm_b_coords(plan, c_op, bad)


def test_shape_mismatch_raises():
    ops = bwd_atoms("gfx950", "fp16")
    with pytest.raises(ValueError, match="wave sizes"):
        F.plan_c_to_a(bwd_atoms("gfx1151", "fp16")[0], ops[0])


@pytest.mark.parametrize("dtype", DTYPES)
def test_swapped_half_selectors_fail_the_proof(dtype):
    op = bwd_atoms("gfx1151", dtype)[0]
    plan = F.plan_c_to_a(op, op)
    bad = dataclasses.replace(
        plan, xlane=tuple((j, s1, s0) for j, s0, s1 in plan.xlane)
    )
    assert not F._proved(bad, op, op)
    wrong_src = dataclasses.replace(
        plan, xlane=tuple(((j + 1) % 4, s0, s1) for j, s0, s1 in plan.xlane)
    )
    assert not F._proved(wrong_src, op, op)


def test_direct_plan_with_wrong_permutation_fails_the_proof():
    ops = bwd_atoms("gfx950", "bf16")
    plan = F.plan_c_to_a(ops[0], ops[1])
    bad = dataclasses.replace(plan, k_perm=tuple(range(32)))
    assert not F._proved(bad, ops[0], ops[1])


# --- emission -----------------------------------------------------------------------


def _relabel_kernel(arch, dtype, a_k):
    ops = bwd_atoms(arch, dtype)
    c_op = ops[0]
    a_op = next(op for op in ops if op.k == a_k)
    plan = F.plan_c_to_a(c_op, a_op)
    b = IRBuilder(f"bwd_frag_relabel_{arch}_{dtype}_{a_k}")
    out = b.param("O", PtrType(F16 if dtype == "fp16" else F.ir_type(dtype), "global"))
    src = b.param("C", PtrType(F32, "global"), readonly=True)
    b.kernel.attrs["max_workgroup_size"] = 64
    lane = b.mod(b.thread_id_x(), b.const_i32(c_op.wave_size))
    tiles = []
    for t in range(plan.c_tiles):
        idx = b.add(b.mul(lane, b.const_i32(8)), b.const_i32(4 * t))
        tiles.append(b.global_load_vN(src, idx, F32, c_op.c_frag_len))
    frag = F.emit_c_to_a(b, plan, tiles, dtype=dtype, lane=lane)
    assert frag.type == VectorType(F.ir_type(dtype), a_op.a_frag_len)
    for s in range(a_op.a_frag_len):
        idx = b.add(b.mul(lane, b.const_i32(a_op.a_frag_len)), b.const_i32(s))
        b.global_store(out, idx, b.vec_extract(frag, s))
    b.ret()
    return b.kernel, plan


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("arch", BWD_ARCHES)
def test_relabel_emission(arch, dtype, flavor):
    for a_k in sorted({op.k for op in bwd_atoms(arch, dtype)}):
        kernel, plan = _relabel_kernel(arch, dtype, a_k)
        ir = lower(kernel, arch=arch, llvm_flavor=flavor)
        n_x = len(re.findall(r"call i32 @llvm\.amdgcn\.permlanex16\.i32\(", ir))
        n_p = len(re.findall(r"call i32 @llvm\.amdgcn\.perm\(", ir))
        if plan.kind == "xlane16":
            assert (n_x, n_p) == (4, 8)  # one permlanex16 + two perm_b32 per dword
        else:
            assert (n_x, n_p) == (0, 0)
        assert "alloca" not in ir
        assert not _TR_B128.search(ir)


def test_xlane_relabel_needs_the_lane():
    op = bwd_atoms("gfx1151", "fp16")[0]
    plan = F.plan_c_to_a(op, op)
    b = IRBuilder("bwd_frag_nolane")
    with pytest.raises(ValueError, match="lane id"):
        F.emit_c_to_a(b, plan, [b.zero_vec_f32(8)], dtype="fp16")


@pytest.mark.parametrize("storage", ["k_inner", "k_outer"])
@pytest.mark.parametrize("arch", BWD_ARCHES)
def test_lds_loader_widths(arch, storage):
    """K-contiguous operands load with vector reads; K-strided ones per slot."""
    for op in bwd_atoms(arch, "fp16"):
        b = IRBuilder(f"bwd_frag_ld_{arch}_{storage}_{op.k}")
        s = b.smem_alloc(F16, (64, 64), "s")
        lane = b.mod(b.thread_id_x(), b.const_i32(op.wave_size))
        coords = F.operand_coords(op, "a")
        frag = F.load_frag_lds(b, coords, s, lane, dtype="fp16", storage=storage)
        assert frag.type == VectorType(F16, op.a_frag_len)
        widths = [
            o.attrs["vec"] for o in b.kernel.body.ops if o.name == "tile.smem_load_vN"
        ]
        if storage == "k_inner":
            assert widths == [min(8, op.a_frag_len)] * max(1, op.a_frag_len // 8)
        else:
            assert widths == [1] * op.a_frag_len


def test_loader_rejects_unknown_storage():
    op = bwd_atoms("gfx942", "fp16")[0]
    b = IRBuilder("bwd_frag_bad")
    s = b.smem_alloc(F16, (16, 16), "s")
    with pytest.raises(ValueError, match="storage"):
        F.load_frag_lds(b, F.operand_coords(op, "a"), s, b.thread_id_x(),
                        dtype="fp16", storage="row")  # fmt: skip


# --- no 128-bit transpose read in the backward ------------------------------------

_TR_B128 = re.compile(r"tr16[._]b128|tr_b128|load\.tr\.b128", re.IGNORECASE)
_FORBIDDEN_CALLS = {"ds_read_tr16_b128", "global_load_tr16_b128"}


def _backward_modules():
    root = _LIB / "kernels"
    return sorted(p for p in root.rglob("*.py") if "attention_bwd" in p.name)


def test_backward_sources_never_call_a_b128_transpose_read():
    mods = _backward_modules()
    assert mods, "no backward module found"
    for path in mods:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_CALLS:
                pytest.fail(f"{path.name}:{node.lineno} uses {node.attr}")
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == (
                "load_tile_transpose"
            ):
                kw = {k.arg: k.value for k in node.keywords}
                rpl = kw.get("rows_per_lane")
                assert isinstance(rpl, ast.Constant) and rpl.value == 4, (
                    f"{path.name}:{node.lineno}: load_tile_transpose needs an "
                    f"explicit rows_per_lane=4 (8 routes to the b128 read)"
                )


def test_no_exported_helper_looks_like_a_builder():
    assert not [n for n in F.__all__ if n.startswith("build_")]


# --- flat LDS views and the transpose-read loader -----------------------------------


def _gfx950_atoms():
    return [
        pytest.param(dt, op, id=f"{dt}-k{op.k}")
        for dt in DTYPES
        for op in bwd_atoms("gfx950", dt)
    ]


@pytest.mark.parametrize("role", ["a", "b"])
@pytest.mark.parametrize("dtype,op", _gfx950_atoms())
def test_tr16_model_delivers_the_catalog_coords(dtype, op, role):
    fc = F.operand_coords(op, role)
    kpl = F.check_tr16_coords(fc)
    assert kpl == (op.a_frag_len if role == "a" else op.b_frag_len)
    for lane in range(64):
        assert F.tr16_model_coords(lane, kpl) == fc.int_coords(lane)
    # each lane addresses its own row; lanes never all point at the origin
    addrs = {F.tr16_lane_address(lane, 0, kpl) for lane in range(64)}
    assert len(addrs) == 64 and (0, 0) in addrs


def test_tr16_refuses_maps_it_does_not_deliver():
    c_op = bwd_atoms("gfx950", "fp16")[0]
    k32 = [op for op in bwd_atoms("gfx950", "fp16") if op.k == 32][0]
    plan = F.plan_c_to_a(c_op, k32)
    assert plan.needs_k_perm
    with pytest.raises(ValueError, match="does not deliver"):
        F.check_tr16_coords(F.kperm_b_coords(plan, c_op, k32))
    # the catalog predicate refuses it; the plan chooser takes the stacked plan
    assert F.tr16_read_plan(F.kperm_b_coords(plan, c_op, k32)).kind == "stacked"
    wmma = bwd_atoms("gfx1151", "fp16")[0]
    with pytest.raises(ValueError, match="wave64"):
        F.check_tr16_coords(F.operand_coords(wmma, "b"))
    with pytest.raises(ValueError, match="wave64"):
        F.tr16_read_plan(F.operand_coords(wmma, "b"))


@pytest.mark.parametrize("role", ["a", "b"])
@pytest.mark.parametrize("dtype,op", _gfx950_atoms())
def test_tr16_plan_catalog_equals_legacy_model(dtype, op, role):
    """Every catalog operand takes the catalog plan, which is the legacy
    ``kpl`` model address for address."""
    fc = F.operand_coords(op, role)
    plan = F.tr16_read_plan(fc)
    kpl = F.check_tr16_coords(fc)
    assert plan.kind == "catalog" and plan.g_stride == fc.frag_len == kpl
    assert plan.k_offsets == tuple(4 * t for t in range(kpl // 4))
    for lane in range(64):
        assert F.tr16_plan_coords(lane, plan) == F.tr16_model_coords(lane, kpl)
        assert F.tr16_plan_coords(lane, plan) == fc.int_coords(lane)
        for t in range(kpl // 4):
            assert F.tr16_plan_address(lane, t, plan) == F.tr16_lane_address(
                lane, t, kpl
            )


def _kperm_b(dtype, producer_k):
    ops = {op.k: op for op in bwd_atoms("gfx950", dtype)}
    c_op, k32 = ops[producer_k], ops[32]
    plan = F.plan_c_to_a(c_op, k32)
    assert plan.needs_k_perm
    return plan, F.kperm_b_coords(plan, c_op, k32), ops


@pytest.mark.parametrize("producer_k", [16, 32])
@pytest.mark.parametrize("dtype", DTYPES)
def test_tr16_stacked_plan_delivers_kperm_b(dtype, producer_k):
    """The K-permuted B of a 16x16x32 consumer (producer C of either atom) is
    delivered by the stacked plan: stride 4, reads at K 0 and 16, on every
    lane and slot; the permutation does not depend on the producer atom."""
    plan, coords, _ = _kperm_b(dtype, producer_k)
    tp = F.tr16_read_plan(coords)
    assert (tp.kind, tp.g_stride, tp.k_offsets) == ("stacked", 4, (0, 16))
    for lane in range(64):
        assert F.tr16_plan_coords(lane, tp) == coords.int_coords(lane)
    other, _, _ = _kperm_b(dtype, 48 - producer_k)
    assert other.k_perm == plan.k_perm


@pytest.mark.parametrize("dtype", DTYPES)
def test_stacked_fragment_is_two_catalog_b16_fragments(dtype):
    """Slots 4t..4t+3 of the K-permuted B are the catalog 16x16x16 B fragment
    shifted by 16 t along K, on every lane."""
    _, coords, ops = _kperm_b(dtype, 16)
    b16 = F.operand_coords(ops[16], "b")
    for lane in range(64):
        got = coords.int_coords(lane)
        cat = b16.int_coords(lane)
        for t in range(2):
            assert got[4 * t : 4 * t + 4] == tuple((mn, k + 16 * t) for mn, k in cat)


@dataclasses.dataclass(frozen=True)
class _TableCoords(F.FragCoords):
    """Coordinates given as a per-lane table (any map, affine or not)."""

    table: tuple = ()

    def int_coords(self, lane: int):
        return self.table[lane]


def test_tr16_plan_refuses_near_misses(monkeypatch):
    """A K-permuted map with two slots of one lane swapped, the permuted map
    without the ``kperm`` flag, stacked plans with swapped offsets or the
    catalog stride, and wave32 coordinates are all refused."""
    _, coords, _ = _kperm_b("fp16", 16)
    per = tuple(coords.int_coords(lane) for lane in range(64))
    assert F.tr16_read_plan(coords).kind == "stacked"
    # (a) lane 3: slots 0 and 1 swapped
    lane3 = list(per[3])
    lane3[0], lane3[1] = lane3[1], lane3[0]
    swapped = per[:3] + (tuple(lane3),) + per[4:]
    fake = _TableCoords(
        role="b", frag_len=8, wave_size=64, deltas=coords.deltas,
        base=coords.base, kperm=True, table=swapped,
    )  # fmt: skip
    with pytest.raises(ValueError, match="does not deliver"):
        F.tr16_read_plan(fake)
    assert F.tr16_read_plan(dataclasses.replace(fake, table=per)).kind == "stacked"
    # (d) the permuted map without the flag: the stacked branch requires it
    with pytest.raises(ValueError, match="does not deliver"):
        F.tr16_read_plan(dataclasses.replace(coords, kperm=False))
    # (b) swapped offsets, (c) the catalog stride: the proof fails
    for wrong in (
        F.Tr16ReadPlan("stacked", 4, (16, 0)),
        F.Tr16ReadPlan("stacked", 8, (0, 16)),
        F.Tr16ReadPlan("catalog", 8, (0, 4)),
    ):
        assert not F._tr16_plan_proved(wrong, per), wrong
    for mutant in (
        F.Tr16ReadPlan("stacked", 8, (0, 16)),
        F.Tr16ReadPlan("stacked", 4, (16, 0)),
        F.Tr16ReadPlan("stacked", 4, (0, 8)),
    ):
        monkeypatch.setattr(F, "_STACKED_PLAN", mutant)
        with pytest.raises(ValueError, match="does not deliver"):
            F.tr16_read_plan(coords)
    monkeypatch.undo()
    # (e) wave32
    with pytest.raises(ValueError, match="wave64"):
        F.tr16_read_plan(dataclasses.replace(coords, wave_size=32))
    with pytest.raises(ValueError, match="kind"):
        F.Tr16ReadPlan("diagonal", 4, (0,))


def _tr16_kernel(arch, dtype, op, role, *, mn0=4, k0=8, coords=None):
    elem = F.ir_type(dtype)
    tag = "kperm" if coords is not None else role
    b = IRBuilder(f"tr16_loader_{arch}_{dtype}_k{op.k}_{tag}")
    out = b.param("OUT", PtrType(elem, "global"))
    b.kernel.attrs["max_workgroup_size"] = 64
    lane = b.thread_id_x()
    smem = b.smem_alloc(elem, (64 * 64 + 256,), "flat")
    view = F.LdsView(smem, 128, 64, 64)
    frag = F.load_frag_lds_tr16(
        b, coords or F.operand_coords(op, role), view, lane, dtype=dtype,
        arch=arch, mn0=mn0, k0=k0,
    )  # fmt: skip
    for s in range(frag.type.count):
        idx = b.add(b.mul(lane, b.const_i32(8)), b.const_i32(s))
        b.global_store(out, idx, b.vec_extract(frag, s), align=2)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("dtype,op", _gfx950_atoms())
def test_tr16_loader_emits_only_b64_reads(dtype, op):
    for role in ("a", "b"):
        ir = lower(_tr16_kernel("gfx950", dtype, op, role), arch="gfx950")
        n = op.a_frag_len if role == "a" else op.b_frag_len
        assert ir.count("call <4 x i16> @llvm.amdgcn.ds.read.tr16.b64(") == n // 4
        assert "tr16.b128" not in ir
    if op.k == 32:  # the K-permuted B: two 64-bit reads, never a b128
        _, coords, _ = _kperm_b(dtype, 16)
        kernel = _tr16_kernel("gfx950", dtype, op, "b", coords=coords)
        ir = lower(kernel, arch="gfx950")
        assert ir.count("call <4 x i16> @llvm.amdgcn.ds.read.tr16.b64(") == 2
        assert "tr16.b128" not in ir and "alloca" not in ir


class _AddrBuilder(F.IntLaneBuilder):
    """Integer builder that records the flat index of every transpose read."""

    def __init__(self):
        self.reads = []

    def ds_read_tr16_b64(self, smem, idx, *, dtype):
        self.reads.append(idx)
        return idx


def _emitted_read_indices(coords, *, k0, mn0, monkeypatch):
    monkeypatch.setattr(F, "_assemble", lambda b, chunks, elem: list(chunks))
    view = F.LdsView(None, 128, 64, 64)
    out = []
    for lane in range(64):
        b = _AddrBuilder()
        F.load_frag_lds_tr16(
            b, coords, view, lane, dtype="fp16", arch="gfx950", mn0=mn0, k0=k0
        )
        out.append(tuple(b.reads))
    return out


@pytest.mark.parametrize("which", ["a16", "b16", "a32", "b32", "kperm"])
def test_tr16_loader_addresses_are_the_proved_plan(which, monkeypatch):
    """The loader's emitted per-lane address program, evaluated on integers
    for every lane, is the address of the plan the proof accepted (catalog
    or stacked), read for read."""
    ops = {op.k: op for op in bwd_atoms("gfx950", "fp16")}
    if which == "kperm":
        _, coords, _ = _kperm_b("fp16", 16)
    else:
        coords = F.operand_coords(ops[int(which[1:])], which[0])
    plan = F.tr16_read_plan(coords)
    assert plan.kind == ("stacked" if which == "kperm" else "catalog")
    k0, mn0 = 8, 4
    got = _emitted_read_indices(coords, k0=k0, mn0=mn0, monkeypatch=monkeypatch)
    for lane in range(64):
        want = []
        for t in range(len(plan.k_offsets)):
            k, mn = F.tr16_plan_address(lane, t, plan)
            want.append(128 + (k0 + k) * 64 + mn0 + mn)
        assert got[lane] == tuple(want), lane


@pytest.mark.parametrize("arch", ["gfx942", "gfx1151", "gfx1201"])
def test_tr16_loader_refuses_other_arches(arch):
    op = bwd_atoms("gfx950", "fp16")[0]
    with pytest.raises(ValueError, match="ds_read_tr16_b64"):
        _tr16_kernel(arch, "fp16", op, "b")


def test_tr16_loader_checks_alignment():
    op = bwd_atoms("gfx950", "fp16")[0]
    with pytest.raises(ValueError, match="multiple of 4"):
        _tr16_kernel("gfx950", "fp16", op, "b", mn0=2)


def test_lds_view_index_and_alignment():
    b = IRBuilder("lds_view_probe")
    smem = b.smem_alloc(F16, (1024,), "flat")
    v = F.LdsView(smem, 64, 40, 8)
    assert v.align_elems == 8
    assert F.LdsView(smem, 64, 36, 8).align_elems == 4
    assert F.LdsView(smem, 2, 8, 8).align_elems == 2
    f32 = F.LdsView(smem, 128, 16, 8, scale=2)
    assert f32.align_elems == 8
    with pytest.raises(ValueError):
        F.LdsView(smem, 3, 8, 8, scale=2)
    with pytest.raises(ValueError):
        F.LdsView(smem, 0, 0, 8)
    # the flat index is base + (row * pitch + col) * scale
    from kernels.common._attention_bwd_frag import IntLaneBuilder

    ib = IntLaneBuilder()
    assert F.LdsView(None, 64, 40, 8).index(ib, 3, 5) == 64 + 3 * 40 + 5
    assert F.LdsView(None, 128, 16, 8, scale=2).index(ib, 2, 4) == 128 + 2 * (
        2 * 16 + 4
    )


@pytest.mark.parametrize(
    "pitch,n,pieces", [(64, 8, 1), (68, 8, 2), (66, 8, 4), (65, 8, 8), (68, 4, 1)]
)
def test_lds_view_splits_vectors_wider_than_its_alignment(pitch, n, pieces):
    """A padded row pitch (``"padN"``) lowers the view's alignment; a vector
    access wider than it becomes aligned pieces at consecutive columns (same
    elements, same order), never one access that claims more alignment."""
    b = IRBuilder("lds_split_probe")
    smem = b.smem_alloc(F16, (4096,), "flat")
    v = F.LdsView(smem, 64, pitch, 8)
    width = n // pieces
    row, col = b.const_i32(1), b.const_i32(8)
    ops = b._region_stack[-1].ops
    n0 = len(ops)
    val = v.load(b, row, col, dtype=F16, n=n)
    assert val.type.count == n
    loads = [op for op in ops[n0:] if op.name == "tile.smem_load_vN"]
    assert [op.attrs["vec"] for op in loads] == [width] * pieces
    assert width <= v.align_elems
    n0 = len(ops)
    v.store(b, row, col, b.zero_vec(F16, n), n)
    stores = [op for op in ops[n0:] if op.name.startswith("tile.smem_store")]
    assert len(stores) == pieces
    assert all(op.attrs.get("vec", 1) == width for op in stores)
    if pieces > 1:
        assert all(op.attrs.get("align", 2) == 2 * width for op in stores)
    # an aligned view (the planner default) is one access
    assert v.align_elems >= n or pieces > 1
