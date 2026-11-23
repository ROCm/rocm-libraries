# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The split dQ kernel of the attention backward (CPU only).

* spec value sets, the default geometry per target, the scale placement of
  the dQ kernel, kernel names, the LDS plan and the register model;
* legality: the targets, geometry, the relabel and transpose-read proofs, the
  LDS capacity and the register budgets;
* the key-tile range of a block of query rows and its interior sub-range,
  exhaustively against the reference keep (typed integer evaluation of the
  emitted formulas);
* IR structure of the lowered kernel: the ``rocke.attn_bwd.v3`` dQ kernargs,
  no atomics, ``readfirstlane``'d key-loop trip counts with the trailing
  barrier kept, accumulators that start after the range, no matrix op in a
  divergent region, the model's MFMA count per key step, K / V read only in
  the key loops and Q / dO only before them (behind the liveness branch), the
  per-lane dQ row stores after the loops, and the knob values' emission.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from functools import cache

import numpy as np
import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower
from rocke.helpers.attention_band import AttnRuntimeBounds, BandEmitter
from rocke.numeric.sdpa_reference import keep as ref_keep

from kernels.common._attention_bwd_dqbody import (
    BwdDqGeometry,
    interior_kv_range,
    kv_tile_range,
)
from kernels.common.attention_bwd import attn_bwd_arch_facts, attn_bwd_params
from kernels.common.attention_bwd_dq import (
    DQ_AGPR_ALLOCS,
    DQ_HEAD_LOOPS,
    DQ_KV_PADS,
    DQ_TUNING_FIELDS,
    AttnBwdDqSpec,
    attn_bwd_dq_default_geometry,
    attn_bwd_dq_grid,
    build_attn_bwd_dq,
    dq_lds_plan,
    dq_register_estimates,
    dq_rows_per_head,
    dq_static_scale,
    dq_static_unit_mults,
    validate_attn_bwd_dq_spec,
)

from ._attention_bwd_test_utils import kernel_signature
from ._attention_bwd_int_eval import IntEvalBuilder

ARCHS = ("gfx942", "gfx950")
HEADS = (32, 64, 128)
DTYPES = ("fp16", "bf16")


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


def test_only_head_size_is_required_and_defaults_are_the_shipped_values():
    s = AttnBwdDqSpec(head_size=64)
    assert s.non_default_fields() == ()
    assert s.kernel_name() == "rocke_attn_bwd_dq_fp16_d64_batched_band_sv8"
    assert (s.head_loop, s.stats_prescale, s.edge_tiles, s.kv_pad) == (
        "single",
        True,
        False,
        0,
    )
    assert set(DQ_TUNING_FIELDS) == {
        "waves", "block_m", "block_n", "atom_g02", "atom_g4", "transpose_source",
        "head_loop", "stats_prescale", "edge_tiles", "kv_pad", "kv_prefetch",
        "waves_per_eu", "agpr_alloc", "scheduler_strategy", "sched",
    }  # fmt: skip
    assert s.sched == "none"


@pytest.mark.parametrize(
    "kw",
    [
        dict(head_size=96),
        dict(head_size=64, dtype="fp32"),
        dict(head_size=64, seq_mode="paged"),
        dict(head_size=64, stage_vec=4),
        dict(head_size=64, mask_class="causal"),
        dict(head_size=64, ws_layout="row"),
        dict(head_size=64, static_scale="dq_before_atomic"),
        dict(head_size=64, waves=3),
        dict(head_size=64, block_m=48),
        dict(head_size=64, block_n=96),
        dict(head_size=64, atom_g02="32x32x8"),
        dict(head_size=64, atom_g4="wmma16x16x16"),
        dict(head_size=64, transpose_source="xt_lds"),
        dict(head_size=64, head_loop="kv"),
        dict(head_size=64, stats_prescale=1),
        dict(head_size=64, edge_tiles=None),
        dict(head_size=64, kv_pad=4),
        dict(head_size=64, kv_prefetch="yes"),
        dict(head_size=64, waves_per_eu=3),
        dict(head_size=64, agpr_alloc=(64, 64)),
        dict(head_size=64, scheduler_strategy="not-a-strategy"),
        dict(head_size=64, sched="stage_table_atomics"),
        dict(head_size=64, xcd_remap=1),
        dict(head_size=64, name="bad name"),
    ],
    ids=lambda kw: "-".join(k for k in kw if k != "head_size") or "head",
)
def test_bad_values_raise(kw):
    with pytest.raises(ValueError):
        AttnBwdDqSpec(**kw)


def test_value_sets_are_accepted():
    for loop in DQ_HEAD_LOOPS:
        AttnBwdDqSpec(head_size=64, head_loop=loop)
    for pad in DQ_KV_PADS:
        AttnBwdDqSpec(head_size=64, kv_pad=pad)
    for a in DQ_AGPR_ALLOCS:
        AttnBwdDqSpec(head_size=64, agpr_alloc=a)
    assert AttnBwdDqSpec(head_size=64, agpr_alloc=[128, 128]).agpr_alloc == (128, 128)


def test_scale_placement_of_the_dq_kernel():
    assert dq_static_scale("fold_ds") == "fold_ds"
    assert dq_static_scale("convert") == "convert"
    assert dq_static_scale("dq_before_atomic") == "convert"
    assert dq_static_scale("runtime") == "runtime"
    with pytest.raises(ValueError):
        dq_static_scale("other")
    assert dq_static_unit_mults("fold_ds") == {"dq_mult"}
    assert dq_static_unit_mults("convert") == {"ds_mult"}
    assert dq_static_unit_mults("runtime") == frozenset()


def test_kernel_name_salts_every_non_default_field():
    base = AttnBwdDqSpec(head_size=128, dtype="bf16", seq_mode="thd", stage_vec=1)
    assert base.kernel_name() == "rocke_attn_bwd_dq_bf16_d128_thd_band_sv1"
    names = {base.kernel_name()}
    for kw in (
        dict(head_loop="group"), dict(edge_tiles=True), dict(stats_prescale=False),
        dict(kv_pad=16), dict(static_scale="runtime"), dict(xcd_remap=True),
        dict(ws_layout="token_major"), dict(waves=2), dict(block_n=32),
    ):  # fmt: skip
        names.add(dataclasses.replace(base, **kw).kernel_name())
    assert len(names) == 10
    assert (
        AttnBwdDqSpec(head_size=64, mask_class="none")
        .kernel_name()
        .endswith("_none_sv8")
    )


# ---------------------------------------------------------------------------
# default geometry and legality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_default_geometry_is_legal_everywhere(arch):
    facts = attn_bwd_arch_facts(arch)
    for d, dt, sv in itertools.product(HEADS, DTYPES, (8, 1)):
        g = attn_bwd_dq_default_geometry(arch, d, dt, sv)
        assert g["transpose_source"] == (
            "tr_read" if facts.has_tr_read else "lds_plain"
        )
        assert g["atom_g4"] == "16x16x16"
        s = validate_attn_bwd_dq_spec(
            AttnBwdDqSpec(head_size=d, dtype=dt, stage_vec=sv), arch
        )
        # VGPR-form MFMA at the start points (no AGPR copy in the key loop)
        assert s.agpr_alloc == (0, 0)
        est = dq_register_estimates(s, arch)
        assert est["agpr"] == 0 and est["arch_vgpr"] <= facts.arch_vgprs
        plan = dq_lds_plan(s, arch)
        assert plan.fits
        # two CTAs per CU fit the LDS of every target
        assert 2 * plan.total_bytes <= facts.lds_capacity_bytes


@pytest.mark.parametrize("arch", ("gfx1151", "gfx1201", "gfx90a"))
def test_other_targets_are_refused(arch):
    with pytest.raises(ValueError, match="split dQ kernel is built for"):
        validate_attn_bwd_dq_spec(AttnBwdDqSpec(head_size=64), arch)
    with pytest.raises(ValueError, match="split dQ kernel is built for"):
        attn_bwd_dq_default_geometry(arch, 64)


def test_geometry_and_proof_rules():
    ok = validate_attn_bwd_dq_spec(AttnBwdDqSpec(head_size=64), "gfx942")
    assert ok.block_m % (16 * ok.waves) == 0
    bad = [
        ("gfx942", dict(waves=8, block_m=64)),  # kM0 % (16 W)
        ("gfx942", dict(atom_g02="16x16x32")),  # no native K = 32 atom
        ("gfx942", dict(transpose_source="tr_read")),  # no transpose read
        ("gfx950", dict(atom_g4="16x16x32")),  # K-permuted B with tr_read
        ("gfx942", dict(head_size=128, block_n=256)),  # K + V exceed 64 KiB
        ("gfx942", dict(waves=1, block_m=128, block_n=256, head_size=128)),  # regs
        ("gfx942", dict(agpr_alloc=(0, 0), waves=1, block_m=32, block_n=256)),
        ("gfx942", dict(waves_per_eu=2, waves=1, block_m=32, block_n=128)),
    ]
    for arch, kw in bad:
        kw = {"head_size": 64, **kw}
        with pytest.raises(ValueError):
            validate_attn_bwd_dq_spec(AttnBwdDqSpec(**kw), arch)
    # the K-permuted G4' atom with the plain transpose read, and padded K + V
    # rows that need the gfx950 LDS
    padded = dict(head_size=128, block_n=128, kv_pad=32)
    with pytest.raises(ValueError, match="LDS plan"):
        validate_attn_bwd_dq_spec(AttnBwdDqSpec(**padded), "gfx942")
    for arch, kw in (
        ("gfx950", dict(atom_g4="16x16x32", transpose_source="lds_plain")),
        ("gfx950", padded),
    ):
        s = validate_attn_bwd_dq_spec(AttnBwdDqSpec(**{"head_size": 64, **kw}), arch)
        geom = BwdDqGeometry.from_spec(s, arch)
        if s.atom_g4 == "16x16x32":
            assert geom.relabel_plan.needs_k_perm and geom.b4_coords.kperm
    with pytest.raises(ValueError, match="K-permuted"):
        validate_attn_bwd_dq_spec(
            AttnBwdDqSpec(head_size=64, atom_g4="16x16x32"), "gfx950"
        )


def test_lds_plan_aliases_the_three_phases():
    s = validate_attn_bwd_dq_spec(AttnBwdDqSpec(head_size=128, kv_pad=16), "gfx950")
    plan = dq_lds_plan(s, "gfx950")
    q_off, q_bytes, _, _ = plan.buffer("q")
    do_off, do_bytes, _, _ = plan.buffer("do")
    k_off, k_bytes, k_pitch, k_rows = plan.buffer("k")
    v_off, v_bytes, _, _ = plan.buffer("v")
    out_off, out_bytes, _, _ = plan.buffer("out")
    assert q_off == k_off == out_off == 0
    assert k_pitch == 128 + 8 and k_rows == s.block_n
    assert k_bytes == s.block_n * (128 * 2 + 16)
    assert do_off >= q_bytes and v_off >= k_bytes
    assert all(x % 128 == 0 for x in (do_off, v_off))
    assert plan.total_bytes == max(do_off + do_bytes, v_off + v_bytes, out_bytes)
    with pytest.raises(KeyError):
        plan.buffer("ds")  # dS never goes through LDS


def test_register_model():
    arch = "gfx942"
    s = validate_attn_bwd_dq_spec(AttnBwdDqSpec(head_size=128), arch)
    w, km, kn, d = s.waves, s.block_m, s.block_n, s.head_size
    cols = km // w
    want_v = (
        24 + 2 * cols * d // 128 + kn * cols // 128 + kn * 16 // 128 + d * 16 // 128
        + 6 * (cols // 16) + 2 * kn * cols // 64 + cols * d // 64
    )  # fmt: skip
    assert dq_register_estimates(s, arch) == {"arch_vgpr": want_v, "agpr": 0}
    r = validate_attn_bwd_dq_spec(dataclasses.replace(s, agpr_alloc=(128, 128)), arch)
    est = dq_register_estimates(r, arch)
    assert est["agpr"] == 2 * kn * cols // 64 + cols * d // 64
    assert est["arch_vgpr"] + est["agpr"] == want_v


def test_rows_per_head_and_grid():
    s = validate_attn_bwd_dq_spec(AttnBwdDqSpec(head_size=64), "gfx942")
    assert dq_rows_per_head(s, 4) == s.block_m
    assert attn_bwd_dq_grid(s, s_q_max=100, h_q=8, group=4, batch=3) == (
        -(-100 // s.block_m),
        8,
        3,
    )
    g = dataclasses.replace(s, head_loop="group")
    assert dq_rows_per_head(g, 4) == s.block_m // 4
    assert attn_bwd_dq_grid(g, s_q_max=100, h_q=8, group=4, batch=3) == (
        -(-100 // (s.block_m // 4)),
        2,
        3,
    )
    assert attn_bwd_dq_grid(g, s_q_max=0, h_q=8, group=4, batch=3)[0] == 1
    with pytest.raises(ValueError, match="divide"):
        dq_rows_per_head(g, 3)


# ---------------------------------------------------------------------------
# key-tile range of a block of query rows (exhaustive)
# ---------------------------------------------------------------------------

BOUNDS = (-1, 0, 1, 3, 7)


def _band(b, len_q, len_kv, br, left, right, runtime):
    off = len_kv - len_q if br else 0
    if runtime:
        bounds = AttnRuntimeBounds(
            b.const_i32(len_q), b.const_i32(len_kv), b.const_i32(off),
            b.const_i32(left), b.const_i32(right),
        )  # fmt: skip
    else:
        bounds = AttnRuntimeBounds(
            b.const_i32(len_q), b.const_i32(len_kv), off, left, right
        )
    return BandEmitter(b, bounds)


def _keep(len_q, len_kv, br, left, right):
    diag = "bottom_right" if br else "top_left"

    def keep(q, k):
        return (
            0 <= q < len_q
            and 0 <= k < len_kv
            and ref_keep(q, k, len_q, len_kv, diag, left, right)
        )

    return keep


def _check_kv_ranges(b, len_q, len_kv, br, left, right, rows, k_n0, runtime):
    band = _band(b, len_q, len_kv, br, left, right, runtime)
    keep = _keep(len_q, len_kv, br, left, right)
    n_interior = 0
    for q0 in range(0, len_q + rows + 1, max(1, rows // 2)):
        n_rows = b.const_i32(rows) if runtime else rows
        ks, n = kv_tile_range(b, band, b.const_i32(q0), n_rows, k_n0=k_n0)
        ks, n = ks.v, n.v
        tiles = sorted(
            {
                k // k_n0
                for q in range(q0, q0 + rows)
                for k in range(len_kv)
                if keep(q, k)
            }
        )
        case = (len_q, len_kv, br, left, right, rows, k_n0, q0)
        if not tiles:
            assert n == 0, case
            continue
        # exact: the visited tiles are the tiles holding a kept cell, contiguous
        assert (ks, n) == (tiles[0], len(tiles)), case
        assert tiles == list(range(tiles[0], tiles[-1] + 1)), case
        i0, i1 = interior_kv_range(
            b, band, b.const_i32(q0), n_rows, b.const_i32(ks), b.const_i32(n),
            k_n0=k_n0,
        )  # fmt: skip
        interior = {
            kt
            for kt in tiles
            if all(
                keep(q, k)
                for q in range(q0, q0 + rows)
                for k in range(kt * k_n0, kt * k_n0 + k_n0)
            )
        }
        assert ks <= i0.v <= i1.v <= ks + n, case
        assert set(range(i0.v, i1.v)) == interior, case
        n_interior += len(interior)
    return n_interior


def test_kv_tile_range_exhaustive():
    """len_q, len_kv in 0..33, both diagonals, bounds in {-1, 0, 1, 3, 7},
    16 rows and 16 keys per tile, runtime bound spellings."""
    b = IntEvalBuilder()
    n_int = 0
    for len_q, len_kv in itertools.product(range(0, 34, 3), range(0, 34, 3)):
        for br, left, right in itertools.product((False, True), BOUNDS, BOUNDS):
            n_int += _check_kv_ranges(b, len_q, len_kv, br, left, right, 16, 16, True)
    assert n_int > 0


@pytest.mark.parametrize("rows,k_n0", [(1, 16), (4, 8), (16, 64), (64, 16), (3, 5)])
@pytest.mark.parametrize("runtime", [False, True])
def test_kv_tile_range_other_tiles_and_spellings(rows, k_n0, runtime):
    b = IntEvalBuilder()
    sizes = (0, 1, 2, 15, 17, 40, 65)
    for len_q, len_kv in itertools.product(sizes, sizes):
        for br, left, right in itertools.product((False, True), BOUNDS, BOUNDS):
            _check_kv_ranges(b, len_q, len_kv, br, left, right, rows, k_n0, runtime)


def test_unmasked_range_is_every_key_tile_of_live_rows():
    b = IntEvalBuilder()
    for len_q, len_kv, rows in ((10, 70, 16), (40, 33, 16), (0, 20, 16), (20, 0, 4)):
        band = BandEmitter(
            b, AttnRuntimeBounds(b.const_i32(len_q), b.const_i32(len_kv))
        )
        for q0 in range(0, len_q + rows, rows):
            ks, n = kv_tile_range(b, band, b.const_i32(q0), rows, k_n0=16)
            live = q0 < len_q and len_kv > 0
            assert (ks.v, n.v) == ((0, -(-len_kv // 16)) if live else (ks.v, 0))
            i0, i1 = interior_kv_range(
                b, band, b.const_i32(q0), rows, ks, n, k_n0=16, masked=False
            )
            whole = q0 + rows <= len_q
            assert i1.v - i0.v == (len_kv // 16 if live and whole else 0)


# ---------------------------------------------------------------------------
# IR structure
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {},
    "group": {"head_loop": "group"},
    "edge": {"edge_tiles": True},
    "group_edge": {"head_loop": "group", "edge_tiles": True},
    "noprescale": {"stats_prescale": False},
    "runtime_scale": {"static_scale": "runtime"},
    "convert_scale": {"static_scale": "convert"},
    "xcd": {"xcd_remap": True},
    "kvpad16": {"kv_pad": 16},
    "w2": {"waves": 2, "block_m": 32, "block_n": 32},
    "agpr128": {"agpr_alloc": (128, 128)},
    "wpe2_minreg": {"waves_per_eu": 2, "scheduler_strategy": "iterative-minreg"},
    "prefetch": {"kv_prefetch": True},
    "prefetch_group_edge": {
        "kv_prefetch": True,
        "head_loop": "group",
        "edge_tiles": True,
    },
}
ALL_ROWS = [
    (arch, d, dt, sm, mc, sv, var)
    for arch in ARCHS
    for d in HEADS
    for dt, sm, mc, sv in (
        ("fp16", "batched", "band", 8),
        ("bf16", "thd", "none", 1),
    )
    for var in VARIANTS
]


def _spec(arch, d, dt, sm, mc, sv, var):
    return AttnBwdDqSpec(
        head_size=d, dtype=dt, seq_mode=sm, mask_class=mc, stage_vec=sv,
        **VARIANTS[var],
    )  # fmt: skip


def _refused(row) -> bool:
    try:
        validate_attn_bwd_dq_spec(_spec(*row), row[0])
    except ValueError:
        return True
    return False


# The validator refuses exactly the spilling lever combinations
# (DQ_SPILLING_COMBOS: kv_prefetch + head_loop = "group" + edge_tiles at d128 on
# gfx950); every other row is built.
REFUSED_ROWS = [r for r in ALL_ROWS if _refused(r)]
INSTANCES = [r for r in ALL_ROWS if r not in REFUSED_ROWS]


@cache
def _kernel(row):
    return build_attn_bwd_dq(_spec(*row), arch=row[0])


@cache
def _ir(row, flavor="llvm22"):
    return lower(_kernel(row), arch=row[0], llvm_flavor=flavor)


def _walk(ops, ancestors=()):
    for op in ops:
        yield op, ancestors
        for r in op.regions:
            yield from _walk(r.ops, ancestors + (op,))


def _key_loops(kernel):
    return [
        op
        for op, _ in _walk(kernel.body.ops)
        if op.name == "scf.for"
        and any(o.name == "tile.mma" for o, _ in _walk(op.regions[0].ops))
    ]


def _derived(kernel, names):
    """The pointers derived from the named kernargs: pointer arithmetic and
    the opaque copies that pin a slab pointer in an SGPR."""
    out = {f"%{n}" for n in names}
    for op, _ in _walk(kernel.body.ops):
        if op.name not in ("tile.global_ptr_add", "tile.inline_asm"):
            continue
        if op.operands and op.operands[0].name in out:
            out.add(op.results[0].name)
    return out


def test_slab_pointers_are_pinned_before_the_key_loops():
    """K, V and dQ each reach the key loops as one opaque 64-bit pointer
    (the empty-asm SGPR copy), formed before the first key loop."""
    for var in ("default", "group_edge", "prefetch_group_edge"):
        row = ("gfx950", 64, "bf16", "thd", "none", 1, var)
        kernel = _kernel(row)
        order = {id(op): i for i, (op, _) in enumerate(_walk(kernel.body.ops))}
        first = _key_loops(kernel)[0]
        # SGPR copies (the rounding point is a VGPR one, "=v,0")
        pins = [
            op
            for op, _ in _walk(kernel.body.ops)
            if op.name == "tile.inline_asm" and op.attrs["constraints"] == "=s,0"
        ]
        assert len(pins) == 3, var
        for op, name in zip(pins, ("K", "V", "dQ")):
            assert op.attrs["template"] == "" and not op.attrs["sideeffect"]
            assert op.operands[0].name in _derived(kernel, (name,)), (var, name)
            assert order[id(op)] < order[id(first)]


def _ids(row):
    return "-".join(map(str, row))


@pytest.mark.parametrize("row", INSTANCES, ids=_ids)
def test_builds_lowers_and_keeps_the_abi(row):
    ir = _ir(row)
    assert "alloca" not in ir and "addrspace(5)" not in ir
    assert "atomicrmw" not in ir
    sig = kernel_signature(ir)
    assert [(n, k) for n, k, _ in sig] == list(attn_bwd_params("dq"))
    for name, kind, text in sig:
        if kind == "Q":
            assert ("readonly" in text) == (name != "dQ"), text
            assert "noalias" not in text
    r = validate_attn_bwd_dq_spec(_spec(*row), row[0])
    assert _kernel(row).name == r.kernel_name()


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("flavor", ("llvm20", "llvm22", "llvm23"))
def test_lowers_at_every_flavor(arch, flavor):
    for var in ("default", "group_edge"):
        ir = _ir((arch, 64, "bf16", "batched", "band", 8, var), flavor)
        assert "define amdgpu_kernel void @" in ir


@pytest.mark.parametrize("row", INSTANCES, ids=_ids)
def test_structure(row):
    arch = row[0]
    kernel = _kernel(row)
    r = validate_attn_bwd_dq_spec(_spec(*row), arch)
    geom = BwdDqGeometry.from_spec(r, arch)
    loops = _key_loops(kernel)
    assert len(loops) == (3 if r.edge_tiles else 1)
    order = {id(op): i for i, (op, _) in enumerate(_walk(kernel.body.ops))}
    kv = _derived(kernel, ("K", "V"))
    qdo = _derived(kernel, ("Q", "dO"))
    dq = _derived(kernel, ("dQ",))
    model = 2 * geom.g0t.mma_per_wave + geom.g4t.mma_per_wave
    first = loops[0]
    for loop in loops:
        upper = loop.operands[1]
        assert upper.op is not None and upper.op.name == "tile.readfirstlane"
        assert loop.attrs.get("elide_trailing_barrier") is False
        body = list(_walk(loop.regions[0].ops))
        assert sum(op.name == "tile.mma" for op, _ in body) == model
        # one barrier publishes K / V, one retires their reads (with
        # kv_prefetch: one retires the reads, one publishes the next tile)
        assert sum(op.name == "tile.sync" for op, _ in body) == 2
        kv_in = [
            op
            for op, _ in body
            if op.name.startswith("memref.global_load") and op.operands[0].name in kv
        ]
        if r.kv_prefetch:
            # the next tile's loads come before the step's first matrix op
            first_mma = next(
                i for i, (op, _) in enumerate(body) if op.name == "tile.mma"
            )
            ids = {id(op): i for i, (op, _) in enumerate(body)}
            assert kv_in and all(ids[id(op)] < first_mma for op in kv_in)
        for op, _ in body:
            assert not op.name.startswith("memref.global_atomic")
            if op.name.startswith("memref.global_store"):
                raise AssertionError("a global store inside the key loop")
    for op, anc in _walk(kernel.body.ops):
        if op.name == "tile.mma":
            assert not any(a.name in ("scf.if", "scf.if_else") for a in anc)
        in_loop = any(a is lp for a in anc for lp in loops)
        if op.name.startswith("memref.global_load") and op.operands[0].name in kv:
            if in_loop:
                continue  # K / V per key step
            # kv_prefetch: the range's first tile, before the loops, behind
            # the non-empty-range branch
            assert r.kv_prefetch and any(a.name == "scf.if" for a in anc)
            assert order[id(op)] < order[id(first)]
        if op.name.startswith("memref.global_load") and op.operands[0].name in qdo:
            # Q / dO once, before the loops, behind the CTA liveness branch
            assert not in_loop
            assert any(a.name == "scf.if" for a in anc)
            assert order[id(op)] < order[id(first)]
        if op.name.startswith("memref.global_store"):
            assert op.operands[0].name in dq, op.name
            assert order[id(op)] > max(order[id(lp)] for lp in loops)
    # the accumulators start after the trip counts
    inits = first.operands[3:]
    assert len(inits) == geom.g4t.frags_m * geom.g4t.frags_n
    for init in inits:
        assert order[id(init.op)] > order[id(first.operands[1].op)]
    # dQ leaves as rows of 8 elements: vector stores (stage_vec 8) or element
    # stores (stage_vec 1), one row chunk per thread per pass
    stores = [
        op
        for op, _ in _walk(kernel.body.ops)
        if op.name.startswith("memref.global_store")
    ]
    chunks = -(-(r.block_m * r.head_size // 8) // geom.threads)
    per_chunk = 1 if r.stage_vec == 8 else 8
    assert len(stores) == chunks * per_chunk


@pytest.mark.parametrize("arch", ARCHS)
def test_knob_emission(arch):
    """Each knob value changes the IR it owns, and only where it is built."""

    def ir(var, **kw):
        row = (arch, 64, "fp16", "batched", "band", 8, var)
        text = _ir(row)
        return text.replace(
            validate_attn_bwd_dq_spec(_spec(*row), arch).kernel_name(), "@K"
        )

    base = ir("default")
    # the XCD remap reads xcd_n / xcd_chunk only when built
    for name in ("xcd_n", "xcd_chunk"):
        uses = re.compile(rf"%{name}\b")
        assert len(uses.findall(base)) == 1
        assert len(uses.findall(ir("xcd"))) > 1
    # the head group G only under head_loop = "group"
    uses = re.compile(r"%G\b")
    assert len(uses.findall(base)) == 1 and len(uses.findall(ir("group"))) > 1
    # fold_ds: ds_mult applied, dq_mult not; convert: the reverse; runtime: both
    for var, ds, dqm in (
        ("default", True, False),
        ("convert_scale", False, True),
        ("runtime_scale", True, True),
    ):
        text = ir(var)
        assert (len(re.findall(r"%ds_mult\b", text)) > 1) == ds, var
        assert (len(re.findall(r"%dq_mult\b", text)) > 1) == dqm, var
    # prescale: one fma per cell instead of the separate subtract and multiply
    assert base.count("llvm.fmuladd") > ir("noprescale").count("llvm.fmuladd")
    assert ir("agpr128") != base and '"amdgpu-agpr-alloc"="128' in ir("agpr128")


def test_relabel_and_transpose_proofs_per_atom():
    for arch, a02, a4, ts, ok in (
        ("gfx942", "16x16x16", "16x16x16", "lds_plain", True),
        ("gfx950", "16x16x32", "16x16x16", "tr_read", True),
        ("gfx950", "16x16x16", "16x16x16", "tr_read", True),
        ("gfx950", "16x16x32", "16x16x32", "lds_plain", True),
        ("gfx950", "16x16x32", "16x16x32", "tr_read", False),
    ):
        spec = AttnBwdDqSpec(
            head_size=64, atom_g02=a02, atom_g4=a4, transpose_source=ts
        )
        if ok:
            s = validate_attn_bwd_dq_spec(spec, arch)
            geom = BwdDqGeometry.from_spec(s, arch)
            assert geom.relabel_plan.kind == "direct"
            assert geom.reader == ("tr16" if ts == "tr_read" else "plain")
        else:
            with pytest.raises(ValueError):
                validate_attn_bwd_dq_spec(spec, arch)


def test_validator_refuses_exactly_the_spilling_lever_combinations():
    """kv_prefetch with head_loop = "group" and edge_tiles at d128 on gfx950
    spills VGPRs to scratch on the production compiler (measured on both
    variant rows; each lever alone and group + edge are clean there): the
    validator refuses exactly that combination at the start tile."""
    from kernels.common.attention_bwd_dq import DQ_SPILLING_COMBOS

    assert DQ_SPILLING_COMBOS == {("gfx950", 128, 4, 64, 64, True, "group", True)}
    assert REFUSED_ROWS == [
        ("gfx950", 128, "fp16", "batched", "band", 8, "prefetch_group_edge"),
        ("gfx950", 128, "bf16", "thd", "none", 1, "prefetch_group_edge"),
    ]
    for row in REFUSED_ROWS:
        with pytest.raises(ValueError, match="spills VGPRs to scratch"):
            validate_attn_bwd_dq_spec(_spec(*row), row[0])
    # one lever less, another arch, head size or tile: built
    base = dict(head_size=128, kv_prefetch=True, head_loop="group", edge_tiles=True)
    validate_attn_bwd_dq_spec(AttnBwdDqSpec(**base), "gfx942")
    validate_attn_bwd_dq_spec(AttnBwdDqSpec(**dict(base, head_size=64)), "gfx950")
    for drop in ("kv_prefetch", "edge_tiles"):
        validate_attn_bwd_dq_spec(
            AttnBwdDqSpec(**dict(base, **{drop: False})), "gfx950"
        )
    validate_attn_bwd_dq_spec(AttnBwdDqSpec(**dict(base, head_loop="single")), "gfx950")
    validate_attn_bwd_dq_spec(AttnBwdDqSpec(**dict(base, block_n=32)), "gfx950")


# dq_scheduler_strategy = "iterative-ilp" crash map of the production compiler:
# every problem shape (dtype x head size x layout x mask x stage_vec x {plain,
# edge tiles, group head loop, K / V prefetch}) at the default dQ tile and the
# 8-wave 128 x 32 tile; the rows whose compile crashes comgr (segfault).
ILP_TILES = {"default": {}, "s3": dict(waves=8, block_m=128, block_n=32, kv_pad=8)}
ILP_VARIANTS = {
    "plain": {},
    "edge": {"edge_tiles": True},
    "group": {"head_loop": "group"},
    "prefetch": {"kv_prefetch": True},
}
ILP_CRASH_ROWS = (
    ('gfx942', 'default', 'bf16', 64, 'batched', 'band', 1, 'edge'),
    ('gfx942', 'default', 'bf16', 64, 'batched', 'band', 1, 'group'),
    ('gfx942', 'default', 'bf16', 64, 'batched', 'band', 1, 'plain'),
    ('gfx942', 'default', 'bf16', 64, 'batched', 'band', 8, 'edge'),
    ('gfx942', 'default', 'bf16', 64, 'batched', 'band', 8, 'group'),
    ('gfx942', 'default', 'bf16', 64, 'batched', 'band', 8, 'plain'),
    ('gfx942', 'default', 'bf16', 64, 'batched', 'none', 1, 'edge'),
    ('gfx942', 'default', 'bf16', 64, 'batched', 'none', 8, 'edge'),
    ('gfx942', 'default', 'bf16', 64, 'thd', 'band', 1, 'edge'),
    ('gfx942', 'default', 'bf16', 64, 'thd', 'band', 1, 'group'),
    ('gfx942', 'default', 'bf16', 64, 'thd', 'band', 1, 'plain'),
    ('gfx942', 'default', 'bf16', 64, 'thd', 'band', 8, 'edge'),
    ('gfx942', 'default', 'bf16', 64, 'thd', 'band', 8, 'group'),
    ('gfx942', 'default', 'bf16', 64, 'thd', 'band', 8, 'plain'),
    ('gfx942', 'default', 'bf16', 64, 'thd', 'none', 1, 'edge'),
    ('gfx942', 'default', 'bf16', 64, 'thd', 'none', 8, 'edge'),
    ('gfx942', 'default', 'bf16', 128, 'batched', 'band', 1, 'prefetch'),
    ('gfx942', 'default', 'bf16', 128, 'batched', 'band', 8, 'prefetch'),
    ('gfx942', 'default', 'bf16', 128, 'batched', 'none', 1, 'prefetch'),
    ('gfx942', 'default', 'bf16', 128, 'batched', 'none', 8, 'prefetch'),
    ('gfx942', 'default', 'bf16', 128, 'thd', 'band', 1, 'prefetch'),
    ('gfx942', 'default', 'bf16', 128, 'thd', 'band', 8, 'prefetch'),
    ('gfx942', 'default', 'bf16', 128, 'thd', 'none', 1, 'prefetch'),
    ('gfx942', 'default', 'bf16', 128, 'thd', 'none', 8, 'prefetch'),
    ('gfx942', 'default', 'fp16', 64, 'batched', 'band', 1, 'edge'),
    ('gfx942', 'default', 'fp16', 64, 'batched', 'band', 1, 'group'),
    ('gfx942', 'default', 'fp16', 64, 'batched', 'band', 1, 'plain'),
    ('gfx942', 'default', 'fp16', 64, 'batched', 'band', 8, 'edge'),
    ('gfx942', 'default', 'fp16', 64, 'batched', 'band', 8, 'group'),
    ('gfx942', 'default', 'fp16', 64, 'batched', 'band', 8, 'plain'),
    ('gfx942', 'default', 'fp16', 64, 'batched', 'none', 1, 'edge'),
    ('gfx942', 'default', 'fp16', 64, 'batched', 'none', 8, 'edge'),
    ('gfx942', 'default', 'fp16', 64, 'thd', 'band', 1, 'edge'),
    ('gfx942', 'default', 'fp16', 64, 'thd', 'band', 1, 'group'),
    ('gfx942', 'default', 'fp16', 64, 'thd', 'band', 1, 'plain'),
    ('gfx942', 'default', 'fp16', 64, 'thd', 'band', 8, 'edge'),
    ('gfx942', 'default', 'fp16', 64, 'thd', 'band', 8, 'group'),
    ('gfx942', 'default', 'fp16', 64, 'thd', 'band', 8, 'plain'),
    ('gfx942', 'default', 'fp16', 64, 'thd', 'none', 1, 'edge'),
    ('gfx942', 'default', 'fp16', 64, 'thd', 'none', 8, 'edge'),
    ('gfx942', 'default', 'fp16', 128, 'batched', 'band', 1, 'prefetch'),
    ('gfx942', 'default', 'fp16', 128, 'batched', 'band', 8, 'prefetch'),
    ('gfx942', 'default', 'fp16', 128, 'batched', 'none', 1, 'prefetch'),
    ('gfx942', 'default', 'fp16', 128, 'batched', 'none', 8, 'prefetch'),
    ('gfx942', 'default', 'fp16', 128, 'thd', 'band', 1, 'prefetch'),
    ('gfx942', 'default', 'fp16', 128, 'thd', 'band', 8, 'prefetch'),
    ('gfx942', 'default', 'fp16', 128, 'thd', 'none', 1, 'prefetch'),
    ('gfx942', 'default', 'fp16', 128, 'thd', 'none', 8, 'prefetch'),
    ('gfx950', 'default', 'bf16', 64, 'batched', 'band', 1, 'prefetch'),
    ('gfx950', 'default', 'bf16', 64, 'batched', 'band', 8, 'prefetch'),
    ('gfx950', 'default', 'bf16', 64, 'thd', 'band', 1, 'prefetch'),
    ('gfx950', 'default', 'bf16', 64, 'thd', 'band', 8, 'prefetch'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'band', 1, 'edge'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'band', 1, 'group'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'band', 1, 'plain'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'band', 1, 'prefetch'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'band', 8, 'edge'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'band', 8, 'group'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'band', 8, 'plain'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'band', 8, 'prefetch'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'none', 1, 'edge'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'none', 1, 'group'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'none', 1, 'plain'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'none', 1, 'prefetch'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'none', 8, 'edge'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'none', 8, 'group'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'none', 8, 'plain'),
    ('gfx950', 'default', 'bf16', 128, 'batched', 'none', 8, 'prefetch'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'band', 1, 'edge'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'band', 1, 'group'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'band', 1, 'plain'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'band', 1, 'prefetch'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'band', 8, 'edge'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'band', 8, 'group'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'band', 8, 'plain'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'band', 8, 'prefetch'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'none', 1, 'edge'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'none', 1, 'group'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'none', 1, 'plain'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'none', 1, 'prefetch'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'none', 8, 'edge'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'none', 8, 'group'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'none', 8, 'plain'),
    ('gfx950', 'default', 'bf16', 128, 'thd', 'none', 8, 'prefetch'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'band', 1, 'edge'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'band', 1, 'group'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'band', 1, 'plain'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'band', 1, 'prefetch'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'band', 8, 'edge'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'band', 8, 'group'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'band', 8, 'plain'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'band', 8, 'prefetch'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'none', 1, 'edge'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'none', 1, 'group'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'none', 1, 'plain'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'none', 1, 'prefetch'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'none', 8, 'edge'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'none', 8, 'group'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'none', 8, 'plain'),
    ('gfx950', 's3', 'bf16', 128, 'batched', 'none', 8, 'prefetch'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'band', 1, 'edge'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'band', 1, 'group'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'band', 1, 'plain'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'band', 1, 'prefetch'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'band', 8, 'edge'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'band', 8, 'group'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'band', 8, 'plain'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'band', 8, 'prefetch'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'none', 1, 'edge'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'none', 1, 'group'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'none', 1, 'plain'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'none', 1, 'prefetch'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'none', 8, 'edge'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'none', 8, 'group'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'none', 8, 'plain'),
    ('gfx950', 's3', 'bf16', 128, 'thd', 'none', 8, 'prefetch'),
)  # fmt: skip


# dQ specs of device-test cells that spill to scratch on (or crash) the
# production compiler: (arch, non-default resolved fields) and the failing rule.
DQ_SPILL_ROWS = (
    (("gfx942", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x16'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('dtype', 'bf16'), ('head_size', 64), ('scheduler_strategy', 'iterative-maxocc'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx942", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x16'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('dtype', 'bf16'), ('head_size', 128), ('scheduler_strategy', 'iterative-maxocc'), ('seq_mode', 'thd'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx942", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x16'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('dtype', 'bf16'), ('head_size', 64), ('scheduler_strategy', 'iterative-maxocc'), ('seq_mode', 'thd'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx942", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x16'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('dtype', 'bf16'), ('head_size', 64), ('mask_class', 'none'), ('scheduler_strategy', 'iterative-maxocc'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx942", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x16'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('head_size', 64), ('scheduler_strategy', 'iterative-maxocc'), ('seq_mode', 'thd'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx942", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x16'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('dtype', 'bf16'), ('head_size', 128), ('mask_class', 'none'), ('scheduler_strategy', 'iterative-maxocc'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill+sgpr_spill"),
    (("gfx942", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x16'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('dtype', 'bf16'), ('head_size', 128), ('scheduler_strategy', 'iterative-maxocc'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill+sgpr_spill"),
    (("gfx942", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x16'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('head_size', 64), ('mask_class', 'none'), ('scheduler_strategy', 'iterative-maxocc'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx942", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x16'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('head_size', 64), ('scheduler_strategy', 'iterative-maxocc'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx942", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x16'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('head_size', 128), ('scheduler_strategy', 'iterative-maxocc'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx942", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x16'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('head_size', 128), ('mask_class', 'none'), ('scheduler_strategy', 'iterative-maxocc'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx950", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x32'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('dtype', 'bf16'), ('head_size', 128), ('scheduler_strategy', 'iterative-maxocc'), ('seq_mode', 'thd'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx950", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x32'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('dtype', 'bf16'), ('head_size', 128), ('mask_class', 'none'), ('scheduler_strategy', 'iterative-maxocc'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx950", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x32'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('head_size', 128), ('scheduler_strategy', 'iterative-maxocc'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx950", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x32'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('head_size', 128), ('mask_class', 'none'), ('scheduler_strategy', 'iterative-maxocc'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx950", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x32'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('dtype', 'bf16'), ('head_size', 128), ('scheduler_strategy', 'iterative-maxocc'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx942", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x16'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('head_size', 128), ('scheduler_strategy', 'iterative-maxocc'), ('seq_mode', 'thd'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx950", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x32'), ('atom_g4', '16x16x16'), ('block_m', 32), ('block_n', 128), ('head_size', 128), ('scheduler_strategy', 'iterative-maxocc'), ('seq_mode', 'thd'), ('transpose_source', 'lds_plain'), ('waves', 2), ('waves_per_eu', 2))), "scratch+vgpr_spill"),
    (("gfx950", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x32'), ('atom_g4', '16x16x16'), ('block_m', 128), ('block_n', 32), ('head_size', 128), ('kv_pad', 8), ('scheduler_strategy', 'iterative-ilp'), ('transpose_source', 'lds_plain'), ('waves', 8))), "compiler crash (comgr segfault)"),
    (("gfx950", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x32'), ('atom_g4', '16x16x32'), ('block_m', 128), ('block_n', 32), ('head_size', 128), ('kv_pad', 8), ('scheduler_strategy', 'iterative-ilp'), ('transpose_source', 'lds_plain'), ('waves', 8))), "compiler crash (comgr segfault)"),
    (("gfx950", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x32'), ('atom_g4', '16x16x16'), ('block_m', 128), ('block_n', 32), ('head_size', 128), ('kv_pad', 8), ('scheduler_strategy', 'iterative-ilp'), ('seq_mode', 'thd'), ('transpose_source', 'lds_plain'), ('waves', 8))), "compiler crash (comgr segfault)"),
    (("gfx950", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x32'), ('atom_g4', '16x16x32'), ('block_m', 128), ('block_n', 32), ('head_size', 128), ('kv_pad', 8), ('scheduler_strategy', 'iterative-ilp'), ('seq_mode', 'thd'), ('transpose_source', 'lds_plain'), ('waves', 8))), "compiler crash (comgr segfault)"),
    (("gfx950", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x32'), ('atom_g4', '16x16x16'), ('block_m', 64), ('block_n', 64), ('head_size', 64), ('scheduler_strategy', 'iterative-ilp'), ('transpose_source', 'lds_plain'), ('waves', 4))), "compiler crash (comgr segfault)"),
    (("gfx950", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x32'), ('atom_g4', '16x16x32'), ('block_m', 64), ('block_n', 64), ('head_size', 64), ('scheduler_strategy', 'iterative-ilp'), ('transpose_source', 'lds_plain'), ('waves', 4))), "compiler crash (comgr segfault)"),
    (("gfx950", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x32'), ('atom_g4', '16x16x16'), ('block_m', 64), ('block_n', 128), ('head_size', 128), ('transpose_source', 'tr_read'), ('waves', 4))), "scratch+vgpr_spill"),
    (("gfx950", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x32'), ('atom_g4', '16x16x16'), ('block_m', 64), ('block_n', 128), ('dtype', 'bf16'), ('head_size', 128), ('mask_class', 'none'), ('seq_mode', 'thd'), ('stage_vec', 1), ('transpose_source', 'tr_read'), ('waves', 4))), "scratch+vgpr_spill"),
    (("gfx942", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x16'), ('atom_g4', '16x16x16'), ('block_m', 64), ('block_n', 128), ('head_size', 64), ('transpose_source', 'lds_plain'), ('waves', 4))), "scratch+vgpr_spill"),
    (("gfx942", (('agpr_alloc', (0, 0)), ('atom_g02', '16x16x16'), ('atom_g4', '16x16x16'), ('block_m', 64), ('block_n', 128), ('dtype', 'bf16'), ('head_size', 64), ('mask_class', 'none'), ('seq_mode', 'thd'), ('stage_vec', 1), ('transpose_source', 'lds_plain'), ('waves', 4))), "scratch+vgpr_spill"),
)  # fmt: skip


def _ilp_spec(arch, tile, dtype, d, seq, mask, sv, var) -> AttnBwdDqSpec:
    from kernels.common.attention_bwd import AttnBwdSpec

    main = AttnBwdSpec(
        head_size=d, dtype=dtype, seq_mode=seq, stage_vec=sv, mask_class=mask,
        dq_mode="split",
    ).resolved(arch)  # fmt: skip
    return main.dq_spec(
        scheduler_strategy="iterative-ilp", **ILP_TILES[tile], **ILP_VARIANTS[var]
    )


def test_dq_resource_refusals_are_exactly_the_crash_map():
    """The validator refuses exactly the dQ specs whose compile crashes the
    production compiler (DQ_RESOURCE_REFUSED, keyed like RESOURCE_REFUSED):
    each crash-map row is refused with its record, the table holds nothing
    else, and every other shape of the map is legal."""
    from kernels.common._attention_bwd_refusals import spec_digest
    from kernels.common.attention_bwd_dq import DQ_RESOURCE_REFUSED

    keys = set()
    for row in ILP_CRASH_ROWS:
        arch = row[0]
        spec = _ilp_spec(*row)
        with pytest.raises(ValueError, match="DQ_RESOURCE_REFUSED"):
            validate_attn_bwd_dq_spec(spec, arch)
        digest = spec_digest(spec.resolved(arch), arch)
        keys.add((arch, digest))
        assert DQ_RESOURCE_REFUSED[(arch, digest)].startswith(
            "compiler crash (comgr segfault)"
        )
    for (arch, items), rule in DQ_SPILL_ROWS:
        spec = AttnBwdDqSpec(**dict(items))
        with pytest.raises(ValueError, match="DQ_RESOURCE_REFUSED"):
            validate_attn_bwd_dq_spec(spec, arch)
        digest = spec_digest(spec.resolved(arch), arch)
        keys.add((arch, digest))
        assert DQ_RESOURCE_REFUSED[(arch, digest)].startswith(rule)
    assert keys == set(DQ_RESOURCE_REFUSED)
    crashing = set(ILP_CRASH_ROWS)
    for row in itertools.product(
        ("gfx942", "gfx950"), ILP_TILES, ("fp16", "bf16"), (32, 64, 128),
        ("batched", "thd"), ("band", "none"), (8, 1), ILP_VARIANTS,
    ):  # fmt: skip
        if row not in crashing:
            validate_attn_bwd_dq_spec(_ilp_spec(*row), row[0])


@pytest.mark.parametrize("var", list(VARIANTS))
@pytest.mark.parametrize("arch", ARCHS)
def test_zero_scratch_no_spills_and_no_key_loop_agpr_copies(arch, var):
    """The resource gate of the dQ kernel on the production flavor and comgr:
    no scratch, no spills, the arch-VGPR and AGPR budgets, LDS within the plan,
    no SGPR-margin warning, and no AGPR copy in the key loops (only MFMA
    touches the dQ accumulator; S^T / dP^T stay VGPRs). The AGPR reservation
    lever (``agpr_alloc = (128, 128)``: the backend puts every MFMA
    accumulator in AGPRs, so the VALU reads S^T / dP^T back) is held to every
    rule but the copy rule. The validator's refused rows are not built."""
    pytest.importorskip("rocke.runtime.comgr")
    from rocke.helpers.compile import compile_kernel

    from ._attention_bwd_resources import (
        GATE_FLAVOR,
        comgr_is_production,
        comgr_version,
        current_flavor,
        evaluate_resource_gate,
        find_objdump,
        pin_production_comgr,
    )

    pin_production_comgr()  # torch's bundled comgr, by path; torch not imported
    if find_objdump() is None:
        pytest.skip("llvm-objdump not found (needed for the key-loop AGPR copy rule)")
    flavor, comgr = current_flavor(), comgr_version()
    if flavor != GATE_FLAVOR or not comgr_is_production(comgr):
        pytest.skip(
            f"resource gate is valid on {GATE_FLAVOR} with the production comgr "
            f"only (flavor {flavor}, comgr {comgr})"
        )
    facts = attn_bwd_arch_facts(arch)
    failures = []
    for row in (r for r in INSTANCES if r[0] == arch and r[-1] == var):
        r = validate_attn_bwd_dq_spec(_spec(*row), arch)
        art = compile_kernel(
            _kernel(row), arch=arch, capture_ir_text=False, backend="python"
        )
        tile = {}
        if r.agpr_alloc == (0, 0):
            tile = dict(block_m=r.block_m, block_n=r.block_n, waves=r.waves)
        res = evaluate_resource_gate(
            art.hsaco,
            arch=arch,
            lds_capacity_bytes=facts.lds_capacity_bytes,
            ir_text=art.llvm_text,
            **tile,
        )
        assert res.valid, (res.flavor, res.comgr)
        if res.failures or res.warnings:
            failures.append((row, res.failures, res.warnings, res.resources.as_dict()))
        plan_bytes = dq_lds_plan(r, arch).total_bytes
        if res.resources.lds_bytes > plan_bytes:
            failures.append((row, f"LDS {res.resources.lds_bytes} > plan {plan_bytes}"))
        if tile:
            assert res.loops and all(x["reads"] == 0 for x in res.loops), res.loops
    assert not failures, failures


def test_c_fragment_holds_one_query_column_per_lane():
    from kernels.common._attention_bwd_frag import eval_layout

    for arch in ARCHS:
        s = validate_attn_bwd_dq_spec(AttnBwdDqSpec(head_size=64), arch)
        g0t = BwdDqGeometry.from_spec(s, arch).g0t
        cols = [len({c for _r, c in lane}) for lane in eval_layout(g0t.op.c_layout())]
        assert set(cols) == {1}
        assert (
            np.unique([lane[0][1] for lane in eval_layout(g0t.op.c_layout())]).size
            == 16
        )
