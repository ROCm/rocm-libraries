# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only tests of the layout-general backward spec (no device, no build).

Covers the defaulted spec fields, the per-arch default geometry, every static
legality rule of the knob catalog (value sets, couplings, LDS footprint, the
two-budget register model reproduced on the worked start-point tables), kernel
name salting and the ordered kernarg lists of the ``rocke.attn_bwd.v3`` ABI.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import itertools

import pytest

from kernels.common.attention_bwd import (
    AGPR_ALLOCS,
    ATTN_BWD_ABI,
    BLOCK_K4_VALUES,
    BWD_ARCHS,
    BWD_DTYPES,
    BWD_HEAD_SIZES,
    DECLARATION_ONLY,
    LOOP_SYNCS,
    NOT_YET_EFFECTIVE_KNOBS,
    STEP_ORDERS,
    _not_built,
    agpr_copy_budget,
    AttnBwdSpec,
    attn_bwd_arch_facts,
    attn_bwd_default_geometry,
    attn_bwd_params,
    build_attn_bwd_main,
    lds_footprint_bytes,
    lds_plan,
    lds_request,
    register_estimates,
    validate_attn_bwd_spec,
)


def _structural(spec, arch):
    """Every legality rule but the recorded resource-gate failures of the
    compiled kernel (``RESOURCE_REFUSED``, pinned in
    ``test_attention_bwd_refusals.py``): the rules these tests are about."""
    return validate_attn_bwd_spec(spec, arch, main_kernel_refusals=False)


def _build(spec, arch):
    """The emitted main kernel of ``spec``, a recorded resource-gate failure
    included (the emission is what these tests check)."""
    from kernels.common.attention_bwd import _emit_main

    return _emit_main(spec, arch, main_kernel_refusals=False)


# ---------------------------------------------------------------------------
# fields and defaults
# ---------------------------------------------------------------------------


def test_only_head_size_is_required():
    required = [
        f.name
        for f in dataclasses.fields(AttnBwdSpec)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    assert required == ["head_size"]


def test_every_catalog_knob_is_a_field():
    names = {f.name for f in dataclasses.fields(AttnBwdSpec)}
    knobs = {
        "dtype",
        "seq_mode",
        "dkv_mode",
        "stage_vec",
        "mask_class",
        "dq_mode",
        "waves",
        "block_m",
        "block_n",
        "block_k4",
        "warp_grid_g4",
        "atom_g02",
        "atom_g13",
        "atom_g4",
        "pt_route",
        "kv_residency",
        "kt_source",
        "transpose_source",
        "ring_depth",
        "global_path",
        "lds_swizzle",
        "sched",
        "setprio",
        "waves_per_eu",
        "edge_tiles",
        "head_pack",
        "acc_in_lds",
        "ws_layout",
        "agpr_alloc",
        "s_in_agpr",
        "asm_tail_nop",
        "asm_cluster",
        "asm_load_hoist",
        "scheduler_strategy",
        "warp_grid_g02",
        "warp_grid_g13",
        "stage_order",
        "loop_sync",
        "step_order",
        "qdo_stage",
        "stats_load",
        "barrier_merge",
        "ring_issue",
        "g4_slicing",
    }
    assert knobs <= names
    # runtime plan values are never compile-time fields
    assert not names & {
        "g_split",
        "lb_order",
        "pair",
        "xcd_n",
        "xcd_chunk",
        "pack_heads",
        "use_worklist",
    }


def test_xcd_remap_is_a_salted_bool_field():
    plain = AttnBwdSpec(head_size=64)
    remap = AttnBwdSpec(head_size=64, xcd_remap=True)
    assert plain.xcd_remap is False
    assert remap.kernel_name("main") != plain.kernel_name("main")
    assert ("xcd_remap", True) in remap.non_default_fields()
    with pytest.raises(ValueError, match="xcd_remap"):
        AttnBwdSpec(head_size=64, xcd_remap=1)


def test_worklist_is_a_salted_thd_bool_field():
    plain = AttnBwdSpec(head_size=64, seq_mode="thd")
    wl = AttnBwdSpec(head_size=64, seq_mode="thd", worklist=True)
    assert plain.worklist is False
    assert wl.kernel_name("main") != plain.kernel_name("main")
    assert ("worklist", True) in wl.non_default_fields()
    with pytest.raises(ValueError, match="worklist"):
        AttnBwdSpec(head_size=64, seq_mode="thd", worklist=1)
    with pytest.raises(ValueError, match="seq_mode='thd'"):
        AttnBwdSpec(head_size=64, worklist=True)
    # the prep spec carries the role, the convert spec does not
    assert wl.aux_spec("prep").worklist is True
    assert wl.aux_spec("convert").worklist is False
    assert plain.aux_spec("prep").worklist is False
    assert wl.aux_spec("prep").kernel_name("prep").endswith("_thd_wl")


def test_abi_tag():
    assert ATTN_BWD_ABI == "rocke.attn_bwd.v3"


@pytest.mark.parametrize("arch", BWD_ARCHS)
@pytest.mark.parametrize("d", BWD_HEAD_SIZES)
@pytest.mark.parametrize("stage_vec", (8, 1))
@pytest.mark.parametrize("seq_mode", ("batched", "thd"))
def test_default_geometry_is_legal(arch, d, stage_vec, seq_mode):
    for dt in BWD_DTYPES:
        for dkv in ("direct", "atomic"):
            for mask in ("band", "none"):
                spec = AttnBwdSpec(
                    head_size=d,
                    dtype=dt,
                    stage_vec=stage_vec,
                    seq_mode=seq_mode,
                    dkv_mode=dkv,
                    mask_class=mask,
                )
                r = validate_attn_bwd_spec(spec, arch)
                for f in dataclasses.fields(r):
                    if f.name in ("lds_swizzle", "scheduler_strategy", "waves_per_eu"):
                        continue
                    if f.name == "agpr_route":
                        assert r.agpr_route is None  # the attr route (no split)
                        continue
                    if f.name == "asm_tail_nop":
                        assert r.asm_tail_nop is None  # no asm route
                        continue
                    if f.name == "agpr_alloc":
                        # VGPR-form MFMA on CDNA (every start point fits it)
                        mfma = attn_bwd_arch_facts(arch).matrix_path == "mfma"
                        assert r.agpr_alloc == ((0, 0) if mfma else None)
                        continue
                    assert getattr(r, f.name) is not None, f.name


def test_default_geometry_start_points():
    g = attn_bwd_default_geometry("gfx942", 128)
    assert (g["block_m"], g["block_n"], g["block_k4"], g["waves"]) == (16, 128, 32, 8)
    assert g["warp_grid_g4"] == (1, 8)
    g = attn_bwd_default_geometry("gfx942", 64)
    assert (g["block_m"], g["block_n"], g["block_k4"], g["waves"]) == (32, 128, 32, 8)
    assert g["warp_grid_g4"] == (2, 4)
    assert g["transpose_source"] == "lds_plain" and g["global_path"] == "vgpr"
    assert g["sched"] == "none"
    g = attn_bwd_default_geometry("gfx950", 128)
    assert (g["block_m"], g["block_n"], g["waves"]) == (16, 128, 8)
    assert g["kt_source"] == "lds" and g["transpose_source"] == "tr_read"
    assert g["atom_g02"] == "16x16x32" and g["atom_g13"] == "16x16x16"
    g = attn_bwd_default_geometry("gfx950", 64)
    assert (g["block_m"], g["block_n"], g["waves"]) == (32, 256, 8)
    g = attn_bwd_default_geometry("gfx1151", 64)
    assert (g["block_m"], g["block_n"], g["waves"]) == (16, 16, 1)
    assert g["acc_in_lds"] is True and g["transpose_source"] == "wmma_lds"
    assert g["scheduler_strategy"] is None
    for arch in ("gfx1151", "gfx1201"):
        g = attn_bwd_default_geometry(arch, 128)
        assert g["scheduler_strategy"] == "iterative-minreg"
    for arch in ("gfx942", "gfx950"):
        g = attn_bwd_default_geometry(arch, 64, stage_vec=1)
        assert (g["block_m"], g["block_n"]) == (16, 64)
        assert g["transpose_source"] == "lds_plain"
        assert g["scheduler_strategy"] is None
        # the narrow d128 tile uses the register-minimising scheduler
        g = attn_bwd_default_geometry(arch, 128, stage_vec=1)
        assert g["scheduler_strategy"] == "iterative-minreg"
        assert attn_bwd_default_geometry(arch, 128)["scheduler_strategy"] is None


@pytest.mark.parametrize("arch", BWD_ARCHS)
def test_one_wave_d128_tiles_resolve_to_the_register_minimising_scheduler(arch):
    one_wave = dict(waves=1, block_m=16, block_n=16, block_k4=16, warp_grid_g4=(1, 1))
    if attn_bwd_arch_facts(arch).matrix_path == "mfma":
        one_wave.update(
            atom_g02="16x16x16", atom_g13="16x16x16", atom_g4="16x16x16",
            kv_residency="lds", kt_source="lds", transpose_source="lds_plain",
        )  # fmt: skip
    for d in BWD_HEAD_SIZES:
        r = AttnBwdSpec(head_size=d, **one_wave).resolved(arch)
        assert r.scheduler_strategy == ("iterative-minreg" if d == 128 else None)
        # an explicit strategy is kept
        r = AttnBwdSpec(head_size=d, scheduler_strategy="max-ilp", **one_wave)
        assert r.resolved(arch).scheduler_strategy == "max-ilp"
    # the multi-wave CDNA start points keep the default scheduler
    if arch in ("gfx942", "gfx950"):
        assert AttnBwdSpec(head_size=128).resolved(arch).scheduler_strategy is None


def test_resolved_agpr_alloc_prefers_the_vgpr_form():
    # An explicit reservation puts the S / dP / dQ MFMAs in AGPR form too (the
    # hot-loop AGPR copy rule fails): VGPR-form whenever the estimate fits it,
    # else the backend's choice.
    assert AttnBwdSpec(head_size=128).resolved("gfx942").agpr_alloc == (0, 0)
    assert AttnBwdSpec(head_size=128).resolved("gfx950").agpr_alloc == (0, 0)
    assert AttnBwdSpec(head_size=128).resolved("gfx1151").agpr_alloc is None
    big = AttnBwdSpec(
        head_size=128, block_m=16, block_n=192, waves=4, warp_grid_g4=(1, 4)
    ).resolved("gfx950")
    assert big.agpr_alloc is None
    explicit = AttnBwdSpec(head_size=128, agpr_alloc=(192, 192)).resolved("gfx950")
    assert explicit.agpr_alloc == (192, 192)


def test_arch_facts():
    f942, f950 = attn_bwd_arch_facts("gfx942"), attn_bwd_arch_facts("gfx950")
    assert f942.wide_k_atom is False and f950.wide_k_atom is True
    assert f942.lds_capacity_bytes == 65536 and f950.lds_capacity_bytes == 163840
    assert (f942.arch_vgprs, f942.agprs) == (256, 256)
    assert attn_bwd_arch_facts("gfx1151").matrix_path == "wmma"
    assert "dma" not in f942.legal_global_paths and "dma" in f950.legal_global_paths
    with pytest.raises(ValueError):
        attn_bwd_arch_facts("gfx90a")


# ---------------------------------------------------------------------------
# register model: the worked start-point tables
# ---------------------------------------------------------------------------


def _cdna(arch, d, km, kn, kk, *, kt, g4=(1, 4), s_in_agpr=False, agpr=None, **kw):
    # The design's start points pair these paths with a stage table; the stage
    # tables are built on the synchronous body without the XT images (fenced
    # read-ahead) and without the ring (refused), and no register term depends
    # on the schedule, so the rows use sched = "none".
    if arch == "gfx942":
        base = dict(
            transpose_source="xt_lds",
            sched="none",
            global_path="vgpr",
            ring_depth=1,
            atom_g02="16x16x16",
            atom_g13="16x16x16",
            atom_g4="16x16x16",
        )
    else:
        base = dict(
            transpose_source="tr_read",
            sched="none",
            global_path="dma",
            ring_depth=2,
            atom_g02="16x16x32",
            atom_g13="16x16x16",
            atom_g4="16x16x32",
        )
    base.update(kw)
    return AttnBwdSpec(
        head_size=d,
        waves=4,
        block_m=km,
        block_n=kn,
        block_k4=kk,
        warp_grid_g4=g4,
        kv_residency="reg",
        kt_source=kt,
        s_in_agpr=s_in_agpr,
        agpr_alloc=agpr,
        **base,
    )


# (arch, d, kM0, kN0, kK4, kt_source, s_in_agpr, G4 grid) -> (arch-VGPR, AGPR).
# The design's worked tables count the G4 dS operands as kK4 wide (kK4
# slicing); G4 is emitted one atom-K step at a time, so the term is
# 2*kM0*k4a / (2*wave). On gfx950 the default G4 atom K equals kK4 (32) and the
# rows are the tables' values; on gfx942 (atom K 16) each row is the table's
# value minus kM0*(kK4 - 16) / 64: 200 - 4, 168 - 4, 184 - 8, 152 - 24. The
# gfx942 tables also count the reference pipeline's one-deep register prefetch
# of the next tile (kM0 * D / 256 on four waves); the estimate counts it with
# the register ring only (ring_depth >= 2), not for the synchronous body, so
# each gfx942 row is a further 8, 8, 8 and 4 lower.
_WORKED = [
    ("gfx942", 128, 16, 128, 32, "reg", False, (1, 4), 188, 128),
    ("gfx942", 128, 16, 128, 32, "lds", False, (1, 4), 156, 128),
    ("gfx942", 64, 32, 128, 32, "reg", False, (1, 4), 168, 64),
    ("gfx942", 32, 32, 128, 64, "reg", False, (2, 2), 124, 32),
    ("gfx950", 128, 16, 192, 32, "reg", False, (1, 4), 252, 192),
    ("gfx950", 128, 16, 192, 32, "lds", False, (1, 4), 204, 192),
    ("gfx950", 128, 32, 128, 32, "reg", False, (1, 4), 264, 128),
    ("gfx950", 128, 32, 128, 32, "lds", False, (1, 4), 232, 128),
    ("gfx950", 128, 32, 128, 32, "reg", True, (1, 4), 232, 160),
    ("gfx950", 64, 32, 256, 32, "reg", False, (1, 4), 272, 128),
    ("gfx950", 64, 32, 256, 32, "lds", False, (1, 4), 240, 128),
    ("gfx950", 64, 32, 256, 32, "reg", True, (1, 4), 208, 192),
]


@pytest.mark.parametrize("row", _WORKED, ids=lambda r: "_".join(map(str, r[:7])))
def test_register_model_reproduces_worked_tables(row):
    """The worked tables are the split plan: dK / dV in AGPRs, S / dP in AGPRs
    too only for the ``s_in_agpr`` rows (G0 / G2 asm), the dQ partial in arch
    VGPRs. A split route expresses that plan."""
    arch, d, km, kn, kk, kt, sag, g4, vgpr, agpr = row
    route = "asm_g13" if sag else "vgpr_form"
    spec = _cdna(arch, d, km, kn, kk, kt=kt, g4=g4, s_in_agpr=sag, agpr=(256, 256))
    # asm_g13 excludes stage tables; on gfx950 (all the s_in_agpr rows) the
    # stage table adds no register term, so the estimate is the same without it
    extra = {"sched": "none", "setprio": False} if route == "asm_g13" else {}
    spec = dataclasses.replace(spec, agpr_route=route, **extra).resolved(arch)
    est = register_estimates(spec, arch)
    assert (est["arch_vgpr"], est["agpr"]) == (vgpr, agpr)
    # every worked row is inside the static prune allowance, so it is compiled
    validate_attn_bwd_spec(spec, arch)


@pytest.mark.parametrize("row", _WORKED, ids=lambda r: "_".join(map(str, r[:7])))
def test_attr_route_keeps_every_accumulator_in_agprs(row):
    """Under the attr route (a reservation without a split route) the backend
    selects the AGPR form for every MFMA: S / dP and the dQ partial move from
    arch VGPRs to AGPRs, whatever ``s_in_agpr`` says (the CK plan)."""
    arch, d, km, kn, kk, kt, sag, g4, vgpr, agpr = row
    wave, w = 64, 4
    s_dp, dq = 2 * km * kn // (wave * w), km * d // (wave * w)
    attr = _cdna(arch, d, km, kn, kk, kt=kt, g4=g4, agpr=(256, 256)).resolved(arch)
    est = register_estimates(attr, arch)
    moved = dq if sag else s_dp + dq  # the sag rows already count S / dP as AGPRs
    assert (est["arch_vgpr"], est["agpr"]) == (vgpr - moved, agpr + moved)
    decl = dataclasses.replace(attr, s_in_agpr=True)
    assert register_estimates(decl, arch) == est  # declaration only
    assert agpr_copy_budget(decl, arch) == {"reads": s_dp + dq, "writes": 0, "moves": 0}
    assert agpr_copy_budget(attr, arch) == {"reads": 0, "writes": 0, "moves": 0}


def test_agpr_copy_budget_per_plan():
    base = dict(head_size=128, waves=4, block_m=32, block_n=128, block_k4=32)
    base.update(warp_grid_g4=(1, 4), kv_residency="reg", kt_source="lds")
    s_dp, dq = 2 * 32 * 128 // 256, 32 * 128 // 256
    for arch in ("gfx942", "gfx950"):
        vf = AttnBwdSpec(agpr_alloc=(0, 0), **base).resolved(arch)
        assert agpr_copy_budget(vf, arch)["reads"] == 0
        for route, sag, reads in (
            ("vgpr_form", False, 0),
            ("agpr_fence", False, 0),
            ("asm_g13", False, 0),
            ("asm_g13", True, s_dp),
            (None, True, s_dp + dq),
            (None, False, 0),
        ):
            spec = AttnBwdSpec(
                agpr_alloc=(256, 256), agpr_route=route, s_in_agpr=sag, **base
            ).resolved(arch)
            got = agpr_copy_budget(spec, arch)
            assert got == {"reads": reads, "writes": 0, "moves": 0}, (route, sag)
    rdna = AttnBwdSpec(head_size=64).resolved("gfx1151")
    assert agpr_copy_budget(rdna, "gfx1151")["reads"] == 0


def test_agpr_route_values_and_couplings():
    ok = AttnBwdSpec(head_size=64, agpr_alloc=(128, 128), agpr_route="attr")
    assert (
        ok.agpr_route is None
        and ok.kernel_name()
        == AttnBwdSpec(head_size=64, agpr_alloc=(128, 128)).kernel_name()
    )
    for bad_alloc in (None, (0, 0), "backend"):
        with pytest.raises(ValueError, match="needs a reservation"):
            AttnBwdSpec(head_size=64, agpr_alloc=bad_alloc, agpr_route="vgpr_form")
    with pytest.raises(ValueError, match="needs a reservation"):
        AttnBwdSpec(
            head_size=64, agpr_alloc=(0, 0), acc_in_lds=True, agpr_route="agpr_fence"
        )
    for route in ("vgpr_form", "agpr_fence"):
        with pytest.raises(ValueError, match="s_in_agpr"):
            AttnBwdSpec(
                head_size=64, agpr_alloc=(128, 128), agpr_route=route, s_in_agpr=True
            )
    AttnBwdSpec(
        head_size=64, agpr_alloc=(128, 128), agpr_route="asm_g13", s_in_agpr=True
    )
    for sched in ("stage_table", "iglp0"):
        with pytest.raises(ValueError, match="asm_g13"):
            AttnBwdSpec(
                head_size=64, agpr_alloc=(128, 128), agpr_route="asm_g13", sched=sched
            )
    with pytest.raises(ValueError):
        AttnBwdSpec(head_size=64, agpr_alloc=(128, 128), agpr_route="split")
    with pytest.raises(ValueError, match="MFMA targets only|implies agpr_alloc"):
        validate_attn_bwd_spec(
            AttnBwdSpec(head_size=64, agpr_alloc=(128, 128), agpr_route="vgpr_form"),
            "gfx1151",
        )
    # built: G0 / G2 as asm too (asm_g13 with s_in_agpr; d128: the gfx950 d64
    # tile is a recorded resource-gate failure, RESOURCE_REFUSED)
    spec = AttnBwdSpec(
        head_size=128, agpr_alloc=(128, 128), agpr_route="asm_g13", s_in_agpr=True
    )
    for arch in ("gfx942", "gfx950"):
        assert build_attn_bwd_main(spec, arch=arch)


@pytest.mark.parametrize(
    "arch,tile",
    [
        ("gfx942", dict(waves=4, block_m=16, block_n=128, kt_source="lds")),
        ("gfx950", dict(waves=4, block_m=32, block_n=128, kt_source="lds")),
    ],
)
def test_asm_g13_issues_g1_g3_as_agpr_asm_clusters(arch, tile):
    """asm_g13: every G1 / G3 MFMA moves from an intrinsic into an inline-asm
    cluster with ``=a`` outputs tied to the accumulators (one cluster per K
    step, each ending in the atom's s_nop); G0 / G2 / G4 stay intrinsics."""
    import re

    from kernels.common._attention_bwd_body import BwdTileGeometry
    from kernels.common._attention_bwd_caps import AGPR_ASM_TAIL_NOP

    kw = dict(head_size=128, dtype="bf16", block_k4=32, warp_grid_g4=(1, 4))
    kw.update(kv_residency="reg", **tile)
    asm = AttnBwdSpec(agpr_alloc=(128, 128), agpr_route="asm_g13", **kw)
    attr = AttnBwdSpec(agpr_alloc=(256, 256), **kw)
    r = validate_attn_bwd_spec(asm, arch)
    g = BwdTileGeometry.from_spec(r, arch)
    ir = _ir_without_name(asm, arch)
    ir0 = _ir_without_name(attr, arch)
    call = r"call <4 x float> @llvm\.amdgcn\.mfma"
    n_int, n_int0 = len(re.findall(call, ir)), len(re.findall(call, ir0))
    n_asm = len(re.findall(r"v_mfma_f32_16x16x\d+_bf16 \$", ir))
    g13 = 2 * g.g1.mma_per_wave  # G1 + G3 per q-step body
    assert n_asm % g13 == 0 and n_asm >= g13
    assert n_int0 - n_int == n_asm  # exactly the G1 / G3 MFMAs moved
    clusters = n_asm // (g.g1.frags_m * g.g1.frags_n)
    nop = AGPR_ASM_TAIL_NOP[g.g1.atom_k]
    assert ir.count(f"s_nop {nop}") >= clusters
    assert "=a" in ir


_ASM = dict(head_size=64, agpr_alloc=(128, 128), agpr_route="asm_g13")


def test_asm_cluster_knob_values_and_couplings():
    from kernels.common.attention_bwd import ASM_CLUSTERS

    assert ASM_CLUSTERS == ("k_step", "gemm")
    base = AttnBwdSpec(**_ASM)
    assert (base.asm_tail_nop, base.asm_cluster) == (None, "k_step")
    # the defaults spelled out salt nothing; a non-default value does
    same = AttnBwdSpec(**_ASM, asm_cluster="k_step", asm_tail_nop=None)
    assert same.kernel_name() == base.kernel_name()
    names = {base.kernel_name()}
    for nop in range(16):
        names.add(AttnBwdSpec(**_ASM, asm_tail_nop=nop).kernel_name())
    names.add(AttnBwdSpec(**_ASM, asm_cluster="gemm").kernel_name())
    assert len(names) == 18
    for bad in (-1, 16, True, 2.0, "4"):
        with pytest.raises(ValueError, match="asm_tail_nop"):
            AttnBwdSpec(**_ASM, asm_tail_nop=bad)
    with pytest.raises(ValueError, match="asm_cluster"):
        AttnBwdSpec(**_ASM, asm_cluster="wave")
    # legal with the asm route only
    for route in (None, "vgpr_form", "agpr_fence"):
        kw = dict(_ASM, agpr_route=route)
        with pytest.raises(ValueError, match="asm_tail_nop is legal"):
            AttnBwdSpec(**kw, asm_tail_nop=4)
        with pytest.raises(ValueError, match="asm_cluster='gemm' is legal"):
            AttnBwdSpec(**kw, asm_cluster="gemm")
    with pytest.raises(ValueError, match="asm_tail_nop is legal"):
        AttnBwdSpec(head_size=64, asm_tail_nop=0)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_asm_knobs_resolve_to_the_code_they_emit(arch):
    """The resolved spec spells an ``asm_tail_nop`` equal to the atom constant
    as ``None`` and a ``gemm`` cluster over one K step as ``k_step``; other
    values are kept."""
    from kernels.common._attention_bwd_caps import AGPR_ASM_TAIL_NOP

    tile = dict(waves=4, block_n=128, kt_source="lds", sched="none", setprio=False)
    tile.update(block_k4=32, warp_grid_g4=(1, 4))
    one = AttnBwdSpec(**_ASM, block_m=16, **tile)
    two = AttnBwdSpec(**_ASM, block_m=32, atom_g13="16x16x16", **tile)
    ref1, ref2 = (validate_attn_bwd_spec(s, arch) for s in (one, two))
    shipped = AGPR_ASM_TAIL_NOP[16]
    r = validate_attn_bwd_spec(dataclasses.replace(one, asm_tail_nop=shipped), arch)
    assert r == ref1 and r.kernel_name() == ref1.kernel_name()
    r = validate_attn_bwd_spec(dataclasses.replace(one, asm_cluster="gemm"), arch)
    assert r == ref1
    r = validate_attn_bwd_spec(dataclasses.replace(two, asm_cluster="gemm"), arch)
    assert r.asm_cluster == "gemm" and r != ref2
    for nop in (7, 11):  # at or above the ISA floor of both targets
        r = validate_attn_bwd_spec(dataclasses.replace(two, asm_tail_nop=nop), arch)
        assert r.asm_tail_nop == nop
    if arch == "gfx950":  # 16x16x32 G1 atom: one K step at block_m = 32
        wide = dataclasses.replace(two, atom_g13="16x16x32", asm_cluster="gemm")
        assert validate_attn_bwd_spec(wide, arch).asm_cluster == "k_step"


def _asm_tile(arch, **kw):
    tile = dict(head_size=128, dtype="bf16", block_k4=32, warp_grid_g4=(1, 4))
    tile.update(kv_residency="reg", waves=4, block_m=32, block_n=128)
    tile.update(kt_source="lds", agpr_alloc=(128, 128), agpr_route="asm_g13")
    tile.update(atom_g13="16x16x16")
    tile.update(kw)
    return AttnBwdSpec(**tile)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_asm_tail_nop_changes_only_the_cluster_nop(arch):
    """A non-default ``asm_tail_nop`` emits the default kernel with each
    cluster's trailing ``s_nop`` count replaced; the G1 / G3 accumulation
    order and every other instruction are unchanged. A count below the ISA's
    MFMA result wait states (``asm_tail_nop_min``) is refused."""
    from kernels.common._attention_bwd_caps import AGPR_ASM_TAIL_NOP, asm_tail_nop_min

    shipped = AGPR_ASM_TAIL_NOP[16]
    ref = _ir_without_name(_asm_tile(arch), arch)
    n_nop = ref.count(f"s_nop {shipped}")
    assert n_nop > 0
    floor = asm_tail_nop_min(arch, "bf16", 16)
    for nop in (floor, 11):
        got = _ir_without_name(_asm_tile(arch, asm_tail_nop=nop), arch)
        assert got != ref
        assert got == ref.replace(f"s_nop {shipped}", f"s_nop {nop}")
    for nop in (0, floor - 1):
        with pytest.raises(ValueError, match="ISA MFMA result wait states"):
            validate_attn_bwd_spec(_asm_tile(arch, asm_tail_nop=nop), arch)


@pytest.mark.parametrize("nop", [None, 7, 11])
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_asm_cluster_gemm_issues_one_cluster_per_gemm(arch, nop):
    """``asm_cluster = "gemm"``: G1 and G3 are each one inline-asm cluster over
    all their K steps (here two) instead of one per K step; the MFMA count, the
    intrinsic MFMAs of G0 / G2 / G4 and the cluster's s_nop are unchanged."""
    import re

    from kernels.common._attention_bwd_body import BwdTileGeometry
    from kernels.common._attention_bwd_caps import (
        AGPR_ASM_TAIL_NOP,
        ASM_MFMA_VALU_LEAD_WAIT_STATES,
    )

    k_step = _asm_tile(arch, asm_tail_nop=nop)
    gemm = _asm_tile(arch, asm_tail_nop=nop, asm_cluster="gemm")
    g = BwdTileGeometry.from_spec(validate_attn_bwd_spec(gemm, arch), arch)
    assert g.asm_cluster == "gemm" and g.g1.k_steps == 2
    ir0, ir1 = _ir_without_name(k_step, arch), _ir_without_name(gemm, arch)
    assert ir0 != ir1
    mfma = r"v_mfma_f32_16x16x16_bf16 \$"
    # every cluster opens with the VALU-write lead s_nop, then its first MFMA
    lead = f"s_nop {ASM_MFMA_VALU_LEAD_WAIT_STATES - 1}"
    first = r"\"" + lead + r"\\0A\\09v_mfma_f32_16x16x16_bf16 \$0,"
    call = r"call <4 x float> @llvm\.amdgcn\.mfma"
    assert len(re.findall(mfma, ir1)) == len(re.findall(mfma, ir0)) > 0
    assert len(re.findall(call, ir1)) == len(re.findall(call, ir0))
    n0, n1 = len(re.findall(first, ir0)), len(re.findall(first, ir1))
    assert n1 > 0 and n0 == g.g1.k_steps * n1
    # one trailing s_nop per cluster (frags > 1: none between the K steps)
    assert g.g1.frags_m * g.g1.frags_n > 1
    if nop != 0:
        want = AGPR_ASM_TAIL_NOP[16] if nop is None else nop
        assert ir1.count(f"s_nop {want}") == n1
        assert ir0.count(f"s_nop {want}") == n0


def _issue_tokens(ir: str) -> str:
    """Program-order tokens of the IR: ``C`` an asm MFMA cluster, ``L`` an LDS
    read (an addrspace(3) load or a 64-bit transpose read)."""
    out = []
    for line in ir.splitlines():
        if " call " in line and "asm sideeffect" in line and "v_mfma_f32" in line:
            out.append("C")
        elif (" load " in line and "addrspace(3)" in line) or (
            " call " in line and "ds.read.tr16.b64(" in line
        ):
            out.append("L")
    return "".join(out)


def test_asm_load_hoist_knob_values_and_couplings():
    base = AttnBwdSpec(**_ASM)
    assert base.asm_load_hoist is False
    on = AttnBwdSpec(**_ASM, asm_load_hoist=True)
    assert on.kernel_name() != base.kernel_name()
    assert AttnBwdSpec(**_ASM, asm_load_hoist=False).kernel_name() == (
        base.kernel_name()
    )
    for bad in (1, "yes", None):
        with pytest.raises(ValueError, match="asm_load_hoist must be a bool"):
            AttnBwdSpec(**_ASM, asm_load_hoist=bad)
    for route in (None, "vgpr_form", "agpr_fence"):
        with pytest.raises(ValueError, match="asm_load_hoist is legal"):
            AttnBwdSpec(**dict(_ASM, agpr_route=route), asm_load_hoist=True)
    with pytest.raises(ValueError, match="asm_load_hoist is legal"):
        AttnBwdSpec(head_size=64, asm_load_hoist=True)


@pytest.mark.parametrize("order", ["g1g3g4", "g3g1g4", "g1g4g3", "g4g3g1"])
@pytest.mark.parametrize("cluster", ["k_step", "gemm"])
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_asm_load_hoist_moves_the_second_gemm_reads(arch, cluster, order):
    """``asm_load_hoist``: every LDS read of the G1 / G3 GEMM issued second
    moves above the first one's first asm cluster; the clusters, the reads and
    every intrinsic MFMA are the same (count and text), so the kernel is the
    unhoisted one with those reads issued earlier."""
    import re

    from kernels.common._attention_bwd_body import BwdTileGeometry

    ref_spec = _asm_tile(arch, asm_cluster=cluster, stage_order=order)
    hoist_spec = dataclasses.replace(ref_spec, asm_load_hoist=True)
    g = BwdTileGeometry.from_spec(validate_attn_bwd_spec(hoist_spec, arch), arch)
    assert g.asm_load_hoist and g.g1.k_steps == 2
    ref, hoist = _ir_without_name(ref_spec, arch), _ir_without_name(hoist_spec, arch)
    assert ref != hoist
    t0, t1 = _issue_tokens(ref), _issue_tokens(hoist)
    assert sorted(t0) == sorted(t1)  # the same clusters and LDS reads
    call = r"call <4 x float> @llvm\.amdgcn\.mfma"
    assert len(re.findall(call, ref)) == len(re.findall(call, hoist))
    asm = re.compile(r'asm sideeffect "([^"]*v_mfma[^"]*)"')
    assert sorted(asm.findall(ref)) == sorted(asm.findall(hoist))
    # every cluster opens with the lead wait (s_nop 1, a newline, a tab)
    assert all(t.startswith("s_nop 1\\0A\\09") for t in asm.findall(hoist))
    # B reads of one G1 / G3 GEMM per q step (A is the register relabel)
    per_frag = g.g1.op.b_frag_len // 4 if g.reader == "tr16" else g.g1.op.b_frag_len
    second = g.g1.frags_n * g.g1.k_steps * per_frag
    n_cl = 2 * (g.g1.k_steps if cluster == "k_step" else 1)  # G1 + G3 clusters

    def bodies(t):
        """Per q-step body (``n_cl`` clusters): the reads right before its
        first cluster and the reads between its first and last cluster."""
        out, pos = [], 0
        while True:
            first = t.find("C", pos)
            if first < 0:
                return out
            k = first
            while k > 0 and t[k - 1] == "L":
                k -= 1
            count, end = 0, first
            while count < n_cl:
                end = t.index("C", end)
                count += 1
                end += 1
            inside = t[first:end].count("L")
            out.append((first - k, inside))
            pos = end

    b0, b1 = bodies(t0), bodies(t1)
    assert len(b0) == len(b1) > 0
    g4_between = order.index("g4") == 2  # G4 (and its reads) between G1 and G3
    for (pre0, in0), (pre1, in1) in zip(b0, b1):
        # the run before the first cluster grows by exactly the second
        # GEMM's reads, and none of them is left inside the clusters' span
        assert pre1 == pre0 + second
        assert in1 == in0 - second
        if not g4_between:
            first_rest = (g.g1.k_steps - 1) * g.g1.frags_n * per_frag
            assert in1 == (first_rest if cluster == "k_step" else 0)


def _wide_g13(d, dtype, **kw):
    """gfx950 start point with the 16x16x32 G1 / G3 atom (kM0 = 32 at d128)."""
    tile = {} if d != 128 else dict(block_m=32)
    return AttnBwdSpec(head_size=d, dtype=dtype, atom_g13="16x16x32", **tile, **kw)


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("d", [32, 64, 128])
def test_wide_g13_atom_on_the_transpose_read(d, dtype):
    """gfx950 ``atom_g13 = 16x16x32`` on the transpose read with the register
    relabel: legal and built; G1 / G3 issue half the MFMAs of the 16x16x16
    twin (all 16x16x32), the transpose-read count is the twin's, no b128 read,
    no alloca; the default atom is unchanged."""
    import re

    from kernels.common._attention_bwd_body import BwdTileGeometry

    wide = validate_attn_bwd_spec(_wide_g13(d, dtype), "gfx950")
    twin = dataclasses.replace(wide, atom_g13="16x16x16")
    assert (wide.transpose_source, wide.pt_route) == ("tr_read", "relabel")
    assert not _not_built(wide) and not _not_built(twin)
    assert attn_bwd_default_geometry("gfx950", d, dtype)["atom_g13"] == "16x16x16"
    g = BwdTileGeometry.from_spec(wide, "gfx950")
    assert g.b13_coords.kperm and g.relabel_plan.needs_k_perm
    ir_w, ir_t = _ir_without_name(wide, "gfx950"), _ir_without_name(twin, "gfx950")
    x16 = r"call <4 x float> @llvm\.amdgcn\.mfma\.f32\.16x16x16"
    x32 = r"call <4 x float> @llvm\.amdgcn\.mfma\.f32\.16x16x32"
    n16 = [len(re.findall(x16, s)) for s in (ir_t, ir_w)]
    n32 = [len(re.findall(x32, s)) for s in (ir_t, ir_w)]
    assert n16[1] == 0 or wide.atom_g02 == "16x16x16" or wide.atom_g4 == "16x16x16"
    moved = n16[0] - n16[1]  # the twin's G1 / G3 MFMAs
    assert moved > 0 and n32[1] - n32[0] == moved // 2
    tr = "call <4 x i16> @llvm.amdgcn.ds.read.tr16.b64("
    assert ir_w.count(tr) == ir_t.count(tr) > 0
    assert "tr16.b128" not in ir_w and "alloca" not in ir_w


def test_wide_g13_geometry_needs_the_permuted_b(monkeypatch):
    """The geometry check pairs the K-permuted relabel with the K-permuted
    G1 / G3 B: catalog B coordinates are refused before any IR."""
    from kernels.common import _attention_bwd_body as body

    wide = validate_attn_bwd_spec(_wide_g13(64, "bf16"), "gfx950")
    body.BwdTileGeometry.from_spec(wide, "gfx950")
    from kernels.common._attention_bwd_frag import operand_coords

    monkeypatch.setattr(
        body, "kperm_b_operand_coords", lambda plan, g0, g1: operand_coords(g1.op, "b")
    )
    with pytest.raises(ValueError, match="same permutation"):
        body.BwdTileGeometry.from_spec(wide, "gfx950")
    with pytest.raises(ValueError, match="block_m >= 32"):
        AttnBwdSpec(head_size=64, atom_g13="16x16x32", block_m=16)


@pytest.mark.parametrize("hoist", [False, True])
def test_wide_g13_atom_on_the_asm_route(hoist):
    """The 16x16x32 G1 / G3 atom with the K-permuted transpose read under
    ``asm_g13`` (one K step: the gemm cluster resolves to k_step), with and
    without the operand hoist: every G1 / G3 MFMA is a 16x16x32 asm MFMA."""
    import re

    spec = _wide_g13(64, "bf16", agpr_alloc=(128, 128), agpr_route="asm_g13")
    spec = dataclasses.replace(spec, asm_load_hoist=hoist, sched="none")
    r = validate_attn_bwd_spec(spec, "gfx950")
    assert not _not_built(r)
    ir = _ir_without_name(spec, "gfx950")
    assert len(re.findall(r"v_mfma_f32_16x16x32_bf16 \$", ir)) > 0
    assert "v_mfma_f32_16x16x16_bf16 $" not in ir
    assert "alloca" not in ir


def test_mfma_asm_mnemonic_table():
    from kernels.common._attention_bwd_caps import mfma_asm_mnemonic

    assert mfma_asm_mnemonic("gfx942", "bf16", 16) == "v_mfma_f32_16x16x16_bf16"
    assert mfma_asm_mnemonic("gfx942", "fp16", 16) == "v_mfma_f32_16x16x16_f16"
    assert mfma_asm_mnemonic("gfx950", "bf16", 32) == "v_mfma_f32_16x16x32_bf16"
    assert mfma_asm_mnemonic("gfx950", "fp16", 32) == "v_mfma_f32_16x16x32_f16"
    with pytest.raises(ValueError, match="gfx950-only"):
        mfma_asm_mnemonic("gfx942", "bf16", 32)
    with pytest.raises(ValueError, match="no AGPRs"):
        mfma_asm_mnemonic("gfx1151", "fp16", 16)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_split_routes_set_the_vgpr_form_option_and_keep_the_reservation(arch):
    """vgpr_form / agpr_fence: the reservation attribute stays, and the comgr
    option that selects the VGPR form for every MFMA is added; the attr route
    gets the reservation only."""
    from rocke.helpers.compile import _comgr_options_for_kernel

    for route, vf in ((None, False), ("vgpr_form", True), ("agpr_fence", True)):
        spec = AttnBwdSpec(head_size=64, agpr_alloc=(128, 128), agpr_route=route)
        k = _build(spec, arch)
        assert k.attrs["agpr_alloc"] == (128, 128)
        assert bool(k.attrs.get("mfma_vgpr_form")) is vf
        assert ("-amdgpu-mfma-vgpr-form" in _comgr_options_for_kernel(k)) is vf


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_agpr_fence_pins_the_carried_accumulators(arch):
    """agpr_fence: every q-step body passes each dK / dV fragment through an
    ``=a`` asm once (one per carried fragment per step variant), and the
    epilogue takes each back through ``=v`` once; vgpr_form adds no asm and
    lowers to the attr route's IR apart from the name (only the comgr option
    differs)."""
    spec = AttnBwdSpec(head_size=64, agpr_alloc=(128, 128), agpr_route="agpr_fence")
    r = _structural(spec, arch)
    g1 = r.waves  # noqa: F841 - documents the tile
    ir = _ir_without_name(spec, arch)
    attr = _ir_without_name(AttnBwdSpec(head_size=64, agpr_alloc=(128, 128)), arch)
    n_a = ir.count('asm "", "=a,0"')
    n_v = ir.count('asm "", "=v,0"') - attr.count('asm "", "=v,0"')
    from kernels.common._attention_bwd_body import BwdTileGeometry

    geom = BwdTileGeometry.from_spec(r, arch)
    per_step = 2 * geom.g1.frags_m * geom.g1.frags_n
    assert n_v == per_step  # once, after the loop
    assert n_a >= per_step and n_a % per_step == 0  # once per step variant
    vf = AttnBwdSpec(head_size=64, agpr_alloc=(128, 128), agpr_route="vgpr_form")
    assert _ir_without_name(vf, arch) == attr


def test_vgpr_form_fits_only_where_the_tables_say():
    # gfx942 d64 fits VGPR-form (232), d32 too (156); d128 does not (316)
    # (the tables' 248 / 184 / 328 less the step-emission dS term and the
    # register prefetch the synchronous body does not have, see _WORKED)
    for d, km, kk, g4, total in ((64, 32, 32, (1, 4), 232), (32, 32, 64, (2, 2), 156)):
        s = _cdna("gfx942", d, km, 128, kk, kt="reg", g4=g4, agpr=(0, 0))
        r = validate_attn_bwd_spec(s, "gfx942")
        assert register_estimates(r, "gfx942") == {"arch_vgpr": total, "agpr": 0}
    with pytest.raises(ValueError, match="316"):
        validate_attn_bwd_spec(
            _cdna("gfx942", 128, 16, 128, 32, kt="reg", agpr=(0, 0)), "gfx942"
        )


def test_agpr_alloc_must_hold_the_estimate():
    # the CK plan of gfx950 d64 32 x 256 W4 keeps dK / dV, S / dP and dQ in
    # AGPRs: 128 + 64 + 8 = 200, so (192, 192) is too small and (256, 256) fits
    for alloc in ((128, 128), (192, 192)):
        s = _cdna("gfx950", 64, 32, 256, 32, kt="reg", s_in_agpr=True, agpr=alloc)
        with pytest.raises(ValueError, match="below the AGPR estimate 200"):
            validate_attn_bwd_spec(s, "gfx950")
    s = _cdna("gfx950", 64, 32, 256, 32, kt="reg", s_in_agpr=True, agpr=(256, 256))
    validate_attn_bwd_spec(s, "gfx950")
    # the split plan of the same tile needs only dK / dV (128) in AGPRs, with
    # K^T from LDS so S / dP / dQ fit the arch file (240)
    s = _cdna("gfx950", 64, 32, 256, 32, kt="lds", agpr=(128, 128))
    validate_attn_bwd_spec(dataclasses.replace(s, agpr_route="vgpr_form"), "gfx950")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("d", [64, 128])
def test_agpr_reservation_must_leave_arch_vgprs(arch, d):
    """An AGPR reservation that fills the per-wave share of the unified file
    is refused: (256, 256) on the eight-wave default tile (two waves per SIMD,
    256 registers per wave) leaves no arch VGPR and fails codegen; the same
    reservation on four waves, and (192, 192) on eight, are legal."""
    eight = AttnBwdSpec(head_size=d, agpr_alloc=(256, 256))
    assert _structural(AttnBwdSpec(head_size=d), arch).waves == 8
    with pytest.raises(ValueError, match="leaves 0 of the 256 registers per wave"):
        _structural(eight, arch)
    _structural(dataclasses.replace(eight, agpr_alloc=(192, 192)), arch)
    four = AttnBwdSpec(
        head_size=d, agpr_alloc=(256, 256), waves=4, block_m=16, block_n=64,
        block_k4=32, warp_grid_g4=(1, 4),
    )  # fmt: skip
    _structural(four, arch)
    # waves_per_eu = 2 halves the share on four waves as well
    with pytest.raises(ValueError, match="registers per wave"):
        _structural(dataclasses.replace(four, waves_per_eu=2), arch)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_accepted_agpr_reservations_compile(arch):
    """Every reservation the validator's structural rules accept on the
    default d64 tile and a four-wave tile compiles (the rule matches the
    backend; a recorded resource-gate failure still compiles)."""
    pytest.importorskip("rocke.runtime.comgr")
    from ._attention_bwd_resources import compile_or_skip

    tiles = (
        {},
        dict(waves=4, block_m=16, block_n=64, block_k4=32, warp_grid_g4=(1, 4)),
    )
    built = 0
    for tile in tiles:
        for alloc in [a for a in AGPR_ALLOCS if isinstance(a, tuple) and a[0]]:
            spec = AttnBwdSpec(head_size=64, agpr_alloc=alloc, **tile)
            try:
                _structural(spec, arch)
            except ValueError:
                continue
            compile_or_skip(
                _build(spec, arch), arch=arch,
                capture_ir_text=False, backend="python",
            )  # fmt: skip
            built += 1
    assert built >= 5


def test_register_budget_beyond_allowance_is_pruned():
    # K, K^T, V resident at kN0 256 / d128 is far above the arch budget
    s = _cdna("gfx950", 128, 32, 256, 32, kt="reg")
    with pytest.raises(ValueError, match="arch-VGPR estimate"):
        validate_attn_bwd_spec(s, "gfx950")


def test_waves_per_eu_needs_the_halved_budget():
    with pytest.raises(ValueError, match="waves_per_eu=2"):
        validate_attn_bwd_spec(
            AttnBwdSpec(head_size=128, waves=4, warp_grid_g4=(1, 4), waves_per_eu=2),
            "gfx942",
        )
    # the gfx950 d64 start point: estimate between 512 / 3 and 512 / 2
    est = register_estimates(AttnBwdSpec(head_size=64).resolved("gfx950"), "gfx950")
    assert 512 // 3 < est["arch_vgpr"] + est["agpr"] <= 512 // 2
    validate_attn_bwd_spec(AttnBwdSpec(head_size=64, waves_per_eu=2), "gfx950")
    for n in (3, 4):
        with pytest.raises(ValueError, match=f"waves_per_eu={n} needs"):
            validate_attn_bwd_spec(AttnBwdSpec(head_size=64, waves_per_eu=n), "gfx950")


# ---------------------------------------------------------------------------
# LDS footprint
# ---------------------------------------------------------------------------


def test_lds_gfx942_d128_stage0_is_exactly_the_capacity():
    s = _cdna("gfx942", 128, 16, 128, 32, kt="reg").resolved("gfx942")
    assert lds_footprint_bytes(s, "gfx942") == 65536
    validate_attn_bwd_spec(s, "gfx942")
    # K^T kept in LDS reuses the staging slot, so it still fits
    validate_attn_bwd_spec(dataclasses.replace(s, kt_source="lds"), "gfx942")
    # K and V both kept in LDS next to the step ring do not
    with pytest.raises(ValueError, match="LDS plan"):
        validate_attn_bwd_spec(
            dataclasses.replace(s, kv_residency="lds", kt_source="lds"), "gfx942"
        )


def test_lds_gfx950_has_room_for_a_three_deep_ring():
    s = _cdna("gfx950", 128, 16, 192, 32, kt="lds", ring_depth=3).resolved("gfx950")
    assert lds_footprint_bytes(s, "gfx950") <= 163840
    validate_attn_bwd_spec(s, "gfx950")


def test_lds_counterexample_resident_kt_next_to_lds_kv_is_rejected():
    # K, V and the K^T image all resident next to a two-deep XT ring: the
    # planner places 3 x 12 KiB resident plus a 29 KiB loop phase (the ring
    # stages each slot's lse2 / Dsum rows next to its Q / dO), > 64 KiB.
    s = _cdna(
        "gfx942",
        32,
        32,
        192,
        64,
        kt="lds",
        g4=(2, 2),
        ring_depth=2,
        sched="none",
    )
    s = dataclasses.replace(s, kv_residency="lds").resolved("gfx942")
    assert (s.kv_residency, s.kt_source, s.transpose_source) == ("lds", "lds", "xt_lds")
    plan = lds_plan(s, "gfx942")
    assert plan.has("kt_res") and plan.has("k_res") and plan.has("v_res")
    assert sorted(b.slot for b in plan.buffers if b.name == "lse2") == [0, 1]
    assert not plan.fits
    assert lds_footprint_bytes(s, "gfx942") == plan.total_bytes > 65536
    # (the validator refuses the XT images under a ring before the LDS budget:
    # they are built on the synchronous body only)
    with pytest.raises(ValueError, match="synchronous body"):
        validate_attn_bwd_spec(s, "gfx942")
    # one ring slot fewer fits
    validate_attn_bwd_spec(dataclasses.replace(s, ring_depth=1), "gfx942")


def test_lds_request_carries_every_planner_input():
    s = AttnBwdSpec(head_size=64, dtype="bf16", lds_swizzle="qt=pad8,dot=xor").resolved(
        "gfx950"
    )
    req = lds_request(s, "gfx950")
    assert (req.head_size, req.k_m0, req.k_n0, req.waves) == (
        64,
        s.block_m,
        s.block_n,
        s.waves,
    )
    assert req.wave_size == 64 and req.dtype == "bf16"
    assert req.swizzle_of("qt", "xor") == "pad8"
    assert req.acc_in_lds is False and req.dq_mode == "atomic"
    r = AttnBwdSpec(head_size=64).resolved("gfx1151")
    rq = lds_request(r, "gfx1151")
    assert rq.wave_size == 32 and rq.acc_in_lds is True
    with pytest.raises(ValueError, match="not resolved"):
        lds_request(AttnBwdSpec(head_size=64), "gfx942")


def _lds_sweep_specs(arch):
    """Every LDS-relevant knob combination of ``arch`` (other knobs fixed legal)."""
    facts = attn_bwd_arch_facts(arch)
    mfma = facts.matrix_path == "mfma"
    atom = facts.legal_atoms[0]
    for d, km, kn, w, (kv, kt), tsrc, ring, gpath, acc, pt in itertools.product(
        BWD_HEAD_SIZES,
        (16, 32, 64),
        (16, 32, 64, 128, 192, 256),
        facts.legal_waves,
        (("reg", "reg"), ("reg", "lds"), ("lds", "lds")),
        facts.legal_transpose_sources,
        facts.legal_ring_depths,
        facts.legal_global_paths,
        (False, True) if mfma else (True,),
        ("relabel", "lds") if mfma else ("lds",),
    ):
        if not mfma and kv == "reg":
            continue
        if gpath == "dma" and tsrc == "xt_lds":
            continue
        if w == 1 and (km, kn) != (16, 16):
            continue
        g4 = (1, w) if d % (16 * w) == 0 else (2, max(1, w // 2))
        yield AttnBwdSpec(
            head_size=d,
            waves=w,
            block_m=km,
            block_n=kn,
            block_k4=16,
            warp_grid_g4=g4,
            atom_g02=atom,
            atom_g13=atom,
            atom_g4=atom,
            pt_route=pt,
            kv_residency=kv,
            kt_source=kt,
            transpose_source=tsrc,
            ring_depth=ring,
            global_path=gpath,
            sched="none",
            setprio=False,
            acc_in_lds=acc,
            agpr_alloc=(0, 0) if (mfma and acc) else None,
        )


@pytest.mark.parametrize("arch", BWD_ARCHS)
def test_validator_and_phase_planner_agree(arch):
    # Over the whole LDS-relevant knob space of the arch: a spec the validator
    # accepts fits the phase plan and reports its footprint; a spec rejected on
    # LDS does not fit; a spec rejected on registers (checked after LDS) fits.
    counts = {"legal": 0, "lds": 0, "reg": 0}
    for spec in _lds_sweep_specs(arch):
        r = spec.resolved(arch)
        plan = lds_plan(r, arch)
        try:
            validate_attn_bwd_spec(spec, arch)
        except ValueError as exc:
            msg = str(exc)
            if msg.startswith("LDS plan"):
                assert not plan.fits, (spec, plan.total_bytes)
                counts["lds"] += 1
            elif "estimate" in msg:
                assert plan.fits, (spec, msg)
                counts["reg"] += 1
            continue
        assert plan.fits, (spec, plan.total_bytes)
        assert lds_footprint_bytes(r, arch) == plan.total_bytes
        counts["legal"] += 1
    assert counts["legal"] > 0 and counts["lds"] > 0, counts


# ---------------------------------------------------------------------------
# legality rules (value sets and couplings)
# ---------------------------------------------------------------------------

_BAD_FIELDS = [
    dict(head_size=256),
    dict(head_size=96),
    dict(head_size=64, dtype="fp32"),
    dict(head_size=64, stage_vec=4),
    dict(head_size=64, seq_mode="paged"),
    # an unknown dq_mode ("split" is built: test_attention_bwd_split.py)
    dict(head_size=64, dq_mode="partial"),
    dict(head_size=64, block_n=48),
    dict(head_size=64, block_m=128),
    dict(head_size=64, atom_g02="32x32x8"),
    dict(head_size=64, sched="iglp2"),
    dict(head_size=64, ws_layout="row_major"),
    dict(head_size=64, agpr_alloc=(64, 64)),
    dict(head_size=64, kv_residency="lds", kt_source="reg"),
    dict(head_size=64, global_path="dma", stage_vec=1),
    dict(head_size=64, global_path="dma", transpose_source="xt_lds"),
    dict(head_size=64, setprio=True, sched="iglp0"),
    dict(head_size=64, s_in_agpr=True, agpr_alloc=(0, 0)),
    dict(head_size=64, acc_in_lds=True, agpr_alloc=(128, 128)),
    dict(head_size=64, atom_g13="16x16x32", block_m=16),
    dict(head_size=64, warp_grid_g4=(1, 2, 3)),
    dict(head_size=64, lds_swizzle="k=zigzag"),
    dict(head_size=64, lds_swizzle="k=xor,k=pad8"),
    dict(head_size=64, lds_swizzle="kxor"),
    dict(head_size=64, scheduler_strategy="not-a-strategy"),
    dict(head_size=64, name="bad name"),
    dict(head_size=64, stage_order="g2g1g3"),
    dict(head_size=64, stage_order="g1g3"),
    dict(head_size=64, stage_order="g4g1g3", sched="stage_table"),
    dict(head_size=64, stage_order="g3g1g4", sched="stage_table_atomics"),
    dict(head_size=64, loop_sync="none"),
    dict(head_size=64, loop_sync=None),
    dict(head_size=64, step_order="kv_outer"),
    dict(head_size=64, step_order=None),
    dict(head_size=64, global_path="dma", loop_sync="lds_only"),
    dict(head_size=64, qdo_stage="interleaved"),
    dict(head_size=64, qdo_stage=None),
    dict(head_size=64, stats_load="prefetch"),
]


@pytest.mark.parametrize(
    "kw",
    _BAD_FIELDS,
    ids=lambda kw: "-".join(f"{k}" for k in kw if k != "head_size") or "head",
)
def test_arch_independent_rules_raise(kw):
    with pytest.raises(ValueError):
        AttnBwdSpec(**kw)


_BAD_FOR_ARCH = [
    ("gfx942", dict(transpose_source="tr_read")),
    ("gfx942", dict(global_path="dma", transpose_source="lds_plain")),
    ("gfx942", dict(atom_g02="16x16x32")),
    ("gfx942", dict(ring_depth=3)),
    ("gfx942", dict(waves=4, block_n=32)),  # kN0 % (16 * W)
    ("gfx942", dict(waves=1, block_m=32, block_n=64, warp_grid_g4=(1, 1))),
    ("gfx942", dict(block_k4=64, block_n=32, waves=2, warp_grid_g4=(1, 2))),
    ("gfx942", dict(warp_grid_g4=(4, 1))),
    ("gfx942", dict(setprio=True)),
    ("gfx950", dict(atom_g4="16x16x32", block_k4=16)),
    ("gfx950", dict(waves_per_eu=3)),
    ("gfx1151", dict(kv_residency="reg", kt_source="lds")),
    ("gfx1151", dict(acc_in_lds=False)),
    ("gfx1151", dict(waves=8)),
    ("gfx1151", dict(pt_route="relabel")),
    ("gfx1151", dict(atom_g02="16x16x16")),
    ("gfx1151", dict(agpr_alloc=(128, 128))),
]


@pytest.mark.parametrize(
    "arch,kw", _BAD_FOR_ARCH, ids=lambda x: x if isinstance(x, str) else "-".join(x)
)
def test_arch_rules_raise(arch, kw):
    with pytest.raises(ValueError):
        validate_attn_bwd_spec(AttnBwdSpec(head_size=64, **kw), arch)


# Small tiles whose register estimates sit far below every budget, so only the
# rule under test can reject them.
_SMALL_W2 = dict(
    head_size=32,
    waves=2,
    block_m=16,
    block_n=32,
    block_k4=16,
    warp_grid_g4=(1, 2),
    atom_g02="16x16x16",
    atom_g13="16x16x16",
    atom_g4="16x16x16",
)
_SMALL_W4 = dict(
    head_size=32,
    waves=4,
    block_m=32,
    block_n=64,
    block_k4=16,
    warp_grid_g4=(2, 2),
    atom_g02="16x16x16",
    atom_g13="16x16x16",
    atom_g4="16x16x16",
)


def test_waves_per_eu_one_to_four_on_every_cdna_target():
    """MFMA targets take occupancy hints 1 to 4 on two and four waves (the
    register budget, 512 / n, is the only rule: these tiles are far below it);
    WMMA targets take None, 1 or 2; 5 is no value."""
    for arch in ("gfx942", "gfx950"):
        for kw in (_SMALL_W2, _SMALL_W4):
            for n in (1, 2, 3, 4):
                r = validate_attn_bwd_spec(AttnBwdSpec(**kw, waves_per_eu=n), arch)
                assert r.waves_per_eu == n
    with pytest.raises(ValueError, match="waves_per_eu=5 not in"):
        AttnBwdSpec(**_SMALL_W2, waves_per_eu=5)
    for n in (3, 4):
        with pytest.raises(ValueError, match="WMMA targets"):
            validate_attn_bwd_spec(AttnBwdSpec(head_size=64, waves_per_eu=n), "gfx1151")


def test_g4_warp_grid_must_divide_d_and_block_m():
    with pytest.raises(ValueError, match="does not divide"):
        validate_attn_bwd_spec(AttnBwdSpec(head_size=32, warp_grid_g4=(1, 4)), "gfx942")
    with pytest.raises(ValueError, match="does not divide"):
        validate_attn_bwd_spec(
            AttnBwdSpec(head_size=128, block_m=16, waves=4, warp_grid_g4=(2, 2)),
            "gfx942",
        )


def test_g02_g13_warp_grids_resolve_and_need_the_lds_route_to_change():
    r = validate_attn_bwd_spec(AttnBwdSpec(**_SMALL_W4), "gfx942")
    assert (r.warp_grid_g02, r.warp_grid_g13) == ((1, 4), (4, 1))
    assert validate_attn_bwd_spec(
        AttnBwdSpec(head_size=64), "gfx1151"
    ).warp_grid_g02 == (1, 1)
    for gname in ("warp_grid_g02", "warp_grid_g13"):
        with pytest.raises(ValueError, match="needs pt_route='lds'"):
            validate_attn_bwd_spec(
                AttnBwdSpec(**_SMALL_W4, **{gname: (2, 2)}), "gfx942"
            )
        lds = validate_attn_bwd_spec(
            AttnBwdSpec(**_SMALL_W4, pt_route="lds", **{gname: (2, 2)}), "gfx942"
        )
        assert getattr(lds, gname) == (2, 2)
        with pytest.raises(ValueError, match="must be"):
            validate_attn_bwd_spec(
                AttnBwdSpec(**_SMALL_W4, pt_route="lds", **{gname: (4, 4)}), "gfx942"
            )
    # (2, W/2) must still divide the tile: G0/G2 rows are block_m
    with pytest.raises(ValueError, match="does not divide"):
        validate_attn_bwd_spec(
            AttnBwdSpec(head_size=64, block_m=16, pt_route="lds", warp_grid_g02=(2, 2)),
            "gfx942",
        )
    with pytest.raises(ValueError):
        AttnBwdSpec(head_size=64, warp_grid_g13=(0, 4))


def test_dma_buffers_read_by_tr_read_force_the_xor_swizzle():
    kw = dict(
        head_size=128, global_path="dma", transpose_source="tr_read", kt_source="lds"
    )
    for bad in ("k=pad8,kt=none,q=none", "q=pad16", "do=none"):
        with pytest.raises(ValueError, match="need 'xor'"):
            validate_attn_bwd_spec(AttnBwdSpec(**kw, lds_swizzle=bad), "gfx950")
    validate_attn_bwd_spec(
        AttnBwdSpec(**kw, lds_swizzle="do=xor,k=xor,q=xor,ds=pad8"), "gfx950"
    )
    validate_attn_bwd_spec(AttnBwdSpec(**kw), "gfx950")  # predictor default
    # without DMA the coupling does not apply
    validate_attn_bwd_spec(
        AttnBwdSpec(head_size=128, transpose_source="tr_read", lds_swizzle="q=pad8"),
        "gfx950",
    )


def test_debug_tile_is_legal_on_every_arch():
    for arch in BWD_ARCHS:
        kw = dict(waves=1, block_m=16, block_n=16, block_k4=16, warp_grid_g4=(1, 1))
        if attn_bwd_arch_facts(arch).matrix_path == "mfma":
            kw.update(
                atom_g02="16x16x16",
                atom_g13="16x16x16",
                atom_g4="16x16x16",
                kv_residency="lds",
                kt_source="lds",
                transpose_source="lds_plain",
                waves_per_eu=None,
            )
        for d in BWD_HEAD_SIZES:
            validate_attn_bwd_spec(AttnBwdSpec(head_size=d, **kw), arch)


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        validate_attn_bwd_spec(AttnBwdSpec(head_size=64), "gfx90a")


def test_swizzle_is_canonical_and_hashable():
    a = AttnBwdSpec(head_size=64, lds_swizzle=" q=pad8 , k=xor")
    b = AttnBwdSpec(head_size=64, lds_swizzle="k=xor,q=pad8")
    assert a.lds_swizzle == "k=xor,q=pad8" and a == b and hash(a) == hash(b)


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------


def test_kernel_name_salted_by_every_non_default_knob():
    base = AttnBwdSpec(head_size=128, dtype="bf16")
    assert base.kernel_name() == "rocke_attn_bwd_main_bf16_d128_batched_direct_band_sv8"
    variants = [
        base,
        dataclasses.replace(base, waves=2),
        dataclasses.replace(base, block_n=64),
        dataclasses.replace(base, edge_tiles=True),
        dataclasses.replace(base, ws_layout="token_major"),
        dataclasses.replace(base, agpr_alloc=(256, 256)),
        dataclasses.replace(base, s_in_agpr=True),
        dataclasses.replace(base, lds_swizzle="k=xor"),
        dataclasses.replace(base, scheduler_strategy=None, sched="iglp0"),
        dataclasses.replace(base, seq_mode="thd"),
        dataclasses.replace(base, stage_vec=1),
        dataclasses.replace(base, mask_class="none"),
        dataclasses.replace(base, dkv_mode="atomic"),
        dataclasses.replace(base, warp_grid_g02=(2, 2)),
        dataclasses.replace(base, warp_grid_g13=(2, 2)),
        dataclasses.replace(base, stage_order="g3g1g4"),
        dataclasses.replace(base, stage_order="g4g1g3"),
        dataclasses.replace(base, loop_sync="lds_only"),
        dataclasses.replace(base, step_order="head_outer"),
        dataclasses.replace(base, qdo_stage="loads_first"),
        dataclasses.replace(base, qdo_stage="joint"),
        dataclasses.replace(base, stats_load="step_top"),
        dataclasses.replace(base, qdo_stage="joint", stats_load="step_top"),
    ]
    names = [v.kernel_name() for v in variants]
    assert len(set(names)) == len(names)
    assert base.kernel_name() == AttnBwdSpec(head_size=128, dtype="bf16").kernel_name()


def test_stage_order_values_and_couplings():
    from kernels.common._attention_bwd_body import (
        DEFAULT_STAGE_ORDER,
        STAGE_ORDERS,
        stage_units,
    )

    assert DEFAULT_STAGE_ORDER == "g1g3g4" == AttnBwdSpec(head_size=64).stage_order
    assert len(set(STAGE_ORDERS)) == 6
    for order in STAGE_ORDERS:
        assert sorted(stage_units(order)) == ["g1", "g3", "g4"]
    # the explicit default is the default kernel (same name, no salt)
    explicit = AttnBwdSpec(head_size=64, stage_order="g1g3g4")
    assert explicit == AttnBwdSpec(head_size=64)
    assert explicit.non_default_fields() == ()
    for arch in BWD_ARCHS:
        for order in STAGE_ORDERS:
            # legal with every sched that has no count model of the order
            for sched in ("none", "iglp0", "iglp1", "fences_only"):
                spec = AttnBwdSpec(head_size=64, stage_order=order, sched=sched)
                assert validate_attn_bwd_spec(spec, arch).stage_order == order
            for sched in ("stage_table", "stage_table_atomics"):
                kw = dict(head_size=64, sched=sched)
                validate_attn_bwd_spec(AttnBwdSpec(**kw), arch)
                if order == DEFAULT_STAGE_ORDER:
                    AttnBwdSpec(**kw, stage_order=order)
                    continue
                with pytest.raises(ValueError, match="count model"):
                    AttnBwdSpec(**kw, stage_order=order)
    # the geometry rejects the coupling too (a spec built around the check)
    from kernels.common._attention_bwd_body import BwdTileGeometry

    r = validate_attn_bwd_spec(AttnBwdSpec(head_size=64, sched="stage_table"), "gfx942")
    assert BwdTileGeometry.from_spec(r, "gfx942").stage_order == "g1g3g4"
    bad = dataclasses.replace(r, sched="none")
    object.__setattr__(bad, "stage_order", "g4g1g3")
    object.__setattr__(bad, "sched", "stage_table")
    with pytest.raises(ValueError, match="count model"):
        BwdTileGeometry.from_spec(bad, "gfx942")
    object.__setattr__(bad, "stage_order", "g4g4g1")
    with pytest.raises(ValueError, match="stage_order"):
        BwdTileGeometry.from_spec(bad, "gfx942")


def test_loop_sync_and_step_order_defaults_and_values():
    """The barrier flavour and the step order default to today's emission
    (spelled-out defaults are the same spec and name); every value is legal on
    every arch (the counted waits on the CDNA targets, refused on the others)
    and salts the name; the LDS-only drain needs the register staging path."""
    assert LOOP_SYNCS == ("full", "lds_only", "counted")
    assert STEP_ORDERS == ("qtile_outer", "head_outer")
    base = AttnBwdSpec(head_size=64)
    assert (base.loop_sync, base.step_order) == ("full", "qtile_outer")
    explicit = AttnBwdSpec(head_size=64, loop_sync="full", step_order="qtile_outer")
    assert explicit == base and explicit.kernel_name() == base.kernel_name()
    assert not explicit.non_default_fields()
    for arch in BWD_ARCHS:
        ref = validate_attn_bwd_spec(base, arch).kernel_name()
        mfma = attn_bwd_arch_facts(arch).matrix_path == "mfma"
        for ls, so in itertools.product(LOOP_SYNCS, STEP_ORDERS):
            spec = AttnBwdSpec(head_size=64, loop_sync=ls, step_order=so)
            if ls == "counted" and not mfma:
                with pytest.raises(ValueError, match="CDNA"):
                    validate_attn_bwd_spec(spec, arch)
                continue
            r = validate_attn_bwd_spec(spec, arch)
            assert (r.loop_sync, r.step_order) == (ls, so)
            assert (r.kernel_name() == ref) == ((ls, so) == ("full", "qtile_outer"))
    with pytest.raises(ValueError, match="global_path='vgpr'"):
        AttnBwdSpec(head_size=64, global_path="dma", loop_sync="lds_only")
    # the DMA path keeps its full barriers (legal with the default flavour)
    AttnBwdSpec(head_size=64, global_path="dma")


# ---------------------------------------------------------------------------
# kernarg ABI (independent copy of the ordered lists)
# ---------------------------------------------------------------------------

_V3 = {
    "main": (
        "Q K V dO dK dV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV SEQ_Q SEQ_KV OFF_Q OFF_KV WORKLIST",
        "q_b q_h k_b k_h v_b v_h do_b do_h dk_b dk_h dv_b dv_h",
        "scale_log2 ds_mult dk_mult dq_mult",
        "q_t k_t v_t do_t dk_t dv_t "
        "h_q h_k h_v G gk gv S_q_max S_kv_max has_len len_stride off64 q_mult q_div kv_mult kv_div "
        "left right bottom_right ws_rows_q ws_rows_kv ws_seg "
        "g_split n_kv_tiles lb_order pair xcd_n xcd_chunk pack_heads use_worklist",
    ),
    "prep": (
        "O dO LSE SEQ_Q SEQ_KV OFF_Q OFF_KV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV WORKLIST",
        "o_b o_h do_b do_h l_b l_h",
        "",
        "o_t do_t l_t h_q h_k h_v S_q_max S_kv_max "
        "has_len len_stride off64 q_mult q_div kv_mult kv_div ws_rows_q ws_rows_kv ws_seg "
        "zero_kv zero_dq use_worklist n_batch wl_kn0",
    ),
    "convert": (
        "SRC DST SEQ OFF",
        "d_b d_h",
        "mult",
        "H d_t S_max has_len len_stride off64 tok_mult tok_div ws_rows ws_seg",
    ),
    "dq": (
        "Q K V dO dQ WS_LSE2 WS_DSUM SEQ_Q SEQ_KV OFF_Q OFF_KV",
        "q_b q_h k_b k_h v_b v_h do_b do_h dq_b dq_h",
        "scale_log2 ds_mult dq_mult",
        "q_t k_t v_t do_t dq_t "
        "h_q h_k h_v G S_q_max S_kv_max has_len len_stride off64 q_mult q_div kv_mult kv_div "
        "left right bottom_right ws_rows_q ws_seg xcd_n xcd_chunk",
    ),
}

# The superseded v2 lists (all-i32 scalars): every v2 name keeps its meaning in
# v3; v3 moves the batch / head strides to i64 and adds the THD row flag.
_V2_NAMES = {
    "main": "q_b q_h q_t k_b k_h k_t v_b v_h v_t do_b do_h do_t dk_b dk_h dk_t dv_b dv_h "
    "dv_t h_q h_k h_v G gk gv S_q_max S_kv_max has_len len_stride off64 q_mult q_div "
    "kv_mult kv_div left right bottom_right ws_rows_q ws_rows_kv g_split n_kv_tiles "
    "lb_order pair xcd_n xcd_chunk pack_heads use_worklist",
    "prep": "o_b o_h o_t do_b do_h do_t l_b l_h l_t h_q h_k h_v S_q_max S_kv_max has_len "
    "len_stride off64 q_mult q_div kv_mult kv_div ws_rows_q ws_rows_kv zero_kv zero_dq "
    "use_worklist n_batch wl_kn0",
    "convert": "H d_b d_h d_t S_max has_len len_stride off64 tok_mult tok_div ws_rows",
}


def _split(params):
    return tuple(tuple(n for n, f in params if f == k) for k in ("Q", "q", "f", "i"))


@pytest.mark.parametrize("stage", sorted(_V3))
def test_params_match_the_v3_lists(stage):
    got = _split(attn_bwd_params(stage, AttnBwdSpec(head_size=64)))
    want = tuple(tuple(s.split()) for s in _V3[stage])
    assert got == want
    params = attn_bwd_params(stage)
    # pointers first, then i64, f32, i32 (the i64 block is 8-byte aligned)
    kinds = [f for _, f in params]
    assert kinds == sorted(kinds, key="Qqfi".index)
    assert len({n for n, _ in params}) == len(params)


@pytest.mark.parametrize("stage", sorted(_V2_NAMES))
def test_v2_scalars_survive_in_v3(stage):
    """Every v2 scalar is still a v3 kernarg; only the batch / head strides
    changed type (i64) and ``ws_seg`` is new."""
    v3 = dict(attn_bwd_params(stage))
    v2 = _V2_NAMES[stage].split()
    assert set(v2) <= set(v3)
    widened = {n for n in v2 if v3[n] == "q"}
    assert widened and all(n.endswith(("_b", "_h")) for n in widened)
    ptrs_f32 = set(_V3[stage][0].split()) | set(_V3[stage][2].split())
    assert set(v3) - set(v2) - ptrs_f32 == {"ws_seg"}


def _ir_without_name(spec, arch):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

    name = _structural(spec, arch).kernel_name("main")
    ir = lower(_build(spec, arch), arch=arch, llvm_flavor="llvm22")
    return ir.replace(name, "KNAME")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1151"])
def test_inert_knob_status_matches_the_emitted_ir(arch):
    """Pins the status of the knobs that are inert under some specs: no knob
    is not yet effective; ``block_k4`` emits the default IR (apart from the
    name) under ``g4_slicing = "step"`` and is inert there by
    ``INERT_KNOBS``, while under key slicing two legal values emit different
    IR; ``s_in_agpr`` is a declaration under the attr route and code under
    ``asm_g13``."""
    from kernels.common.attention_bwd import INERT_KNOBS

    assert set(NOT_YET_EFFECTIVE_KNOBS) == set()
    assert set(DECLARATION_ONLY) == {"s_in_agpr"}
    mfma = arch != "gfx1151"
    if mfma:
        base = AttnBwdSpec(head_size=64, agpr_alloc=(128, 128))
    else:  # the two-wave RDNA tile (the one-wave tile has one legal block_k4)
        base = AttnBwdSpec(
            head_size=64, waves=2, block_m=16, block_n=32, block_k4=32,
            warp_grid_g4=(1, 2),
        )  # fmt: skip
    ref = _ir_without_name(base, arch)
    k4 = _structural(base, arch).block_k4
    tried = 0
    for alt in BLOCK_K4_VALUES:
        if alt == k4:
            continue
        spec = dataclasses.replace(base, block_k4=alt)
        try:
            r = _structural(spec, arch)
        except ValueError:
            continue
        assert spec.kernel_name() != base.kernel_name()
        assert INERT_KNOBS["block_k4"](r, arch)
        assert _ir_without_name(spec, arch) == ref, f"block_k4={alt} is effective"
        tried += 1
    assert tried
    if not mfma:  # key slicing needs MFMA register accumulators
        with pytest.raises(ValueError, match="g4_slicing"):
            validate_attn_bwd_spec(
                dataclasses.replace(base, g4_slicing="slice_db"), arch
            )
        return
    sliced = {}
    for alt in BLOCK_K4_VALUES:
        spec = dataclasses.replace(base, g4_slicing="slice_db", block_k4=alt)
        try:
            r = _structural(spec, arch)
        except ValueError:
            continue
        assert not INERT_KNOBS["block_k4"](r, arch)
        sliced[alt] = _ir_without_name(spec, arch)
    assert len(sliced) >= 2 and len(set(sliced.values())) == len(sliced)
    assert ref not in sliced.values()
    # s_in_agpr is a declaration under the attr route (base's route) ...
    spec = dataclasses.replace(base, s_in_agpr=True)
    assert DECLARATION_ONLY["s_in_agpr"](spec)
    assert _ir_without_name(spec, arch) == ref, "s_in_agpr is effective"
    # ... and code under asm_g13 (G0 / G2 as asm clusters; d128: the gfx950
    # d64 tile with asm G0 / G2 is a recorded resource-gate failure)
    split = dataclasses.replace(base, head_size=128, agpr_route="asm_g13", sched="none")
    split_s = dataclasses.replace(split, s_in_agpr=True)
    assert not DECLARATION_ONLY["s_in_agpr"](split_s)
    assert _ir_without_name(split_s, arch) != _ir_without_name(split, arch)


def test_params_reject_unknown_stage():
    with pytest.raises(ValueError):
        attn_bwd_params("epilogue")


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def test_builder_signature_and_unbuilt_knobs(monkeypatch):
    import kernels.common.attention_bwd as ab

    sig = inspect.signature(build_attn_bwd_main)
    assert list(sig.parameters) == ["spec", "arch"]
    assert sig.parameters["arch"].kind is inspect.Parameter.KEYWORD_ONLY
    # a finding of the emission check (_not_built) raises before any IR
    wide = AttnBwdSpec(head_size=64, transpose_source="tr_read", atom_g13="16x16x32")
    with monkeypatch.context() as mp:
        mp.setattr(ab, "_not_built", lambda spec: ["an unbuilt emission"])
        with pytest.raises(NotImplementedError, match="not built for"):
            build_attn_bwd_main(wide, arch="gfx950")
    # masks, edge tiles, head packing, atomic dK / dV, THD sequences, the
    # token-major workspace, the XT transposes, the xor swizzle and the
    # K-permuted 16x16x32 G1 / G3 B on the transpose read are built
    build_attn_bwd_main(wide, arch="gfx950")
    for kw in (
        dict(transpose_source="xt_lds"),
        dict(transpose_source="xt_lds", lds_swizzle="qt=xor,dot=pad8,q=xor"),
        dict(lds_swizzle="q=xor,do=xor,ds=xor,k_stage=xor,dkv_out=xor"),
        dict(mask_class="band"),
        dict(seq_mode="thd"),
        dict(seq_mode="thd", dkv_mode="atomic", head_pack=True),
        dict(seq_mode="thd", ws_layout="token_major"),
        dict(mask_class="band", edge_tiles=True),
        dict(mask_class="none", head_pack=True),
        dict(dkv_mode="atomic"),
        dict(dkv_mode="atomic", head_pack=True, edge_tiles=True),
        dict(loop_sync="lds_only"),
        dict(step_order="head_outer"),
        dict(step_order="head_outer", edge_tiles=True, loop_sync="lds_only"),
    ):
        assert _build(AttnBwdSpec(head_size=64, **kw), "gfx942")
    # illegal specs are rejected first (ValueError)
    with pytest.raises(ValueError):
        build_attn_bwd_main(
            AttnBwdSpec(head_size=64, transpose_source="tr_read"), arch="gfx942"
        )


def test_aux_spec_matches_when_the_aux_module_exists():
    if importlib.util.find_spec("kernels.common.attention_bwd_aux") is None:
        pytest.skip("the shared prep / convert module is not in this tree")
    spec = AttnBwdSpec(
        head_size=64, dtype="bf16", seq_mode="thd", stage_vec=1, ws_layout="token_major"
    )
    aux = spec.aux_spec("prep")
    for f in ("head_size", "dtype", "seq_mode", "stage_vec", "ws_layout"):
        assert getattr(aux, f) == getattr(spec, f)


# --------------------------------------------------------------------------- inert kK4,
# the backend agpr_alloc value and the workgroup register share
def _legal_tiles(arch, d):
    """(W, kM0, kN0, G4 grid) points of the geometry catalog on ``arch``."""
    facts = attn_bwd_arch_facts(arch)
    for w in facts.legal_waves:
        for km in (16, 32, 64):
            for kn in (16, 32, 64, 128, 192, 256):
                for g4 in {(1, w), (2, max(1, w // 2))}:
                    if g4[0] * g4[1] != w:
                        continue
                    yield dict(waves=w, block_m=km, block_n=kn, warp_grid_g4=g4)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1151"])
@pytest.mark.parametrize("d", [64, 128])
def test_register_estimate_does_not_depend_on_inert_block_k4(arch, d):
    """G4 is emitted one atom-K step at a time, so kK4 changes neither the
    estimate nor the resolved agpr_alloc of any tile (it used to, through the
    dS term, which made kK4 a hidden proxy for agpr_alloc)."""
    seen = 0
    for tile in _legal_tiles(arch, d):
        outs = set()
        for kk in (16, 32, 64):
            if tile["block_n"] % kk:
                continue
            spec = AttnBwdSpec(head_size=d, block_k4=kk, **tile)
            try:
                r = spec.resolved(arch)
                est = register_estimates(r, arch)
            except ValueError:
                continue
            outs.add((r.agpr_alloc, est["arch_vgpr"], est["agpr"]))
        assert len(outs) <= 1, (tile, outs)
        seen += bool(outs)
    assert seen > 10


def test_backend_agpr_alloc_is_kept_and_emits_no_reservation():
    spec = AttnBwdSpec(head_size=64, agpr_alloc="backend")
    assert spec.agpr_alloc == "backend"
    for arch in ("gfx942", "gfx950"):
        r = spec.resolved(arch)
        assert r.agpr_alloc == "backend"  # the policy is not consulted
        # accumulators are counted as AGPRs, as for a policy result of None
        assert register_estimates(r, arch)["agpr"] > 0
        k = build_attn_bwd_main(spec, arch=arch)
        assert "agpr_alloc" not in k.attrs
        none_k = _build(AttnBwdSpec(head_size=64, agpr_alloc=(128, 128)), arch)
        assert none_k.attrs["agpr_alloc"] == (128, 128)
    with pytest.raises(ValueError, match="acc_in_lds"):
        AttnBwdSpec(head_size=64, agpr_alloc="backend", acc_in_lds=True)
    with pytest.raises(ValueError):
        validate_attn_bwd_spec(
            AttnBwdSpec(head_size=64, agpr_alloc="backend"), "gfx1151"
        )
    with pytest.raises(ValueError):
        AttnBwdSpec(head_size=64, agpr_alloc="vgpr")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_eight_wave_tiles_respect_the_workgroup_register_share(arch):
    """Two waves of an eight-wave CTA share each SIMD's 512-register file, so
    arch VGPRs + AGPRs per wave are capped at 256 for every agpr_alloc value
    (the rule used to apply to explicit reservations only)."""
    big = dict(waves=8, block_m=64, block_n=128, block_k4=32, warp_grid_g4=(1, 8))
    for alloc in (None, "backend", (128, 128)):
        spec = AttnBwdSpec(head_size=128, agpr_alloc=alloc, **big)
        r = spec.resolved(arch)
        est = register_estimates(r, arch)
        if est["arch_vgpr"] + est["agpr"] <= 256:
            continue  # this alloc fits; nothing to check
        with pytest.raises(ValueError, match="workgroup share|registers per wave"):
            validate_attn_bwd_spec(spec, arch)
    # four-wave tiles keep the whole 512-register file per wave
    w4 = AttnBwdSpec(
        head_size=128, waves=4, block_m=16, block_n=64, block_k4=32,
        warp_grid_g4=(1, 4), agpr_alloc="backend",
    )  # fmt: skip
    validate_attn_bwd_spec(w4, arch)


def test_shipped_defaults_resolve_as_before():
    """The estimate fix changes no shipped default: every CDNA default still
    resolves to the VGPR form."""
    for arch in ("gfx942", "gfx950"):
        for d in (32, 64, 128):
            for sv in (8, 1):
                for seq in ("batched", "thd"):
                    r = AttnBwdSpec(head_size=d, stage_vec=sv, seq_mode=seq).resolved(
                        arch
                    )
                    assert r.agpr_alloc == (0, 0), (arch, d, sv, seq)


# --------------------------------------------------------------------------- staging
# issue order (qdo_stage) and the stats load placement (stats_load)


def test_staging_knob_defaults_and_value_sets():
    from kernels.common._attention_bwd_body import COPY_ORDERS
    from kernels.common.attention_bwd import QDO_STAGES, STATS_LOADS

    fields = {f.name: f.default for f in dataclasses.fields(AttnBwdSpec)}
    assert fields["qdo_stage"] == "split" and fields["stats_load"] == "at_use"
    assert QDO_STAGES == COPY_ORDERS == ("split", "loads_first", "joint")
    assert STATS_LOADS == ("at_use", "step_top")
    # the defaults are not salted (existing kernel names are unchanged)
    plain = AttnBwdSpec(head_size=64)
    explicit = AttnBwdSpec(head_size=64, qdo_stage="split", stats_load="at_use")
    assert explicit == plain and explicit.kernel_name() == plain.kernel_name()
    assert not plain.non_default_fields()
    # they need no arch resolution (never None) and survive resolution as given
    for arch in BWD_ARCHS:
        r = AttnBwdSpec(head_size=64, stats_load="step_top").resolved(arch)
        assert (r.qdo_stage, r.stats_load) == ("split", "step_top")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1151", "gfx1201"])
@pytest.mark.parametrize("d", BWD_HEAD_SIZES)
@pytest.mark.parametrize("stage_vec", (8, 1))
def test_joint_staging_is_legal_exactly_below_one_pass(arch, d, stage_vec):
    """``qdo_stage = "joint"`` gives the Q and dO tiles one pass of the CTA
    (whole waves per tile): legal iff ``kM0 * D / 8`` chunks are fewer than the
    threads; ``loads_first`` and ``step_top`` are legal on every default."""
    base = AttnBwdSpec(head_size=d, stage_vec=stage_vec)
    r = validate_attn_bwd_spec(base, arch)
    wave = attn_bwd_arch_facts(arch).wave_size
    chunks, threads = r.block_m * d // 8, r.waves * wave
    joint = dataclasses.replace(base, qdo_stage="joint")
    if chunks < threads:
        assert chunks % wave == 0
        assert validate_attn_bwd_spec(joint, arch).qdo_stage == "joint"
    else:
        with pytest.raises(ValueError, match="qdo_stage='joint' needs"):
            validate_attn_bwd_spec(joint, arch)
    for kw in (dict(qdo_stage="loads_first"), dict(stats_load="step_top")):
        validate_attn_bwd_spec(dataclasses.replace(base, **kw), arch)


def test_joint_staging_is_legal_on_every_cdna_wide_default():
    """The CDNA ``stage_vec = 8`` start points leave half the CTA idle in each
    of the two staging copies, which is what the joint distribution fills."""
    for arch in ("gfx942", "gfx950"):
        for d in BWD_HEAD_SIZES:
            validate_attn_bwd_spec(AttnBwdSpec(head_size=d, qdo_stage="joint"), arch)


@pytest.mark.parametrize(
    "kw", [dict(qdo_stage="loads_first"), dict(qdo_stage="joint"),
           dict(stats_load="step_top")],
    ids=lambda kw: "-".join(f"{k}={v}" for k, v in kw.items()),
)  # fmt: skip
def test_staging_knobs_exclude_the_ring_and_dma(kw):
    """The ring / DMA builds own the issue order of the staging and the stats;
    the synchronous-staging knobs are refused with them."""
    with pytest.raises(ValueError, match="the ring owns the issue order"):
        validate_attn_bwd_spec(AttnBwdSpec(head_size=128, ring_depth=2, **kw), "gfx942")
    with pytest.raises(ValueError, match="the ring owns the issue order"):
        validate_attn_bwd_spec(
            AttnBwdSpec(
                head_size=128, global_path="dma", transpose_source="tr_read",
                kt_source="lds", **kw,
            ),
            "gfx950",
        )  # fmt: skip


# --------------------------------------------------------------------------- G4 key
# slicing (g4_slicing, block_k4)

# Kernel names of the shipped defaults (bf16, batched, direct, band), captured
# before g4_slicing existed: the new field must salt nothing at its default.
_SHIPPED_NAMES = {
    (
        "gfx942",
        32,
        8,
    ): "rocke_attn_bwd_main_bf16_d32_batched_direct_band_sv8_0077f3e190",
    (
        "gfx942",
        32,
        1,
    ): "rocke_attn_bwd_main_bf16_d32_batched_direct_band_sv1_5e92067c5a",
    (
        "gfx942",
        64,
        8,
    ): "rocke_attn_bwd_main_bf16_d64_batched_direct_band_sv8_38b0cbda18",
    (
        "gfx942",
        64,
        1,
    ): "rocke_attn_bwd_main_bf16_d64_batched_direct_band_sv1_c05ec75a26",
    (
        "gfx942",
        128,
        8,
    ): "rocke_attn_bwd_main_bf16_d128_batched_direct_band_sv8_f52227d365",
    (
        "gfx942",
        128,
        1,
    ): "rocke_attn_bwd_main_bf16_d128_batched_direct_band_sv1_fb78b18cdb",
    (
        "gfx950",
        32,
        8,
    ): "rocke_attn_bwd_main_bf16_d32_batched_direct_band_sv8_973d4b68e3",
    (
        "gfx950",
        32,
        1,
    ): "rocke_attn_bwd_main_bf16_d32_batched_direct_band_sv1_386ba3ff7c",
    (
        "gfx950",
        64,
        8,
    ): "rocke_attn_bwd_main_bf16_d64_batched_direct_band_sv8_b9735b1ec5",
    (
        "gfx950",
        64,
        1,
    ): "rocke_attn_bwd_main_bf16_d64_batched_direct_band_sv1_792e99dc01",
    (
        "gfx950",
        128,
        8,
    ): "rocke_attn_bwd_main_bf16_d128_batched_direct_band_sv8_a4e953ae55",
    (
        "gfx950",
        128,
        1,
    ): "rocke_attn_bwd_main_bf16_d128_batched_direct_band_sv1_c57260b883",
    (
        "gfx1151",
        32,
        8,
    ): "rocke_attn_bwd_main_bf16_d32_batched_direct_band_sv8_9dd72c63d8",
    (
        "gfx1151",
        64,
        1,
    ): "rocke_attn_bwd_main_bf16_d64_batched_direct_band_sv1_9dd72c63d8",
    (
        "gfx1151",
        128,
        8,
    ): "rocke_attn_bwd_main_bf16_d128_batched_direct_band_sv8_041c3d0a66",
    (
        "gfx1201",
        128,
        1,
    ): "rocke_attn_bwd_main_bf16_d128_batched_direct_band_sv1_041c3d0a66",
}


def test_g4_slicing_default_keeps_names():
    from kernels.common.attention_bwd import G4_SLICINGS

    assert G4_SLICINGS == ("step", "slice", "slice_db", "slice_db_g3")
    plain = AttnBwdSpec(head_size=64)
    assert plain.g4_slicing == "step" and not plain.non_default_fields()
    assert AttnBwdSpec(head_size=64, g4_slicing="step") == plain
    for (arch, d, sv), name in _SHIPPED_NAMES.items():
        spec = AttnBwdSpec(head_size=d, dtype="bf16", stage_vec=sv)
        r = validate_attn_bwd_spec(spec, arch)
        assert r.g4_slicing == "step"
        assert r.kernel_name() == name, (arch, d, sv)
    # a non-default value salts the name
    names = {AttnBwdSpec(head_size=64, g4_slicing=v).kernel_name() for v in G4_SLICINGS}
    assert len(names) == len(G4_SLICINGS)


def test_g4_slicing_values_and_couplings():
    for bad in ("db", "", None, 1):
        with pytest.raises(ValueError, match="g4_slicing"):
            AttnBwdSpec(head_size=64, g4_slicing=bad)
    for v in ("slice", "slice_db", "slice_db_g3"):
        for issue in ("fragment", "fragment_regs"):
            with pytest.raises(ValueError, match="dq_issue='block'"):
                AttnBwdSpec(head_size=64, g4_slicing=v, dq_issue=issue)
        with pytest.raises(ValueError, match="register accumulators"):
            AttnBwdSpec(head_size=64, g4_slicing=v, acc_in_lds=True)
        for arch in ("gfx1151", "gfx1201"):  # WMMA: LDS accumulators
            with pytest.raises(ValueError, match="register accumulators"):
                validate_attn_bwd_spec(AttnBwdSpec(head_size=64, g4_slicing=v), arch)
        for arch in ("gfx942", "gfx950"):
            validate_attn_bwd_spec(AttnBwdSpec(head_size=64, g4_slicing=v), arch)
            # kK4: a multiple of the G4 atom K dividing kN0
            spec = AttnBwdSpec(
                head_size=64, g4_slicing=v, block_n=64, block_k4=64, waves=4,
                warp_grid_g4=(1, 4), block_m=16,
            )  # fmt: skip
            validate_attn_bwd_spec(spec, arch)
            with pytest.raises(ValueError, match="block_n"):
                validate_attn_bwd_spec(
                    dataclasses.replace(spec, block_n=32, waves=2, warp_grid_g4=(1, 2)),
                    arch,
                )
    with pytest.raises(ValueError, match="atom K"):
        validate_attn_bwd_spec(
            AttnBwdSpec(head_size=64, g4_slicing="slice", block_k4=16), "gfx950"
        )
    # slice_db_g3 reads G4's first slice under G3: G3 right before G4
    from kernels.common._attention_bwd_body import G4_AFTER_G3_ORDERS, STAGE_ORDERS

    assert G4_AFTER_G3_ORDERS == ("g1g3g4", "g3g4g1")
    for order in STAGE_ORDERS:
        kw = dict(head_size=64, g4_slicing="slice_db_g3", stage_order=order)
        if order in G4_AFTER_G3_ORDERS:
            validate_attn_bwd_spec(AttnBwdSpec(**kw), "gfx942")
        else:
            with pytest.raises(ValueError, match="stage_order"):
                AttnBwdSpec(**kw)
        for v in ("slice", "slice_db"):  # any order
            validate_attn_bwd_spec(
                AttnBwdSpec(head_size=64, g4_slicing=v, stage_order=order), "gfx942"
            )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_g4_slicing_resolves_to_the_code_it_emits(arch):
    """``slice`` with one atom-K step per slice is ``step`` and ``slice_db``
    over one slice is ``slice`` (same instruction order): the resolved spec,
    its name and its IR are those of the value it emits. ``slice_db_g3`` is
    kept (it moves the first slice under G3)."""
    k4a = 32 if arch == "gfx950" else 16
    base = AttnBwdSpec(head_size=64, dtype="bf16")
    r0 = validate_attn_bwd_spec(dataclasses.replace(base, block_k4=k4a), arch)
    one = validate_attn_bwd_spec(
        dataclasses.replace(base, g4_slicing="slice", block_k4=k4a), arch
    )
    assert one == r0 and one.kernel_name() == r0.kernel_name()
    assert one.g4_slicing == "step"
    # one slice needs kK4 = kN0: a four-wave 16 x 64 tile with kK4 = 64
    tile = dict(block_n=64, block_k4=64, waves=4, warp_grid_g4=(1, 4), block_m=16)
    db = validate_attn_bwd_spec(
        dataclasses.replace(base, g4_slicing="slice_db", **tile), arch
    )
    sl = validate_attn_bwd_spec(
        dataclasses.replace(base, g4_slicing="slice", **tile), arch
    )
    assert db.g4_slicing == "slice" and db == sl
    assert _ir_without_name(
        dataclasses.replace(base, g4_slicing="slice_db", **tile), arch
    ) == (_ir_without_name(dataclasses.replace(base, g4_slicing="slice", **tile), arch))
    g3 = validate_attn_bwd_spec(
        dataclasses.replace(base, g4_slicing="slice_db_g3", **tile), arch
    )
    assert g3.g4_slicing == "slice_db_g3"
    # a sliced value keeps its kK4; two slices or more are distinct kernels
    two = validate_attn_bwd_spec(
        dataclasses.replace(base, g4_slicing="slice_db", block_k4=2 * k4a), arch
    )
    assert two.g4_slicing == "slice_db" and two.block_k4 == 2 * k4a


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("kt", ["reg", "lds"])
def test_register_estimate_slices(arch, kt):
    """Under ``step`` the estimate does not depend on kK4; under key slicing
    the dS term is ``2*kM0*kK4 / (2*wave)`` (plus the K^T slices from LDS),
    monotone in kK4, and equal to ``step`` at kK4 = atom K for the dS term."""
    k4a = 32 if arch == "gfx950" else 16
    tile = dict(head_size=128, waves=4, block_m=16, block_n=128, warp_grid_g4=(1, 4))
    tile.update(kv_residency="reg", kt_source=kt, agpr_alloc=(256, 256))
    tile.update(agpr_route="vgpr_form")
    step = {}
    for v in ("step", "slice", "slice_db", "slice_db_g3"):
        prev = None
        for k4 in (16, 32, 64):
            if k4 % k4a:
                continue
            spec = AttnBwdSpec(block_k4=k4, g4_slicing=v, **tile).resolved(arch)
            est = register_estimates(spec, arch)["arch_vgpr"]
            if spec.g4_slicing == "step":
                step.setdefault(arch, est)
                assert est == step[arch]
                continue
            want = step[arch] + 16 * (k4 - k4a) // 64
            if kt == "lds":
                want += 2 * 128 * k4 // (2 * 64 * 4)
            assert est == want, (v, k4)
            assert prev is None or est > prev
            prev = est


# The asm_g13 + s_in_agpr tiles (four waves, K / V / K^T in registers) and
# their (arch-VGPR, AGPR) estimates and smallest legal reservation: the split
# plan with S / dP in AGPRs, the dQ partial in arch VGPRs.
_ASM_S_TILES = [
    ("gfx950", 128, 32, 128, (232, 160), (192, 192)),
    ("gfx950", 128, 16, 192, (228, 216), (256, 256)),
    ("gfx942", 128, 16, 128, (172, 144), (192, 192)),
    ("gfx942", 64, 32, 128, (136, 96), (128, 128)),
]


@pytest.mark.parametrize(
    "row", _ASM_S_TILES, ids=lambda r: f"{r[0]}-d{r[1]}-{r[2]}x{r[3]}"
)
def test_asm_g13_s_in_agpr_issues_g0_g2_as_zero_init_clusters(row):
    """``asm_g13`` with ``s_in_agpr``: G0 / G2 join G1 / G3 as AGPR-form asm
    clusters (no G0 / G2 intrinsic left; G4 keeps its intrinsics), the first
    K step of each G0 / G2 GEMM reads srcC 0 with no tied input, every
    cluster starts with the lead s_nop and ends with the tail s_nop; the
    register estimate and the copy budget are the S / dP-in-AGPR plan."""
    import re

    from kernels.common._attention_bwd_body import BwdTileGeometry
    from kernels.common._attention_bwd_caps import AGPR_ASM_TAIL_NOP

    arch, d, km, kn, est, alloc = row
    kw = dict(head_size=d, dtype="bf16", waves=4, block_m=km, block_n=kn)
    kw.update(block_k4=32, warp_grid_g4=(1, 4), kv_residency="reg", kt_source="reg")
    kw.update(agpr_route="asm_g13", sched="none")
    spec = AttnBwdSpec(agpr_alloc=alloc, s_in_agpr=True, **kw)
    r = validate_attn_bwd_spec(spec, arch)
    assert tuple(register_estimates(r, arch).values()) == est
    for smaller in ((128, 128), (192, 192)):
        if smaller[0] < alloc[0]:
            with pytest.raises(ValueError, match="below the AGPR estimate"):
                validate_attn_bwd_spec(
                    dataclasses.replace(spec, agpr_alloc=smaller), arch
                )
    g = BwdTileGeometry.from_spec(r, arch)
    wave, w = 64, 4
    assert agpr_copy_budget(r, arch) == {
        "reads": 2 * km * kn // (wave * w),
        "writes": 0,
        "moves": 0,
    }
    plain = dataclasses.replace(spec, s_in_agpr=False)
    ir, ir0 = _ir_without_name(spec, arch), _ir_without_name(plain, arch)
    call = r"call <4 x float> @llvm\.amdgcn\.mfma"
    n_int, n_int0 = len(re.findall(call, ir)), len(re.findall(call, ir0))
    n_asm = len(re.findall(r"v_mfma_f32_16x16x\d+_bf16 \$", ir))
    n_asm0 = len(re.findall(r"v_mfma_f32_16x16x\d+_bf16 \$", ir0))
    g02 = 2 * g.g0.mma_per_wave
    assert n_asm - n_asm0 == n_int0 - n_int  # exactly the G0 / G2 MFMAs moved
    assert (n_asm - n_asm0) % g02 == 0 and n_asm > n_asm0
    assert n_int % g.g4.mma_per_wave == 0  # only G4 is an intrinsic
    steps = (n_asm - n_asm0) // g02  # q-step bodies in the kernel
    # srcC 0 on the first K step of G0 and of G2 (one per fragment)
    frags = g.g0.frags_m * g.g0.frags_n
    assert len(re.findall(r"v_mfma_f32_16x16x\d+_bf16 \$\d+, \$\d+, \$\d+, 0", ir)) == (
        2 * frags * steps
    )
    tail = AGPR_ASM_TAIL_NOP[g.g1.atom_k]
    clusters = ir.count('asm sideeffect "s_nop 1\\0A')
    assert clusters == ir.count(f's_nop {tail}", ')
    assert "=a" in ir
    # S then dP pinned to the top 2 * R AGPRs of the reservation (n, n): one
    # pinned output per fragment in every G0 / G2 cluster
    n, rr = alloc[0], frags * 4
    per = g.g0.k_steps if r.asm_cluster == "k_step" else 1
    for i in range(2 * frags):
        lo = n - 2 * rr + 4 * i
        assert ir.count(f"={{a[{lo}:{lo + 3}]}}") == steps * per, (lo, steps, per)
    assert "={a[" not in ir0  # the G1 / G3 clusters are not pinned


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_asm_cluster_canonical_spelling_counts_g0_g2(arch):
    """With ``s_in_agpr`` the G0 / G2 asm GEMMs (K = D) count for the
    ``asm_cluster = "gemm"`` canonical spelling: a single G1 / G3 K step no
    longer makes ``gemm`` the ``k_step`` cluster when G0 / G2 have several."""
    kw = dict(head_size=64, agpr_alloc=(128, 128), agpr_route="asm_g13", sched="none")
    kw.update(block_m=16, block_n=64, waves=4, warp_grid_g4=(1, 4), block_k4=32)
    one = AttnBwdSpec(asm_cluster="gemm", **kw).resolved(arch)
    assert one.asm_cluster == "k_step"  # G1 / G3: one K step
    s = AttnBwdSpec(asm_cluster="gemm", s_in_agpr=True, **kw).resolved(arch)
    assert s.asm_cluster == "gemm"  # G0 / G2: D / atom K > 1 steps


def test_aux_spec_carries_the_aux_tuning_values():
    spec = AttnBwdSpec(head_size=64, stats_load="step_top", qdo_stage="joint")
    prep = spec.aux_spec("prep", rows_per_cta=16)
    conv = spec.aux_spec("convert", rows_per_cta=32, convert_vec=16)
    assert (prep.rows_per_cta, prep.convert_vec) == (16, 8)
    assert (conv.rows_per_cta, conv.convert_vec) == (32, 16)
    # main-kernel knobs never reach the aux kernels
    assert spec.aux_spec("prep") == AttnBwdSpec(head_size=64).aux_spec("prep")
    with pytest.raises(ValueError, match="convert stage only"):
        spec.aux_spec("prep", convert_vec=16)
