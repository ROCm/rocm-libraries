# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the WP2d grouped / wide stage functions (``stages2.moe_wide``, ``stages2.b3_wide``, ``stages2.q2_wide``).

The GPU gates (bitwise vs the plain FlyDSL mono grouped build in situ, the whole scratch vs the served hybrid, must-fail
mutants, DESIGN 9.1 timing) run in ``hackathon-prep/qwen38/truemono/WP2d/wp2d_insitu.py``."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("rocke")

from kernels.gfx950.qwen38_mono import abi as A  # noqa: E402
from kernels.gfx950.qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function  # noqa: E402
from kernels.gfx950.qwen38_mono.stages2 import b3_wide as BW  # noqa: E402
from kernels.gfx950.qwen38_mono.stages2 import moe_wide as W  # noqa: E402
from kernels.gfx950.qwen38_mono.stages2 import q2_wide as QW  # noqa: E402

FLAVOR = os.environ.get("ROCKE_LLVM_FLAVOR", "llvm22")
WIDE = ((6, 3), (8, 4), (9, 3), (12, 3), (12, 4), (16, 4), (6, 2), (8, 2))


def _atom_root():
    root = os.environ.get("ATOM_TRUEMONO_ROOT") or os.environ.get("Q38_ATOM_DIR")
    if not root:
        try:
            import atom  # noqa: F401

            return None
        except Exception:  # noqa: BLE001
            pytest.skip("no ATOM checkout (ATOM_TRUEMONO_ROOT)")
    return root


def _lower(kern, sym):
    return lower_device_function(kern, arch="gfx950", symbol=sym, llvm_flavor=FLAVOR,
                                 knobs=DeviceFunctionKnobs(lds="arg", inline="always")).ir  # fmt: skip


# ---------------------------------------------------------------- moe_wide


def test_moe_wide_m3_placement_matches_atom():
    root = _atom_root()
    from kernels.gfx950.qwen38_mono.mono_layout import _import_atom

    LW = _import_atom("atom.models.qwen4_exp_mono.kernels.layout_wide", root)
    for rows, g in WIDE:
        L = LW.for_wide(rows, g)
        assert W.wide_m3(rows) == (L.base["m3"], L.tasks["m3"]), (rows, g)


@pytest.mark.parametrize("rows,group", WIDE)
def test_moe_wide_lowers_and_fits_the_arena(rows, group):
    root = _atom_root()
    arena = W.arena_words(rows, group, root)
    for spec in ("", "gu_round=2", "epi=dense,dn_round=8", "gu_next=first", "gu_next=epi,dn_early=0,order=fill"):
        kern, plan, words = W.build_function("m4", rows, group, W.knobs_from_spec(spec), arena=arena)
        assert words <= arena
        ir = _lower(kern, f"rocke_q38_m4_r{rows}g{group}")
        assert f"@rocke_q38_m4_r{rows}g{group}" in ir
    kern, _p, _w = W.build_function("m5", rows, group, W.WideKnobs())
    assert "ret void" in _lower(kern, f"rocke_q38_m5_r{rows}g{group}")


def test_moe_wide_knob_spec_round_trip_and_refusals():
    for spec in ("", "gu_round=2,epi=dense", "dn_round=2,dn_early=0,order=fill,w_cpol=2", "mb_impl=asm",
                 "sh_ctas=32,sh_base=160,iod_split=0,line_order=router_dn,combine_base=-1"):  # fmt: skip
        k = W.knobs_from_spec(spec)
        assert W.knobs_from_spec(W.knobs_spec(k)) == k
    for bad in ("gu_round=3", "gu_next=x", "epi=x", "dn_round=0", "dn_early=2", "order=x", "router=1",
                "gu_round=2,gu_next=first", "mutate=nope"):  # fmt: skip
        with pytest.raises(ValueError):
            W.knobs_from_spec(bad)
    with pytest.raises(ValueError):
        W.WidePlan(4, A.EXPERTS, W.WideKnobs().fused()).validate()


def test_moe_wide_every_mutant_changes_the_ir():
    base = _lower(W.build_function("m4", 8, 4, W.WideKnobs())[0], "rocke_q38_m4_r8g4")
    nbase = _lower(W.build_function("m4", 8, 4, W.knobs_from_spec("numa=dn"))[0], "rocke_q38_m4_r8g4")
    for m in W.MUTANTS:
        pre, ref = ("numa=dn,", nbase) if m == "numa_copy" else ("", base)
        ir = _lower(W.build_function("m4", 8, 4, W.knobs_from_spec(f"{pre}mutate={m}"))[0], "rocke_q38_m4_r8g4")
        assert ir != ref, m


def test_moe_wide_numa_area_layout_and_lowering(tmp_path):
    """Knob numa: the compact logical area holds 16 rows' dn and ys, disjoint, inside ATOM's WIDE_LOGICAL; every
    mode lowers, polls the area (the header's page mask read), and the sidecar names the header offset."""
    assert W.NUMA_YS0 >= W.WIDE_ROWS * A.DN_ROW * (A.TOP_K + 1)
    assert (W.NUMA_YS0 + W.WIDE_ROWS * W.NUMA_YS_ROW) * A.PAIR <= W.WIDE_NUMA_LOGICAL
    assert W.WIDE_NUMA_HDR % 256 == 0 and W.WIDE_NUMA_HDR >= A.mailbox_bytes(W.WIDE_ROWS)
    off = _lower(W.build_function("m4", 16, 4, W.WideKnobs())[0], "rocke_q38_m4_r16g4")
    irs = {}
    for mode in ("dn", "ys", "both"):
        irs[mode] = _lower(W.build_function("m4", 16, 4, W.knobs_from_spec(f"numa={mode}"))[0], "rocke_q38_m4_r16g4")
        assert irs[mode] != off and str(W.WIDE_NUMA_HDR) in irs[mode], mode
    assert len(set(irs.values())) == 3
    for bad in ("numa=x", "mutate=numa_copy", "numa=ys,mutate=numa_copy"):
        with pytest.raises(ValueError):
            W.knobs_from_spec(bad)
    root = _atom_root()
    w = W.write_functions(str(tmp_path), ["r16g4"], W.knobs_from_spec("numa=both"), flavor=FLAVOR, atom_root=root)
    assert {x["stage"] for x in w} == {"m4", "m5"}
    import json

    assert json.load(open(tmp_path / "m4_r16g4.json"))["numa_wide_hdr"] == W.WIDE_NUMA_HDR
    assert "numa_wide_hdr" not in json.load(open(tmp_path / "m5_r16g4.json"))


def test_moe_wide_ballot_asm_removes_every_ir_ballot():
    """Knob ballot=asm: no ``amdgcn.ballot`` left (the direct backend lowers no ``tile.wave_ballot``), the asm
    compare in its place; the default keeps the IR ballots."""
    spec = "map=groups,gu_round=2,gu_next=epi,epi=compact"
    ir0 = _lower(W.build_function("m4", 16, 4, W.knobs_from_spec(spec))[0], "rocke_q38_m4_r16g4")
    ir1 = _lower(W.build_function("m4", 16, 4, W.knobs_from_spec(spec + ",ballot=asm"))[0], "rocke_q38_m4_r16g4")
    assert "amdgcn.ballot" in ir0 and "v_cmp_ne_u32_e64" not in ir0
    assert "amdgcn.ballot" not in ir1 and "v_cmp_ne_u32_e64" in ir1
    with pytest.raises(ValueError):
        W.knobs_from_spec("ballot=x")


def test_moe_wide_numa_header_matches_atom():
    root = _atom_root()
    from kernels.gfx950.qwen38_mono.mono_layout import _import_atom

    try:
        NU = _import_atom("atom.models.qwen4_exp_mono.kernels.numa", root)
    except ImportError as e:  # numa.py imports flydsl
        pytest.skip(f"ATOM numa needs {e}")
    if not hasattr(NU, "wide_header_offset"):
        pytest.skip("this ATOM has no ATOM_Q38_NUMA_WIDE")
    assert NU.wide_header_offset() == W.WIDE_NUMA_HDR
    assert NU.WIDE_LOGICAL == W.WIDE_NUMA_LOGICAL
    assert NU.header_offset() == W.MF.NUMA_HDR


def test_moe_wide_plan_covers_every_unit():
    """Every (slot, unit) of every U is on exactly one CTA (the line math the body emits)."""
    plan = W.WidePlan(16, A.EXPERTS, W.WideKnobs().fused())
    for U in (10, 37, 100, 160):
        gu, dn = {}, {}
        for bid, (_k, g, d, _j) in plan.assignment(U).items():
            for x in g:
                gu[x] = gu.get(x, 0) + 1
            for x in d:
                dn[x] = dn.get(x, 0) + 1
        want_gu = {(s, u) for s in range(-1, U) for u in range(W.GU_UNITS)}
        want_dn = {(s, u) for s in range(-1, U) for u in range(W.DN_UNITS)}
        assert set(gu) == want_gu and all(v == 1 for v in gu.values())
        assert set(dn) == want_dn and all(v == 1 for v in dn.values())


def test_moe_wide_writes_grouped_narrow_and_wide(tmp_path):
    root = _atom_root()
    w = W.write_functions(str(tmp_path), ["r4g2", "r8g4"], W.WideKnobs(), flavor=FLAVOR, atom_root=root)
    stages = {(x["stage"], x["variant"]) for x in w}
    assert stages == {("m3", "r4g2"), ("m4", "r4g2"), ("m5", "r4g2"), ("m4", "r8g4"), ("m5", "r8g4")}
    import json

    side = json.load(open(tmp_path / "m4_r8g4.json"))
    assert side["requires"] == {"m5": side["pair"]} and side["wide"]
    narrow = json.load(open(tmp_path / "m3_r4g2.json"))
    assert narrow["symbol"] == "rocke_q38_m3_r4g2" and narrow["fused"]


# ---------------------------------------------------------------- b3_wide


@pytest.mark.parametrize("rows,group", [x for x in WIDE if x[1] >= 2])
def test_b3_wide_lowers_and_matches_the_flydsl_phase(rows, group):
    root = _atom_root()
    from kernels.gfx950.qwen38_mono.mono_layout import _import_atom, from_atom

    LW = _import_atom("atom.models.qwen4_exp_mono.kernels.layout_wide", root)
    lay = from_atom(root)
    for spec in ("", "w_order=group", "w_stores=end", "split=2", "norms=redundant,hoist=gates_z"):
        try:
            kern, words = BW.build_function_kernel(BW.WideShape(rows, group), BW.knobs_from_spec(spec), lay,
                                                   W.arena_words(rows, group, root))  # fmt: skip
        except ValueError as e:  # a knob set past the layer's arena is refused, never built
            assert spec and "LDS arena" in str(e), (spec, e)
            continue
        if spec == "":
            # the default arena is FlyDSL's B3 phase word for word (``layout_wide`` "d3")
            assert words == LW.for_wide(rows, group).phases["d3"]
        assert "@rocke_q38_b3_" in _lower(kern, f"rocke_q38_b3_r{rows}g{group}")


def test_b3_wide_refusals_and_mutants():
    for bad in ("numa=1", "xchg=numa2", "pre=dma", "core_put=direct", "stores=end", "w_bax=x", "w_mutate=x"):
        with pytest.raises((ValueError, TypeError)):
            BW.knobs_from_spec(bad)
    with pytest.raises(ValueError):
        BW.WideShape(4, 2)
    root = _atom_root()
    from kernels.gfx950.qwen38_mono.mono_layout import from_atom

    lay = from_atom(root)
    base = _lower(BW.build_function_kernel(BW.WideShape(16, 4), BW.WideB3Knobs(), lay)[0], "rocke_q38_b3_r16g4")
    for m in BW.MUTANTS:
        ir = _lower(BW.build_function_kernel(BW.WideShape(16, 4), BW.knobs_from_spec(f"w_mutate={m}"), lay)[0],
                    "rocke_q38_b3_r16g4")  # fmt: skip
        assert ir != base, m


# ---------------------------------------------------------------- q2_wide


def test_q2_grouped_lowers_every_variant_and_mutant():
    root = _atom_root()
    from kernels.gfx950.qwen38_mono.mono_layout import from_atom

    lay = from_atom(root)
    for rows, group in ((4, 2),) + WIDE:
        base = _lower(QW.build_function_kernel(rows, group, QW.Q2WideKnobs(), lay), f"rocke_q38_q2_r{rows}g{group}")
        for m in QW.MUTANTS:
            if m == "pass_row" and rows <= 8:
                continue
            ir = _lower(QW.build_function_kernel(rows, group, QW.knobs_from_spec(f"mutate={m}"), lay),
                        f"rocke_q38_q2_r{rows}g{group}")  # fmt: skip
            assert ir != base, (rows, group, m)
    with pytest.raises(ValueError):
        QW.build_function_kernel(8, 4, QW.knobs_from_spec("lsrc=args"), lay)
