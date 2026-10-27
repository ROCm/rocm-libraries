# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Spec, IR, ABI and resource tests of the unified (paged) backward main kernel.

The instance set covers gfx942, gfx950, gfx1151 and gfx1201: every head size
and dtype in both page modes and both mask classes on the arch's default tile,
plus (d64, bf16) every shipped tile configuration (the ``"multi"`` default and
``kN0 = 64`` tiles, the ``"sub"`` instances, the one-wave fallback), head
packing, edge tiles and the narrow-access instances.

* spec: ``head_size`` is the only required field; the knob fields are the
  general family's (minus the problem keys the unified family fixes) with the
  same defaults; illegal page classes raise before any IR;
* lowering at every LLVM flavor; kernel signature equal to the
  ``rocke.attn_bwd_unified.v1`` main list, no ``noalias``, inputs
  ``readonly``; the paged aux roles' signatures equal their lists; unique
  kernel names;
* block table: every entry is read inside a 0/1-trip loop whose trip is
  ``j < ceil(len_kv / page)`` (an entry past the live blocks is never read),
  once per page of the tile (one in the ``"multi"`` class, ``kN0 / p`` in the
  ``"sub"`` class), outside the q loops;
* K / V cache reads only inside the CTA liveness branch, never in a loop; no
  matrix op inside a branch; the q step equals the general family's THD step
  op for op (same tile step);
* 64-bit rebase of every cache, the block table and the workspaces;
* epilogues: direct page mode stores dK / dV into the gradient caches under a
  branch and never touches the page workspace; atomic page mode adds ``2 *
  kN0 * D / (W * wave)`` fp32 values per lane into ``WS_PDK`` / ``WS_PDV``
  inside the liveness branch and never touches the gradient caches;
* counts per q step equal the stage-table model;
* the resource gate on the shipped instances (comgr + disassembly).
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from functools import cache

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common._attention_bwd_body import BwdTileGeometry
from kernels.common.attention_bwd import (
    AttnBwdSpec,
    attn_bwd_arch_facts,
    build_attn_bwd_main,
    lds_plan,
)
from kernels.common.attention_bwd_aux import (
    AttnBwdAuxSpec,
    build_attn_bwd_page_convert,
    build_attn_bwd_page_zero,
)
from kernels.common.attention_bwd_unified import (
    UNIFIED_KNOBS,
    AttnBwdUnifiedSpec,
    attn_bwd_unified_params,
    build_attn_bwd_unified_main,
    validate_attn_bwd_unified_spec,
)
from kernels.common.attention_bwd_unified_plan import attn_bwd_unified_tile_configs

from ._attention_bwd_test_utils import kernel_signature
from .test_attention_bwd_ir import _walk, q_loops, step_bodies

ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
CDNA = ("gfx942", "gfx950")
FLAVORS = ("llvm20", "llvm22", "llvm23")
HEADS = (32, 64, 128)
DTYPES = ("fp16", "bf16")
MODES = ("direct", "atomic")
MASKS = ("band", "none")
INPUTS = {"Q", "KC", "VC", "dO", "WS_LSE2", "WS_DSUM", "CU_SEQLENS_Q", "SEQ_LENS_KV",
          "BLOCK_TABLE", "WORKLIST"}  # fmt: skip
VARIANTS = {
    "plain": {},
    "pack": dict(head_pack=True),
    "edge": dict(edge_tiles=True),
    "sv1": dict(stage_vec=1),
    "sv1_pack": dict(stage_vec=1, head_pack=True),
}


def _rows():
    rows = []
    for arch in ARCHS:
        for d in HEADS:
            for dt in DTYPES:
                for mode in MODES:
                    for mask in MASKS:
                        rows.append((arch, "default", d, dt, mode, mask, "plain"))
        for cfg in attn_bwd_unified_tile_configs(arch, 64, 8):
            for mode in MODES:
                rows.append((arch, cfg, 64, "bf16", mode, "band", "plain"))
        for var in VARIANTS:
            if var == "plain":
                continue
            for mode in MODES:
                rows.append((arch, "default", 64, "bf16", mode, "band", var))
                if arch in CDNA:
                    rows.append((arch, "sub32", 64, "bf16", mode, "band", var))
    # one row per distinct instance (on RDNA the fallback is the default tile)
    out, seen = [], set()
    for r in dict.fromkeys(rows):
        if not _legal(r):
            continue
        key = (r[0], resolved_of(*r).kernel_name())
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _tile_knobs(arch, cfg, d, var):
    sv = VARIANTS[var].get("stage_vec", 8)
    cfgs = attn_bwd_unified_tile_configs(arch, d, sv)
    return dict(cfgs[cfg][1]) if cfg in cfgs else None


def spec_of(arch, cfg, d, dt, mode, mask, var):
    knobs = _tile_knobs(arch, cfg, d, var)
    if knobs is None:
        raise ValueError(f"{cfg} not shipped")
    return AttnBwdUnifiedSpec(
        head_size=d, dtype=dt, page_mode=mode, mask_class=mask,
        **{**knobs, **VARIANTS[var]},
    )  # fmt: skip


def _legal(row):
    try:
        validate_attn_bwd_unified_spec(spec_of(*row), row[0])
    except ValueError:
        return False
    return True


def _ids(row):
    return "-".join(map(str, row))


@cache
def kernel_of(*row):
    return build_attn_bwd_unified_main(spec_of(*row), arch=row[0])


@cache
def ir_of(*row, flavor="llvm22"):
    return lower(kernel_of(*row), arch=row[0], llvm_flavor=flavor)


def resolved_of(*row):
    return validate_attn_bwd_unified_spec(spec_of(*row), row[0])


ROWS = _rows()


def _derived(kernel, names):
    out = {f"%{n}" for n in names}
    for op, _ in _walk(kernel.body.ops):
        if op.name == "tile.global_ptr_add" and op.operands[0].name in out:
            out.add(op.results[0].name)
    return out


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


def test_required_fields_and_knob_fields():
    req = [
        f.name
        for f in dataclasses.fields(AttnBwdUnifiedSpec)
        if f.default is dataclasses.MISSING
    ]
    assert req == ["head_size"]
    general = {f.name: f.default for f in dataclasses.fields(AttnBwdSpec)}
    unified = {f.name: f.default for f in dataclasses.fields(AttnBwdUnifiedSpec)}
    fixed = {"seq_mode", "dkv_mode", "dq_mode"}
    assert set(general) - fixed - {
        "head_size",
        "dtype",
        "mask_class",
        "stage_vec",
        "name",
    } == set(UNIFIED_KNOBS)
    for k in UNIFIED_KNOBS:
        assert unified[k] == general[k], k
    assert unified["page_mode"] == "atomic" and unified["sub_page"] is None
    assert unified["mask_class"] == "band" and unified["dtype"] == "bf16"


def test_general_spec_mapping():
    s = AttnBwdUnifiedSpec(head_size=64, page_mode="direct", edge_tiles=True)
    g = s.general_spec()
    assert (g.seq_mode, g.dkv_mode, g.dq_mode, g.edge_tiles) == (
        "thd",
        "direct",
        "atomic",
        True,
    )
    assert AttnBwdUnifiedSpec(head_size=64).general_spec().dkv_mode == "atomic"


@pytest.mark.parametrize(
    "kw,arch,match",
    [
        (dict(page_mode="paged"), "gfx942", "page_mode"),
        (dict(sub_page=24), "gfx942", "sub_page"),
        (dict(sub_page=0), "gfx942", "sub_page"),
        (dict(sub_page=True), "gfx942", "sub_page"),
        (dict(sub_page=128), "gfx942", "below it"),  # kN0 = 128: the multi class
        (dict(sub_page=48), "gfx942", "divide"),
        (dict(sub_page=16), "gfx1151", "below it"),  # one-wave kN0 = 16
        (dict(waves=3), "gfx942", "waves"),
        (dict(mask_class="causal"), "gfx942", "mask_class"),
    ],
)
def test_illegal_specs_raise(kw, arch, match):
    with pytest.raises(ValueError, match=match):
        validate_attn_bwd_unified_spec(AttnBwdUnifiedSpec(head_size=64, **kw), arch)


def test_kernel_names_are_salted_and_unique():
    names = {}
    for row in ROWS:
        name = kernel_of(*row).name
        assert name == resolved_of(*row).kernel_name()
        assert name.startswith(
            f"rocke_attn_bwd_unified_main_{row[3]}_d{row[2]}_{row[4]}_{row[5]}_"
        )
        assert names.setdefault((row[0], name), row) == row, (name, row)


def test_tile_configs_per_arch():
    for arch in CDNA:
        for d in HEADS:
            cfgs = attn_bwd_unified_tile_configs(arch, d, 8)
            assert {"default", "fallback"} <= set(cfgs)
            assert any(c == "sub" for c, _ in cfgs.values())
            fb = validate_attn_bwd_unified_spec(
                AttnBwdUnifiedSpec(head_size=d, **cfgs["fallback"][1]), arch
            )
            assert (fb.waves, fb.block_n) == (1, 16)
    for arch in ("gfx1151", "gfx1201"):
        cfgs = attn_bwd_unified_tile_configs(arch, 64, 8)
        assert set(cfgs) == {"default", "fallback"}


# ---------------------------------------------------------------------------
# lowering, signature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", ROWS, ids=_ids)
def test_instance_builds_and_lowers(row):
    ir = ir_of(*row)
    assert "alloca" not in ir and "addrspace(5)" not in ir
    assert "define amdgpu_kernel void @" in ir


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("arch", ARCHS)
def test_lowers_at_every_flavor(arch, flavor):
    for row in ROWS:
        if row[0] == arch and row[2] == 64 and row[6] in ("plain", "pack"):
            assert "define amdgpu_kernel void @" in ir_of(*row, flavor=flavor)


@pytest.mark.parametrize("arch", ARCHS)
def test_kernarg_order_matches_params(arch):
    for row in ROWS:
        if row[0] != arch or row[2] != 128:
            continue
        sig = kernel_signature(ir_of(*row))
        assert [(n, k) for n, k, _ in sig] == list(attn_bwd_unified_params("main"))
        for name, kind, text in sig:
            assert "noalias" not in text
            if kind == "Q":
                assert ("readonly" in text) == (name in INPUTS), text


@pytest.mark.parametrize("arch", ARCHS)
def test_page_role_signatures(arch):
    for d in HEADS:
        spec = AttnBwdUnifiedSpec(head_size=d).aux_spec("page_zero")
        assert spec.kv_target == "paged" and spec.ws_layout == "head_major"
        for stage, build in (
            ("page_zero", build_attn_bwd_page_zero),
            ("page_convert", build_attn_bwd_page_convert),
        ):
            ir = lower(build(spec, arch=arch), arch=arch, llvm_flavor="llvm22")
            sig = kernel_signature(ir)
            assert [(n, k) for n, k, _ in sig] == list(attn_bwd_unified_params(stage))
            assert all("noalias" not in t for _n, _k, t in sig)
    assert AttnBwdUnifiedSpec(head_size=64).aux_spec("prep") == AttnBwdAuxSpec(
        head_size=64, dtype="bf16", seq_mode="thd"
    )


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


def _bt_loads(kernel):
    bt = _derived(kernel, ("BLOCK_TABLE",))
    return [
        (op, anc)
        for op, anc in _walk(kernel.body.ops)
        if op.name.startswith("memref.global_load") and op.operands[0].name in bt
    ]


@pytest.mark.parametrize("row", ROWS, ids=_ids)
def test_block_table_reads_are_guarded_by_the_live_blocks(row):
    kernel = kernel_of(*row)
    r = resolved_of(*row)
    loads = _bt_loads(kernel)
    pieces = 1 if r.sub_page is None else r.block_n // r.sub_page
    assert len(loads) == pieces
    in_q = {id(op) for lp in q_loops(kernel) for op, _ in _walk(lp.regions[0].ops)}
    ops = [op for op, _ in _walk(kernel.body.ops)]
    by_result = {id(res): op for op in ops for res in op.results}
    for op, anc in loads:
        assert id(op) not in in_q
        guard = anc[-1]
        assert guard.name == "scf.for"
        trip = guard.operands[1].op
        assert trip.name == "arith.select"
        cond = trip.operands[0].op
        assert cond.name == "arith.cmp" and cond.attrs["pred"] == "lt"
        # the compare's bound is ceil(len_kv / page): (len + page - 1) / page
        bound = by_result[id(cond.operands[1])]
        assert bound.name == "arith.div"
        # the loaded index is the compared block index
        assert op.operands[1] is cond.operands[0]


@pytest.mark.parametrize("row", ROWS, ids=_ids)
def test_cache_reads_inside_the_liveness_branch(row):
    kernel = kernel_of(*row)
    loops = q_loops(kernel)
    kv = _derived(kernel, ("KC", "VC"))
    n = 0
    for op, anc in _walk(kernel.body.ops):
        if op.name.startswith("memref.global_load") and op.operands[0].name in kv:
            assert any(a.name == "scf.if" for a in anc), op.name
            assert not any(lp in anc for lp in loops)
            n += 1
        if op.name == "tile.mma":
            assert not any(a.name in ("scf.if", "scf.if_else") for a in anc)
    assert n > 0


def _census(ops):
    """Op counts of a q step, without the zero extensions: the unified ABI
    passes the Q / dO strides as i32 (widened in the step's 64-bit head
    rebase), the general v3 ABI as i64."""
    return Counter(op.name for op, _ in _walk(ops) if op.name != "arith.zext")


@pytest.mark.parametrize(
    "row", [r for r in ROWS if r[6] in ("plain", "pack", "edge")], ids=_ids
)
def test_q_step_is_the_general_thd_step(row):
    """The unified q step is the general family's THD q step, op for op."""
    r = resolved_of(*row)
    g = r.general_spec()
    gk = build_attn_bwd_main(g, arch=row[0])
    ours = [_census(ops) for ops in step_bodies(kernel_of(*row))]
    theirs = [_census(ops) for ops in step_bodies(gk)]
    assert ours == theirs


@pytest.mark.parametrize("row", ROWS, ids=_ids)
def test_epilogue_by_page_mode(row):
    arch = row[0]
    r = resolved_of(*row)
    kernel = kernel_of(*row)
    wave = attn_bwd_arch_facts(arch).wave_size
    in_q = {id(op) for lp in q_loops(kernel) for op, _ in _walk(lp.regions[0].ops)}
    caches = _derived(kernel, ("DKC", "DVC"))
    pws = _derived(kernel, ("WS_PDK", "WS_PDV"))
    epi_atomics, cache_stores, pws_uses = [], [], []
    for op, anc in _walk(kernel.body.ops):
        if not op.operands:
            continue
        ptr = op.operands[0].name
        if op.name == "memref.global_atomic_add_f32" and id(op) not in in_q:
            epi_atomics.append((op, anc))
        if op.name.startswith("memref.global_store") and ptr in caches:
            cache_stores.append((op, anc))
        if op.name.startswith("memref.global") and ptr in pws:
            pws_uses.append(op)
    if r.page_mode == "direct":
        assert cache_stores and not epi_atomics and not pws_uses
        for _op, anc in cache_stores:
            assert any(a.name == "scf.if" for a in anc)
    else:
        assert not cache_stores
        assert all(op.operands[0].name in pws for op, _ in epi_atomics)
        assert len(epi_atomics) == 2 * r.block_n * r.head_size // (r.waves * wave)
        for _op, anc in epi_atomics:
            assert any(a.name == "scf.if" for a in anc)


@pytest.mark.parametrize("row", ROWS, ids=_ids)
def test_mma_and_atomic_counts_per_step(row):
    arch = row[0]
    r = resolved_of(*row)
    geom = BwdTileGeometry.from_spec(r.general_spec(), arch)
    m = geom.stage_table().mfma
    wave = attn_bwd_arch_facts(arch).wave_size
    for ops in step_bodies(kernel_of(*row)):
        mmas = [op for op, _ in _walk(ops) if op.name == "tile.mma"]
        assert len(mmas) == m.g0 + m.g2 + m.g1 + m.g3 + m.g4
        atomics = [
            op for op, _ in _walk(ops) if op.name == "memref.global_atomic_add_f32"
        ]
        assert len(atomics) == r.block_m * r.head_size // (r.waves * wave)


@pytest.mark.parametrize("arch", ARCHS)
def test_rebase_is_64bit(arch):
    """Every cache, the block table and every workspace is addressed through a
    64-bit byte offset (``getelementptr inbounds i8 ... i64``)."""
    for row in ROWS:
        if row[0] != arch or row[2] != 64 or row[6] != "plain":
            continue
        ir = ir_of(*row)
        body = ir.split(") #0 {", 1)[1]
        direct = row[4] == "direct"
        used = ["KC", "VC", "BLOCK_TABLE", "WS_LSE2", "WS_DSUM", "WS_DQ"]
        used += ["DKC", "DVC"] if direct else ["WS_PDK", "WS_PDV"]
        unused = ["WORKLIST"] + (["WS_PDK", "WS_PDV"] if direct else ["DKC", "DVC"])
        for name in used:
            uses = re.findall(rf"[^\n]*%{name}\b[^\n]*", body)
            assert uses, (row, name)
            for u in uses:
                assert "getelementptr inbounds i8" in u and "i64" in u, u
        for name in unused:
            assert not re.search(rf"%{name}\b", body), (row, name)


# ---------------------------------------------------------------------------
# resource gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", ROWS, ids=_ids)
def test_zero_scratch_no_spills_and_lds_equal_to_the_plan(row):
    """Resource gate of every shipped instance (comgr notes and disassembly):
    no scratch, no spills, register budgets, LDS equal to the phase plan, no
    AGPR copies in the q loop beyond the model."""
    pytest.importorskip("rocke.runtime.comgr")
    from ._attention_bwd_resources import (
        compile_or_skip,
        evaluate_resource_gate,
        find_objdump,
        pin_production_comgr,
    )

    # torch's bundled (production) comgr, pinned by path; torch not imported
    pin_production_comgr()
    if find_objdump() is None:
        pytest.skip("llvm-objdump not found (needed for the q-loop AGPR copy rule)")
    arch = row[0]
    facts = attn_bwd_arch_facts(arch)
    g = resolved_of(*row).general_spec()
    art = compile_or_skip(
        kernel_of(*row), arch=arch, capture_ir_text=False, backend="python"
    )
    res = evaluate_resource_gate(
        art.hsaco, arch=arch, lds_capacity_bytes=facts.lds_capacity_bytes, spec=g
    )
    if not res.valid:
        pytest.skip(
            f"resource gate is valid on llvm22 with the production comgr only "
            f"(flavor {res.flavor}, comgr {res.comgr})"
        )
    assert not res.failures, res.failures
    assert not res.warnings, (res.warnings, res.resources.sgpr)
    plan_bytes = lds_plan(g, arch).total_bytes
    assert res.resources.lds_bytes <= plan_bytes, (res.resources.lds_bytes, plan_bytes)
