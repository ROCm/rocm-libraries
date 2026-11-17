# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``dq_mode = "split"`` outside the dQ kernel (CPU only).

* the spec: ``"split"`` is accepted (``SPLIT_DQ_AVAILABLE``) on the split-dQ
  targets and refused elsewhere; the G4-only knobs are inert under it; the
  register model drops the G4 terms; the LDS plan drops the dS tile and every
  K^T buffer;
* the main kernel with a ``NullSink``: no dQ atomic and no ``WS_DQ`` access,
  the MFMA count per q step drops by exactly the G4 count, one barrier fewer
  per step on the relabel route, no K^T fragments, the same dK / dV code
  otherwise (stage orders keep G1 before G3 or after it), and the knobs that
  are inert under split emit identical IR;
* the plan: the launch list, prep without the dQ zero fill, no dQ workspace,
  the dQ kernel's kernargs, scale placements and grid, the head-group rule,
  the dQ knobs and their refusals, the XCD chunk, the declaration;
* the run glue, the compile cache stage, the tuning registry's dQ axes and
  the sweep driver's handling of split candidates.
"""

from __future__ import annotations

import dataclasses
import re
from functools import cache

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from benchmarks.common import attention_bwd_compile as bc
from benchmarks.common import attention_bwd_sweep as sw
from dispatch.attention_bwd import tuning_common as tc
from kernels.common._attention_bwd_body import STAGE_ORDERS, BwdTileGeometry
from kernels.common._attention_bwd_lds import BwdLdsRequest, plan_bwd_lds
from kernels.common.attention_bwd import (
    INERT_KNOBS,
    SPLIT_DQ_ARCHS,
    SPLIT_DQ_AVAILABLE,
    AttnBwdSpec,
    build_attn_bwd_main,
    lds_plan,
    register_estimates,
    validate_attn_bwd_spec,
)
from kernels.common.attention_bwd_dq import (
    AttnBwdDqSpec,
    validate_attn_bwd_dq_spec,
)
from kernels.common.attention_bwd_plan import (
    DQ_KNOBS,
    FEATURE_GROUPS,
    AttnBwdPolicy,
    AttnBwdRequest,
    attn_bwd_declaration,
    attn_bwd_plan,
    attn_bwd_workspace_bytes,
)
from kernels.common.attention_bwd_run import _BUILDERS, pack_attn_bwd_args

from ._attention_bwd_test_utils import fake_bench_ok
from .sdpa.bwd_kernel_cases import config_knobs, custom_rules

ARCHS = SPLIT_DQ_ARCHS
FULL = AttnBwdPolicy(declared_groups=frozenset(FEATURE_GROUPS), waive_unverified=True)


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


def test_split_is_built_on_the_mfma_targets_only():
    assert SPLIT_DQ_AVAILABLE and ARCHS == ("gfx942", "gfx950")
    s = AttnBwdSpec(head_size=64, dq_mode="split")
    assert s.kernel_name() != AttnBwdSpec(head_size=64).kernel_name()
    for arch in ARCHS:
        assert validate_attn_bwd_spec(s, arch).dq_mode == "split"
    for arch in ("gfx1151", "gfx1201"):
        with pytest.raises(ValueError, match="split dQ kernel is built for"):
            validate_attn_bwd_spec(s, arch)


@pytest.mark.parametrize("arch", ARCHS)
def test_register_model_drops_the_g4_terms(arch):
    for d in (32, 64, 128):
        a = validate_attn_bwd_spec(AttnBwdSpec(head_size=d), arch)
        s = validate_attn_bwd_spec(AttnBwdSpec(head_size=d, dq_mode="split"), arch)
        # pin the AGPR form so both estimates use one placement
        a, s = (dataclasses.replace(x, agpr_alloc=(0, 0)) for x in (a, s))
        wave, w = 64, a.waves
        km, kn = a.block_m, a.block_n
        g4 = 2 * km * int(a.atom_g4.rsplit("x", 1)[1]) // (2 * wave)
        dq = km * d // (wave * w)
        kt = kn * d // (2 * wave * w) if a.kt_source == "reg" else 0
        drop = (
            register_estimates(a, arch)["arch_vgpr"]
            - register_estimates(s, arch)["arch_vgpr"]
        )
        assert drop == g4 + dq + kt, (arch, d)


@pytest.mark.parametrize("arch", ARCHS)
def test_lds_plan_drops_the_ds_tile_and_every_kt_buffer(arch):
    for kv, kt, ts in (
        ("reg", "reg", "lds_plain"),
        ("reg", "lds", "lds_plain"),
        ("lds", "lds", "lds_plain"),
        ("reg", "reg", "xt_lds"),
        ("reg", "lds", "xt_lds"),
    ):
        base = dict(
            head_size=128, k_m0=16, k_n0=128, waves=8, kv_residency=kv,
            kt_source=kt, transpose_source=ts,
        )  # fmt: skip
        atomic = plan_bwd_lds(BwdLdsRequest(**base), arch=arch, strict=False)
        split = plan_bwd_lds(
            BwdLdsRequest(dq_mode="split", **base), arch=arch, strict=False
        )
        assert atomic.has("ds") and not split.has("ds")
        assert not split.has("kt_res") and not split.has("kt_stage")
        # a resident K stays only where K itself is LDS-resident
        assert split.has("k_res") == (kv == "lds")
        assert split.total_bytes <= atomic.total_bytes
    # the LDS route of dS^T keeps the tile (G3 reads it)
    lds_route = BwdLdsRequest(
        head_size=64, k_m0=32, k_n0=128, waves=4, pt_route="lds", dq_mode="split"
    )
    assert plan_bwd_lds(lds_route, arch=arch).has("ds")


@pytest.mark.parametrize("arch", ARCHS)
def test_g4_knobs_are_inert_under_split(arch):
    s = validate_attn_bwd_spec(AttnBwdSpec(head_size=64, dq_mode="split"), arch)
    a = validate_attn_bwd_spec(AttnBwdSpec(head_size=64), arch)
    for k in ("atom_g4", "warp_grid_g4", "kt_source", "dq_issue"):
        assert INERT_KNOBS[k](s, arch), k
    for k in ("atom_g4", "warp_grid_g4", "kt_source"):
        assert not INERT_KNOBS[k](a, arch), k
    # the IR agrees: another G4 atom / grid, K^T source or dQ issue order emits
    # the same split main kernel (kernel name masked)
    ref = _main_ir_masked(s, arch)
    other = dict(
        dq_issue="fragment_regs",
        kt_source="lds" if s.kt_source == "reg" else "reg",
        warp_grid_g4=(2, s.waves // 2),
    )
    if arch == "gfx950":
        other["atom_g4"] = "16x16x16"
    for k, v in other.items():
        t = dataclasses.replace(s, **{k: v})
        try:
            t = validate_attn_bwd_spec(t, arch)
        except ValueError:
            continue  # a coupling of the knob (for example kt_source with LDS K / V)
        assert _main_ir_masked(t, arch) == ref, k


def _main_ir_masked(spec, arch):
    r = validate_attn_bwd_spec(spec, arch)
    text = lower(build_attn_bwd_main(spec, arch=arch), arch=arch, llvm_flavor="llvm22")
    return text.replace(r.kernel_name("main"), "@KERNEL")


# ---------------------------------------------------------------------------
# the split main kernel
# ---------------------------------------------------------------------------


def _walk(ops, ancestors=()):
    for op in ops:
        yield op, ancestors
        for r in op.regions:
            yield from _walk(r.ops, ancestors + (op,))


def _q_loops(kernel):
    loops = [
        op
        for op, _ in _walk(kernel.body.ops)
        if op.name == "scf.for"
        and any(o.name == "tile.mma" for o, _ in _walk(op.regions[0].ops))
    ]
    ids = {id(lp) for lp in loops}
    return [
        lp for lp in loops if not any(id(o) in ids for o, _ in _walk(lp.regions[0].ops))
    ]


def _derived(kernel, names):
    out = {f"%{n}" for n in names}
    for op, _ in _walk(kernel.body.ops):
        if op.name == "tile.global_ptr_add" and op.operands[0].name in out:
            out.add(op.results[0].name)
    return out


MAIN_VARIANTS = {
    "band": dict(mask_class="band"),
    "none_pack": dict(mask_class="none", head_pack=True),
    "band_edge": dict(mask_class="band", edge_tiles=True),
    "band_atomic": dict(mask_class="band", dkv_mode="atomic"),
    "thd_band_atomic_pack": dict(seq_mode="thd", dkv_mode="atomic", head_pack=True),
    "sv1_band": dict(stage_vec=1),
    "lds_route": dict(pt_route="lds"),
    "ldssync_hout": dict(loop_sync="lds_only", step_order="head_outer"),
}
MAIN_ROWS = [
    (arch, cfg, d, var)
    for arch in ARCHS
    for cfg in ("default", "debug")
    for d in (32, 64, 128)
    for var in MAIN_VARIANTS
]


def _main_spec(arch, cfg, d, var, dq_mode):
    # the variant's knobs override the configuration's (the debug
    # configuration already sets some of them, e.g. ``pt_route``)
    knobs = {**config_knobs(cfg, arch), **MAIN_VARIANTS[var]}
    return AttnBwdSpec(head_size=d, dtype="bf16", dq_mode=dq_mode, **knobs)


@cache
def _main_kernel(arch, cfg, d, var, dq_mode):
    return build_attn_bwd_main(_main_spec(arch, cfg, d, var, dq_mode), arch=arch)


@pytest.mark.parametrize("row", MAIN_ROWS, ids=lambda r: "-".join(map(str, r)))
def test_split_main_has_no_g4(row):
    arch = row[0]
    try:
        r_split = validate_attn_bwd_spec(_main_spec(*row, "split"), arch)
        r_atomic = validate_attn_bwd_spec(_main_spec(*row, "atomic"), arch)
    except ValueError as ex:
        pytest.skip(f"not legal: {ex}")
    split, atomic = _main_kernel(*row, "split"), _main_kernel(*row, "atomic")
    geom = BwdTileGeometry.from_spec(r_atomic, arch)
    ws_dq = _derived(split, ("WS_DQ",))
    # no dQ atomic, no WS_DQ use beyond the signature
    for op, _ in _walk(split.body.ops):
        assert not any(getattr(v, "name", None) in ws_dq for v in op.operands), op.name
    ir = lower(split, arch=arch, llvm_flavor="llvm22")
    assert len(re.findall(r"%WS_DQ\b", ir)) == 1
    if r_split.dkv_mode == "direct":
        assert "atomicrmw" not in ir
    loops_s, loops_a = _q_loops(split), _q_loops(atomic)
    assert len(loops_s) == len(loops_a)
    for ls, la in zip(loops_s, loops_a):
        body_s = [op.name for op, _ in _walk(ls.regions[0].ops)]
        body_a = [op.name for op, _ in _walk(la.regions[0].ops)]
        assert body_s.count("tile.mma") == body_a.count("tile.mma") - (
            geom.g4.mma_per_wave
        )
        assert body_s.count("memref.global_atomic_add_f32") == 0
        barriers = ("tile.sync", "tile.sync_lds_only")
        nb_s = sum(body_s.count(x) for x in barriers)
        nb_a = sum(body_a.count(x) for x in barriers)
        # the relabel route drops the barrier that publishes the dS tile and
        # the dS tile's element stores (the LDS route keeps both: G3 reads
        # dS^T from there)
        relabel = r_split.pt_route == "relabel"
        assert nb_s == nb_a - (1 if relabel else 0)
        ds_cells = geom.g0.frags_m * geom.g0.frags_n * geom.g0.op.c_frag_len
        assert body_s.count("tile.smem_store") == body_a.count("tile.smem_store") - (
            ds_cells if relabel else 0
        )
    assert lds_plan(r_split, arch).total_bytes <= lds_plan(r_atomic, arch).total_bytes
    assert not lds_plan(r_split, arch).has("ds") or r_split.pt_route == "lds"


@pytest.mark.parametrize("var", list(MAIN_VARIANTS))
@pytest.mark.parametrize("arch", ARCHS)
def test_split_main_resource_gate(arch, var):
    """The split main kernels pass the resource gate of the main kernel on the
    production flavor and comgr (rule 6 with the split register plan: no dQ
    partial in the AGPR-copy budget)."""
    pytest.importorskip("rocke.runtime.comgr")
    from rocke.helpers.compile import compile_kernel

    from kernels.common.attention_bwd import attn_bwd_arch_facts

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
        pytest.skip("llvm-objdump not found (needed for the q-loop AGPR copy rule)")
    flavor, comgr = current_flavor(), comgr_version()
    if flavor != GATE_FLAVOR or not comgr_is_production(comgr):
        pytest.skip(
            f"resource gate is valid on {GATE_FLAVOR} with the production comgr "
            f"only (flavor {flavor}, comgr {comgr})"
        )
    facts = attn_bwd_arch_facts(arch)
    failures = []
    for row in (r for r in MAIN_ROWS if r[0] == arch and r[-1] == var):
        try:
            r = validate_attn_bwd_spec(_main_spec(*row, "split"), arch)
        except ValueError:
            continue  # not a legal configuration (test_split_main_has_no_g4)
        art = compile_kernel(
            _main_kernel(*row, "split"), arch=arch, capture_ir_text=False,
            backend="python",
        )  # fmt: skip
        res = evaluate_resource_gate(
            art.hsaco, arch=arch, lds_capacity_bytes=facts.lds_capacity_bytes, spec=r
        )
        assert res.valid, (res.flavor, res.comgr)
        if res.failures or res.warnings:
            failures.append((row, res.failures, res.warnings, res.resources.as_dict()))
        if res.resources.lds_bytes > lds_plan(r, arch).total_bytes:
            failures.append((row, "LDS above the plan"))
    assert not failures, failures


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("order", STAGE_ORDERS)
def test_split_stage_orders_keep_g1_and_g3(arch, order):
    """Under split a stage order only places G1 against G3: the orders with
    the same relative G1 / G3 order emit the same kernel."""
    spec = AttnBwdSpec(head_size=64, dtype="bf16", dq_mode="split", stage_order=order)
    want = "g1g3" if order.index("g1") < order.index("g3") else "g3g1"
    ref = AttnBwdSpec(
        head_size=64,
        dtype="bf16",
        dq_mode="split",
        stage_order="g1g3g4" if want == "g1g3" else "g3g1g4",
    )
    assert _main_ir_masked(spec, arch) == _main_ir_masked(ref, arch)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def _req(**kw):
    base = dict(b=2, h_q=8, h_k=2, h_v=2, s_q=100, s_kv=72, d_qk=64, d_v=64)
    base.update(kw)
    return AttnBwdRequest(**base)


def _plan(tuned, arch="gfx942", **req):
    policy = dataclasses.replace(FULL, tuning=lambda *a: dict(tuned))
    return attn_bwd_plan(_req(**req), arch, policy=policy)


SPLIT = {"dq_mode": "split"}


@pytest.mark.parametrize("arch", ARCHS)
def test_split_launches_and_workspace(arch):
    for req, dkv in (({}, "direct"), ({"h_v": 4}, "atomic")):
        plan = _plan(SPLIT, arch, **req)
        stages = [s.stage for s in plan.launches]
        if dkv == "direct":
            assert stages == ["prep", "main", "dq"]
        else:
            assert stages == ["prep", "main", "dq", "convert", "convert"]
            assert [s.pointers["SRC"] for s in plan.launches[3:]] == ["WS_DK", "WS_DV"]
        prep, main, dq = plan.launches[:3]
        assert prep.scalars["zero_dq"] == 0 and prep.pointers["WS_DQ"] is None
        assert main.pointers["WS_DQ"] is None
        assert "WS_DQ" not in plan.workspace_layout
        assert plan.workspace_bytes == attn_bwd_workspace_bytes(
            _req(**req), dq_mode="split"
        )
        spec = plan.specs["dq"]
        assert spec == validate_attn_bwd_dq_spec(plan.specs["main"].dq_spec(), arch)
        assert dq.kernel == spec.kernel_name()
        assert dq.block == spec.waves * 64
        assert dq.grid == (-(-100 // spec.block_m), 8, 2)
        assert dq.pointers["dQ"] == "dq" and dq.pointers["WS_LSE2"] == "WS_LSE2"
        # G: the query heads that share one K head and one V head
        assert dq.scalars["G"] == (4 if dkv == "direct" else 2)
        assert plan.runtime["dq_mode"] == "split"
        # the problem fields of the dQ kernel follow the main spec
        m = plan.specs["main"]
        for f in ("head_size", "dtype", "seq_mode", "stage_vec", "mask_class"):
            assert getattr(spec, f) == getattr(m, f)
    # an atomic-dQ plan is unchanged: no dq launch, no dq spec, no dq runtime
    plain = _plan({}, arch)
    assert [s.stage for s in plain.launches] == ["prep", "main", "convert"]
    assert "dq" not in plain.specs and "dq_mode" not in plain.runtime


@pytest.mark.parametrize("placement", ("fold_ds", "convert", "dq_before_atomic"))
def test_split_scale_placements(placement):
    plan = _plan({**SPLIT, "scale_placement": placement})
    scale = 1.0 / 8.0
    dq = plan.launches[2]
    want = {
        "fold_ds": (scale, 1.0, "fold_ds"),
        "convert": (1.0, scale, "convert"),
        "dq_before_atomic": (1.0, scale, "convert"),
    }[placement]
    assert (dq.scalars["ds_mult"], dq.scalars["dq_mult"]) == pytest.approx(want[:2])
    assert plan.specs["dq"].static_scale == want[2]
    pinned = _plan({**SPLIT, "scale_placement": placement, "static_scale": "runtime"})
    assert pinned.specs["dq"].static_scale == "runtime"


def test_dq_knobs_are_routed_checked_and_refused():
    plan = _plan(
        {
            **SPLIT,
            "dq_head_loop": "group",
            "dq_edge_tiles": True,
            "dq_block_n": 32,
            "dq_kv_pad": 16,
            "dq_stats_prescale": False,
        }
    )
    s = plan.specs["dq"]
    assert (s.head_loop, s.edge_tiles, s.block_n, s.kv_pad, s.stats_prescale) == (
        "group",
        True,
        32,
        16,
        False,
    )
    # head_loop = "group": G = 4 heads per CTA, kM0 / G rows each
    assert plan.launches[2].grid == (-(-100 // (s.block_m // 4)), 2, 2)
    with pytest.raises(ValueError, match="need dq_mode='split'"):
        _plan({"dq_edge_tiles": True})
    with pytest.raises(ValueError, match="unknown knobs"):
        _plan({**SPLIT, "dq_ring": 2})
    with pytest.raises(ValueError, match="divide"):
        _plan({**SPLIT, "dq_head_loop": "group"}, h_q=6, h_k=2, h_v=2)  # G = 3
    with pytest.raises(ValueError, match="split dQ kernel is built for"):
        _plan(SPLIT, "gfx1151")
    with pytest.raises(ValueError):
        _plan({**SPLIT, "dq_atom_g02": "16x16x32"}, "gfx942")


def test_dq_xcd_chunk():
    base = _plan(SPLIT, num_xcds=8)
    gx = base.launches[2].grid[0]
    for value, want in ((0, 0), ("gx", gx), ("2gx", 2 * gx), (64, 64)):
        plan = _plan({**SPLIT, "dq_xcd_chunk": value}, num_xcds=8)
        dq = plan.launches[2]
        assert (dq.scalars["xcd_n"], dq.scalars["xcd_chunk"]) == (
            (8, want) if want else (0, 0)
        )
        assert plan.specs["dq"].xcd_remap is bool(want)
        assert plan.runtime["dq_xcd_chunk"] == want
        # the main kernel keeps its own XCD decision
        assert plan.specs["main"].xcd_remap is False
    with pytest.raises(ValueError, match="num_xcds"):
        _plan({**SPLIT, "dq_xcd_chunk": "gx"})
    with pytest.raises(ValueError, match="dq_xcd_chunk"):
        _plan({**SPLIT, "dq_xcd_chunk": 3}, num_xcds=8)


def test_split_with_head_packing_and_thd():
    # short query spans: the main kernel packs the group's heads, dQ is the
    # dQ kernel's
    plan = _plan(SPLIT, s_q=1, s_kv=40)
    assert plan.specs["main"].head_pack and plan.runtime["pack_heads"] == 4
    assert [s.stage for s in plan.launches] == ["prep", "main", "dq"]
    thd = _plan(SPLIT, layout="thd", max_total_q=210, max_total_kv=150)
    dq = thd.launches[2]
    assert thd.specs["dq"].seq_mode == "thd"
    assert dq.pointers["OFF_Q"] == "offsets_q" and dq.pointers["OFF_KV"] == "offsets_kv"
    assert dq.scalars["ws_seg"] == 0 and dq.scalars["ws_rows_q"] == 210


def test_declaration_states_the_dq_determinism_per_mode():
    d = attn_bwd_declaration("gfx942", policy=FULL)
    text = d["determinism"]["dq"]
    assert "dq_mode='atomic'" in text and "dq_mode='split'" in text
    assert "deterministic" in text.split("dq_mode='split'")[0].rsplit(";", 1)[1]


def test_custom_configuration_accepts_the_dq_knobs():
    rules = custom_rules('[{"knobs": {"dq_mode": "split", "dq_edge_tiles": true}}]')
    assert rules[0][1] == {"dq_mode": "split", "dq_edge_tiles": True}
    with pytest.raises(ValueError, match="not knobs"):
        custom_rules('[{"knobs": {"dq_nonsense": 1}}]')


# ---------------------------------------------------------------------------
# run glue, compile cache
# ---------------------------------------------------------------------------


def test_run_glue_builds_and_packs_the_dq_launch():
    from kernels.common.attention_bwd_dq import build_attn_bwd_dq

    assert _BUILDERS["dq"] is build_attn_bwd_dq
    plan = _plan(SPLIT)
    roles = {
        r: 0x1000 * (i + 1) for i, r in enumerate("q k v o do lse dq dk dv".split())
    }
    dq = plan.launches[2]
    blob = pack_attn_bwd_args(
        dq, tensors=roles, workspace=1 << 20, workspace_layout=plan.workspace_layout
    )
    from kernels.common.attention_bwd import attn_bwd_params

    sizes = {"Q": 8, "q": 8, "f": 4, "i": 4}
    assert len(blob) == sum(sizes[f] for _n, f in attn_bwd_params("dq"))
    assert blob[: 8 * 5] == b"".join(
        roles[r].to_bytes(8, "little") for r in ("q", "k", "v", "do", "dq")
    )


def test_compile_cache_knows_the_dq_stage():
    assert "dq" in bc.STAGES
    from kernels.common.attention_bwd_dq import build_attn_bwd_dq

    assert bc._builder("dq") is build_attn_bwd_dq
    spec = validate_attn_bwd_dq_spec(AttnBwdDqSpec(head_size=64), "gfx942")
    assert bc.spec_identity("dq", spec).startswith("dq|AttnBwdDqSpec|")


# ---------------------------------------------------------------------------
# registry and sweep
# ---------------------------------------------------------------------------


def test_registry_dq_axes_are_the_plan_dq_knobs_and_legal():
    dq_axes = [a for a in tc.BWD_KNOB_AXES if a.kind == tc.KIND_DQ]
    assert {a.name for a in dq_axes} == set(DQ_KNOBS)
    assert tc.backward_axis("dq_mode").not_built == ()
    # couplings a value needs (its tile or transpose read)
    contexts = {
        ("waves", 8): {"block_m": 128},
        ("block_m", 16): {"waves": 1},
        ("block_m", 32): {"waves": 2},
        ("atom_g4", "16x16x32"): {"transpose_source": "lds_plain"},
    }
    for a in dq_axes:
        assert a.family == 8 and a.level == "production"
        assert not a.applies(a.choices[1], {"dq_mode": "atomic"})
        assert a.applies(a.choices[1], {"dq_mode": "split"})
        if a.name == "dq_xcd_chunk":
            assert a.stage == "runtime"
            continue  # a plan value (test_dq_xcd_chunk)
        assert a.stage == "structure" and a.name in sw.DQ_EXTRA
        field = a.name[3:]
        for v in a.choices:
            kw = {field: v, **contexts.get((field, v), {})}
            legal = []
            for arch in ARCHS:
                for d in (64, 128):
                    try:
                        validate_attn_bwd_dq_spec(
                            AttnBwdDqSpec(head_size=d, **kw), arch
                        )
                    except ValueError:
                        continue
                    legal.append((arch, d))
            assert legal, (a.name, v)


def _cand(**knobs):
    sl = sw.primary_slice("gfx942", "general", 64)
    return sw.Candidate.make(sl, {**sw.default_knobs(sl), **knobs})


def test_sweep_candidates_carry_the_dq_kernel():
    plain = _cand()
    split = _cand(dq_mode="split")
    edge = _cand(dq_mode="split", dq_edge_tiles=True)
    for c in (plain, split, edge):
        assert sw.legality(c).ok, sw.legality(c).detail
    assert sw.legality(plain).dq is None
    assert sw.legality(split).dq == validate_attn_bwd_dq_spec(
        AttnBwdDqSpec(head_size=64, dtype="bf16"), "gfx942"
    )
    assert edge.spec() == split.spec()  # dQ knobs are not main spec fields
    assert edge.dq_knobs() == {"edge_tiles": True}
    keys = {sw.effective_key(c) for c in (plain, split, edge)}
    assert len(keys) == 3
    # a dQ knob at the dQ kernel's own resolved value is the same configuration
    same = _cand(dq_mode="split", dq_waves=sw.legality(split).dq.waves)
    assert sw.effective_key(same) == sw.effective_key(split)
    bad = _cand(dq_edge_tiles=True)
    assert not sw.legality(bad).ok and sw.legality(bad).rule == "coupling"
    illegal = _cand(dq_mode="split", dq_atom_g02="16x16x32")
    assert not sw.legality(illegal).ok
    assert sw.legality(illegal).rule.startswith("dq_")
    # the plan sees the same dQ knobs
    plan = _plan({k: v for k, v in edge.tuning_knobs().items()}, d_qk=64, d_v=64)
    assert plan.specs["dq"].edge_tiles


def test_structure_stage_varies_the_dq_knobs_on_split_points():
    c = _cand()
    rel = sw.Release(sw.ALL_RELEASED, "production")
    pts = sw.structures(c, rel)
    split = [p for p in pts if p.get("dq_mode") == "split"]
    atomic = [p for p in pts if p.get("dq_mode", "atomic") != "split"]
    assert split and atomic
    # atomic points carry no dQ knob; split points vary one dQ knob at a time,
    # except a wave count paired with its 16 * waves row tile
    assert not any(k in p for p in atomic for k in DQ_KNOBS)
    pair = {"dq_waves", "dq_block_m"}
    for p in split:
        got = {k for k in DQ_KNOBS if k in p}
        assert len(got) <= 1 or (got == pair and p["dq_block_m"] == 16 * p["dq_waves"])
    dq = sw.dq_values(c, rel)
    assert dq[0] == {} and len(dq) == len({repr(sorted(v.items())) for v in dq})
    singles = [v for v in dq if len(v) <= 1]
    for n in sw.DQ_EXTRA:
        axis = tc.backward_axis(n)
        got = {v[n] for v in singles if n in v}
        assert got == set(axis.values("gfx942", "production")) - {axis.default}, n
    rows = set(tc.backward_axis("dq_block_m").values("gfx942", "production"))
    pairs = [v for v in dq if len(v) > 1]
    assert pairs == [
        {"dq_waves": w, "dq_block_m": 16 * w}
        for w in tc.backward_axis("dq_waves").values("gfx942", "production")
        if w is not None and 16 * w in rows
    ]
    # eight waves are legal only on the paired 128-row tile at the d64 start
    sl = c.slice
    lg = {
        repr(sorted(v.items())): sw.legality(
            sw.Candidate.make(sl, {**c.k, "dq_mode": "split", **v})
        )
        for v in dq
        if v.get("dq_waves") == 8
    }
    assert not lg[repr([("dq_waves", 8)])].ok
    assert lg[repr([("dq_block_m", 128), ("dq_waves", 8)])].ok
    # an unreleased family 8 adds nothing
    assert sw.dq_values(c, sw.Release()) == [{}]
    assert not any("dq_mode" in p for p in sw.structures(c, sw.Release()))


def test_runtime_stage_adds_the_dq_xcd_chunk_for_split_candidates():
    sl = sw.primary_slice("gfx942", "general", 64)
    split = sw.Candidate.make(sl, {**sw.default_knobs(sl), "dq_mode": "split"})
    plain = sw.Candidate.make(sl, sw.default_knobs(sl))
    rel = sw.Release(sw.ALL_RELEASED, "production")
    cells = sw.representative_cells(sl)
    got = sw.stage_candidates(
        "runtime", sl, rel=rel, prior={"schedule": [split, plain]}, cells=cells
    )
    dq_x = [c for c in got if "dq_xcd_chunk" in c.r]
    assert dq_x and all(c.k.get("dq_mode") == "split" for c in dq_x)
    assert {c.r["dq_xcd_chunk"] for c in dq_x} == {"gx", "2gx", "4gx", 64}
    assert all(c.r["num_xcds"] == sw.SWEEP_NUM_XCDS["gfx942"] for c in dq_x)
    # not released: nothing; RDNA (no split dQ, no XCD hint): nothing
    assert sw._dq_xcd_values(sl, sw.Release()) == []
    rdna = sw.primary_slice("gfx1151", "general", 64)
    assert sw._dq_xcd_values(rdna, rel) == []


def test_compile_stage_compiles_and_gates_the_dq_kernel_of_split_candidates(
    tmp_path, monkeypatch
):
    """A split candidate's dQ kernel compiles with its main kernel and is held
    to the dQ resource rules (``static_prune_dq``): a failing dQ kernel prunes
    the candidate (``pruned:dq_<rule>``), a dQ compile error records
    ``dq_<error>``; an atomic candidate compiles its main kernel only."""
    from benchmarks.common import attention_bwd_bench as bb

    sl = sw.primary_slice("gfx942", "general", 64)
    plain = sw.Candidate.make(sl, sw.default_knobs(sl))
    split = plain.with_knobs(dq_mode="split")
    pruned = split.with_knobs(dq_edge_tiles=True)
    broken = split.with_knobs(dq_kv_pad=16)
    group = [plain, split, pruned, broken]
    assert all(sw.legality(c).ok for c in group)
    monkeypatch.setattr(sw, "stage_candidates", lambda *a, **k: list(group))

    class Entry:
        def __init__(self, name):
            self.kernel_name, self.hsaco_sha, self.key = name, "sha", "key"
            self.hsaco = b"\x7fELF"

    class Res:
        def __init__(self, stage, spec):
            self.ok = not (stage == "dq" and spec.kv_pad == 16)
            self.error = None if self.ok else "compile_error"
            self.detail, self.cached = "", False
            self.entry = Entry(spec.kernel_name()) if self.ok else None

    compiled = []

    def fake_batch(items, **kw):
        compiled.extend(items)
        return [Res(st, s) for st, s in items]

    class Gate:
        def static_prune(self, arch, spec, hsaco):
            return None, {"main": 1}

        def static_prune_dq(self, arch, spec, hsaco):
            return ("sgpr_spill" if spec.edge_tiles else None), {"dq": 1}

    fake_bench = fake_bench_ok(bb)

    monkeypatch.setattr(bc, "compile_batch", fake_batch)
    monkeypatch.setattr(bb, "main", fake_bench)

    class Toolchain:
        def record(self):
            return {"comgr": "c"}

    drv = sw.Driver(sw.Options(out=tmp_path, arch="gfx942", release=sw.Release(sw.ALL_RELEASED), gate="x:y"))  # fmt: skip
    drv._gate, drv._toolchain = Gate(), Toolchain()
    drv._cache = type("C", (), {"root": tmp_path / "cache"})()
    counts = drv.run_stage(sl, "geometry")
    stages = [st for st, _ in compiled]
    assert stages.count("main") == 4 and stages.count("dq") == 3
    assert {s.kernel_name() for st, s in compiled if st == "dq"} == {
        sw.legality(c).dq.kernel_name() for c in (split, pruned, broken)
    }
    recs = sw.read_records(tmp_path, sl, "geometry")
    assert recs[pruned.key()]["status"] == "pruned:dq_sgpr_spill"
    assert recs[pruned.key()]["dq_resources"] == {"dq": 1}
    assert recs[broken.key()]["status"] == "dq_compile_error"
    for c in (plain, split):
        assert recs[c.key()]["status"] == "timed"
        assert recs[c.key()]["resources"] == {"main": 1}
    assert recs[split.key()]["dq_kernel"] == sw.legality(split).dq.kernel_name()
    assert "dq_kernel" not in recs[plain.key()]
    assert counts["pruned"] == 1 and counts["compile_fail"] == 1
