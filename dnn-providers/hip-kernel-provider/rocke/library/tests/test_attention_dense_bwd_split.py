# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``dq_mode = "split"`` on the dense route (CPU only).

* the dense main kernel under split: built (no longer a not-built value), a
  ``NullSink`` (no dQ atomic, no ``WS_DQ`` use beyond the signature, the MFMA
  count per q step down by exactly the G4 count), with the prefetch ring and
  atomic dK / dV;
* the dense dQ kernel (:mod:`kernels.common.attention_dense_bwd_dq`): its
  spec (problem fields, the dQ spec's problem fields, kernel names salted by
  every configuration value), validation, the ``rocke.attn_dense_bwd_dq.v1``
  signature, no atomics, buffer resources only under ``buffer_oob``, the key
  loops the compile-time classification keeps, and that classification
  against a brute-force cell rule over small shapes and every mask form;
* the plan: ``prep -> dense -> dense_dq [-> convert x2]``, prep without the
  dQ zero fill, the general split plan's workspace and dQ multipliers, the dQ
  grid, the ``DQ_KNOBS`` routing and refusals, the dQ XCD chunk, a given
  spec with the dQ values in ``runtime``;
* the run glue, the compile cache stage, the dense registry's ``dq_mode``
  and dQ axes, and the sweep's dense split points, legality and compile stage.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
import struct

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from benchmarks.common import attention_bwd_compile as bc
from kernels.common._attention_bwd_body import BwdTileGeometry
from kernels.common.attention_bwd_dq import AttnBwdDqSpec, validate_attn_bwd_dq_spec
from kernels.common.attention_bwd_plan import DQ_KNOBS, attn_bwd_plan
from kernels.common.attention_dense_bwd_dq import (
    ATTN_DENSE_BWD_DQ_ABI,
    AttentionDenseBwdDqSpec,
    attn_dense_bwd_dq_params,
    build_attention_dense_bwd_dq,
    dense_dq_static_classes,
    validate_attn_dense_bwd_dq_spec,
)
from kernels.common.attention_dense_bwd_plan import (
    attn_dense_bwd_plan,
    dense_bwd_spec_for,
)
from kernels.common.attention_dense_bwd_spec import (
    dense_bwd_not_built_reasons,
    validate_attn_dense_bwd_spec,
)

from ._attention_bwd_test_utils import kernel_signature
from .sdpa.bwd_kernel_cases import config_policy
from .test_attention_dense_bwd import _BUILD, ARCHS, _req, _spec

SPLIT = {"dq_mode": "split"}


def _walk(ops):
    for op in ops:
        yield op
        for r in op.regions:
            yield from _walk(r.ops)


def _loops(kernel):
    """Innermost ``scf.for`` loops that hold a matrix op."""
    loops = [
        op
        for op in _walk(kernel.body.ops)
        if op.name == "scf.for"
        and any(o.name == "tile.mma" for o in _walk(op.regions[0].ops))
    ]
    ids = {id(lp) for lp in loops}
    return [
        lp
        for lp in loops
        if not any(id(o) in ids for o in _walk(lp.regions[0].ops) if o is not lp)
    ]


def _count(ops, name):
    return sum(1 for op in _walk(ops) if op.name == name)


# ---------------------------------------------------------------------------
# the dense main kernel under split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_split_is_built_on_the_dense_route(arch):
    spec = _spec(arch, dq_mode="split")
    assert dense_bwd_not_built_reasons(spec) == []
    assert spec.kernel_name() != _spec(arch).kernel_name()
    kernel = _BUILD[arch](spec, arch=arch)
    assert kernel.name == spec.kernel_name()


_MAIN_SPLIT = (
    ("plain", {}),
    ("unmasked_d64", dict(causal=False, head_size=64)),
    ("ring_vgpr2", dict(body_knobs={"ring_depth": 2})),
    ("atomic_dkv", dict(dkv_mode="atomic")),
)


@pytest.mark.parametrize("name,over", _MAIN_SPLIT, ids=[n for n, _ in _MAIN_SPLIT])
@pytest.mark.parametrize("arch", ARCHS)
def test_split_main_has_no_g4(arch, name, over):
    split = _spec(arch, dq_mode="split", **over)
    atomic = _spec(arch, **over)
    r_split = validate_attn_dense_bwd_spec(split, arch)
    r_atomic = validate_attn_dense_bwd_spec(atomic, arch)
    ks, ka = _BUILD[arch](split, arch=arch), _BUILD[arch](atomic, arch=arch)
    ir = lower(ks, arch=arch, llvm_flavor="llvm22")
    # the WS_DQ kernarg is in the signature only
    assert len(re.findall(r"%WS_DQ\b", ir)) == 1
    if split.dkv_mode == "direct":
        assert "atomicrmw" not in ir
    geom = BwdTileGeometry.from_spec(r_atomic, arch)
    ls, la = _loops(ks), _loops(ka)
    assert len(ls) == len(la) > 0
    for s_loop, a_loop in zip(ls, la):
        ns = _count(s_loop.regions[0].ops, "tile.mma")
        na = _count(a_loop.regions[0].ops, "tile.mma")
        assert ns == na - geom.g4.mma_per_wave
        assert _count(s_loop.regions[0].ops, "memref.global_atomic_add_f32") == 0
    assert r_split.dq_mode == "split" and r_atomic.dq_mode == "atomic"


# ---------------------------------------------------------------------------
# the dense dQ kernel: spec
# ---------------------------------------------------------------------------

_PROBLEM = dict(
    seqlen_q=256, seqlen_kv=256, num_query_heads=8, num_kv_heads=2, head_size=128
)


def _dq(**kw):
    return AttentionDenseBwdDqSpec(**{**_PROBLEM, **kw})


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_dq_iglp1_marks_a_fresh_compile_process(arch):
    """The dense dQ kernel runs the general key step: ``sched = "iglp1"`` of
    its dQ values marks it to compile in a fresh process (``iglp_opt(1)``
    keeps LLVM state across the compiles of a process); other values do not."""
    from rocke.helpers.compile import COMGR_FRESH_PROCESS_ATTR

    for sched in ("none", "iglp0", "iglp1", "stage_table"):
        dq = AttnBwdDqSpec(head_size=128, dtype="bf16", sched=sched)
        k = build_attention_dense_bwd_dq(_dq(dq=dq), arch=arch)
        got = k.attrs.get(COMGR_FRESH_PROCESS_ATTR, False)
        assert got is (sched == "iglp1"), sched


def test_dq_spec_required_fields_and_defaults():
    required = sorted(
        f.name
        for f in dataclasses.fields(AttentionDenseBwdDqSpec)
        if f.default is dataclasses.MISSING
    )
    assert required == sorted(_PROBLEM)
    s = _dq()
    assert s.dq == AttnBwdDqSpec(
        head_size=128, dtype="bf16", mask_class="band", ws_layout="head_major"
    )
    assert _dq(causal=False).dq.mask_class == "none"
    assert s.group == 4 and s.masked
    assert s.kernel_name() == (
        "rocke_attn_dense_bwd_dq_bf16_d128_bshd_sq256_sk256_hq8_hk2_causaltl"
    )
    assert _dq(batch=2).kernel_name().endswith("_causaltl_b2")


def test_dq_spec_refuses_a_mismatched_dq_spec():
    for bad in (
        AttnBwdDqSpec(head_size=64, dtype="bf16"),
        AttnBwdDqSpec(head_size=128, dtype="fp16"),
        AttnBwdDqSpec(head_size=128, dtype="bf16", mask_class="none"),
        AttnBwdDqSpec(head_size=128, dtype="bf16", seq_mode="thd"),
        AttnBwdDqSpec(head_size=128, dtype="bf16", stage_vec=1),
        AttnBwdDqSpec(head_size=128, dtype="bf16", ws_layout="token_major"),
    ):
        with pytest.raises(ValueError, match="does not match"):
            _dq(dq=bad)
    with pytest.raises(ValueError):
        _dq(num_query_heads=6, num_kv_heads=4)  # not a head multiple
    with pytest.raises(ValueError):
        _dq(oob="wrap")


def test_dq_kernel_names_salt_every_configuration_value():
    base = _dq().kernel_name()
    names = {base}
    for kw in (
        dict(oob="clamp"),
        dict(ws_layout="token_major", dq=None),
        dict(dq=AttnBwdDqSpec(head_size=128, dtype="bf16", edge_tiles=True)),
        dict(dq=AttnBwdDqSpec(head_size=128, dtype="bf16", block_m=128)),
        dict(dq=AttnBwdDqSpec(head_size=128, dtype="bf16", xcd_remap=True)),
        dict(dq=AttnBwdDqSpec(head_size=128, dtype="bf16", static_scale="convert")),
    ):
        n = _dq(**kw).kernel_name()
        assert n not in names, kw
        names.add(n)
    # the problem tokens stay readable
    assert (
        _dq(layout="bhsd", causal=False)
        .kernel_name()
        .endswith("_bhsd_sq256_sk256_hq8_hk2_none")
    )


@pytest.mark.parametrize("arch", ARCHS)
def test_dq_spec_of_a_dense_main_spec(arch):
    main = _spec(arch, dq_mode="split", oob="clamp", ws_layout="token_major")
    body = validate_attn_dense_bwd_spec(main, arch)
    s = AttentionDenseBwdDqSpec.of(main, body.dq_spec(edge_tiles=True))
    for f in _PROBLEM:
        assert getattr(s, f) == getattr(main, f)
    assert (s.oob, s.ws_layout, s.dq.edge_tiles) == ("clamp", "token_major", True)
    assert s.dq.static_scale == "fold_ds"


@pytest.mark.parametrize("arch", ARCHS)
def test_dq_validation(arch):
    r = validate_attn_dense_bwd_dq_spec(_dq(), arch)
    assert r == validate_attn_bwd_dq_spec(_dq().dq, arch)
    with pytest.raises(ValueError, match="arch"):
        validate_attn_dense_bwd_dq_spec(_dq(), "gfx1151")
    group = AttnBwdDqSpec(head_size=128, dtype="bf16", head_loop="group")
    validate_attn_dense_bwd_dq_spec(_dq(dq=group), arch)  # G = 4 | 64
    with pytest.raises(ValueError, match="divide"):
        validate_attn_dense_bwd_dq_spec(
            _dq(dq=group, num_query_heads=6, num_kv_heads=2), arch
        )  # G = 3
    with pytest.raises(ValueError, match="32-bit"):
        validate_attn_dense_bwd_dq_spec(
            _dq(seqlen_q=1 << 20, seqlen_kv=1 << 20, num_query_heads=64), arch
        )


# ---------------------------------------------------------------------------
# the dense dQ kernel: IR
# ---------------------------------------------------------------------------

_READONLY = {"Q", "K", "V", "dO", "WS_LSE2", "WS_DSUM"}


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize(
    "kw",
    [
        {},
        dict(oob="clamp"),
        dict(dq=AttnBwdDqSpec(head_size=128, dtype="bf16", xcd_remap=True)),
        dict(dq=AttnBwdDqSpec(head_size=128, dtype="bf16", head_loop="group")),
    ],
)
def test_dq_signature_is_the_abi_and_no_atomics(arch, kw):
    spec = _dq(**kw)
    ir = lower(
        build_attention_dense_bwd_dq(spec, arch=arch), arch=arch, llvm_flavor="llvm22"
    )
    sig = kernel_signature(ir)
    assert [(n, k) for n, k, _ in sig] == list(attn_dense_bwd_dq_params())
    assert ATTN_DENSE_BWD_DQ_ABI == "rocke.attn_dense_bwd_dq.v1"
    for name, kind, text in sig:
        if kind == "Q":
            assert ("readonly" in text) == (name in _READONLY), text
    assert "atomicrmw" not in ir
    assert spec.kernel_name() in ir


@pytest.mark.parametrize("arch", ARCHS)
def test_dq_buffer_resources_only_under_buffer_oob(arch):
    def n_rsrc(**kw):
        k = build_attention_dense_bwd_dq(_dq(**kw), arch=arch)
        return _count(k.body.ops, "tile.buffer_rsrc")

    assert n_rsrc() > 0
    assert n_rsrc(oob="clamp") == 0
    # a head-packed CTA reads Q / dO through the global pointer: K / V only
    group = AttnBwdDqSpec(head_size=128, dtype="bf16", head_loop="group")
    assert 0 < n_rsrc(dq=group) < n_rsrc()


def _key_loops(**kw):
    return len(_loops(build_attention_dense_bwd_dq(_dq(**kw), arch="gfx950")))


def test_dq_key_loops_follow_the_static_classification():
    edge = AttnBwdDqSpec(head_size=128, dtype="bf16", edge_tiles=True)
    edge_none = AttnBwdDqSpec(
        head_size=128, dtype="bf16", edge_tiles=True, mask_class="none"
    )
    # unmasked aligned lengths: every visited tile is interior, one loop
    assert _key_loops(causal=False, dq=edge_none) == 1
    # causal: edge, interior and edge loops
    assert _key_loops(dq=edge) == 3
    # no edge_tiles: the single edge loop of the general kernel
    assert _key_loops() == 1
    assert _key_loops(causal=False) == 1
    # an unmasked ragged tail: edge loops come back
    assert _key_loops(causal=False, seqlen_kv=200, dq=edge_none) == 3
    # every visited tile an edge tile (a narrow band): one edge loop
    band = AttnBwdDqSpec(head_size=128, dtype="bf16", edge_tiles=True)
    assert _key_loops(causal=False, window_left=3, window_right=3, dq=band) == 1


def _brute_classes(s_q, s_kv, rows, kn, left, right, br):
    off = s_kv - s_q if br else 0

    def keep(q, k):
        if q >= s_q or k >= s_kv:
            return False
        lo = -(1 << 30) if left == -1 else q + off - left
        hi = (1 << 30) if right == -1 else q + off + right
        return lo <= k <= hi

    any_edge = any_interior = False
    for q0 in range(0, s_q, rows):
        for k0 in range(0, s_kv, kn):
            cells = [
                keep(q, k) for q in range(q0, q0 + rows) for k in range(k0, k0 + kn)
            ]
            if not any(cells):
                continue
            if all(cells):
                any_interior = True
            else:
                any_edge = True
    return any_edge, any_interior


_MASKS = [
    (-1, -1, False),
    (-1, 0, False),
    (-1, 0, True),
    (15, 0, False),
    (31, 0, True),
    (7, 3, False),
    (9, -1, True),
    (-1, 5, False),
    (0, 0, False),
]


@pytest.mark.parametrize("mask", _MASKS)
def test_static_classes_match_a_brute_force_cell_rule(mask):
    left, right, br = mask
    masked = not (left == -1 and right == -1)
    for rows, kn in ((16, 16), (16, 64), (64, 32), (32, 128)):
        for s_q, s_kv in itertools.product((1, 16, 33, 64, 100), (16, 47, 64, 130)):
            got = dense_dq_static_classes(
                s_q=s_q, s_kv=s_kv, rows=rows, k_n0=kn, left=left, right=right,
                bottom_right=br, masked=masked,
            )  # fmt: skip
            assert got == _brute_classes(s_q, s_kv, rows, kn, left, right, br), (
                s_q, s_kv, rows, kn,
            )  # fmt: skip


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def _plan(arch, knobs=None, **req):
    pol = config_policy("default", arch, {**SPLIT, "head_pack": False, **(knobs or {})})
    return attn_dense_bwd_plan(_req(causal=True, **req), arch, policy=pol)


def _general(arch, knobs=None, **req):
    pol = config_policy("default", arch, {**SPLIT, "head_pack": False, **(knobs or {})})
    return attn_bwd_plan(_req(causal=True, **req), arch, policy=pol)


@pytest.mark.parametrize("arch", ARCHS)
def test_split_launches_workspace_and_multipliers(arch):
    for knobs, dkv in (({}, "direct"), ({"g_split": 2}, "atomic")):
        plan, general = _plan(arch, knobs), _general(arch, knobs)
        stages = [s.stage for s in plan.launches]
        if dkv == "direct":
            assert stages == ["prep", "dense", "dense_dq"]
        else:
            assert stages == ["prep", "dense", "dense_dq", "convert", "convert"]
            assert [s.pointers["SRC"] for s in plan.launches[3:]] == ["WS_DK", "WS_DV"]
        prep, dense, dq = plan.launches[:3]
        assert prep == general.launches[0]
        assert prep.scalars["zero_dq"] == 0 and prep.pointers["WS_DQ"] is None
        assert dense.pointers["WS_DQ"] is None
        assert "WS_DQ" not in plan.workspace_layout
        assert plan.workspace_bytes == general.workspace_bytes
        assert dict(plan.workspace_layout) == dict(general.workspace_layout)
        spec = plan.specs["dense_dq"]
        assert "dq" not in plan.specs and "main" not in plan.specs
        assert isinstance(spec, AttentionDenseBwdDqSpec)
        assert dq.kernel == spec.kernel_name() and dq.block == spec.dq.waves * 64
        gdq = next(s for s in general.launches if s.stage == "dq")
        for k in ("scale_log2", "ds_mult", "dq_mult"):
            assert dq.scalars[k] == gdq.scalars[k], k
        # the general dQ kernel's grid: the same CTAs
        assert dq.grid == gdq.grid
        assert spec.dq == general.specs["dq"]
        assert dq.scalars["batch"] == 2 and dq.scalars["xcd_n"] == 0
        assert sorted(list(dq.pointers) + list(dq.scalars)) == sorted(
            n for n, _ in attn_dense_bwd_dq_params()
        )
        assert plan.runtime["dq_mode"] == "split"
        assert plan.specs["dense"].dq_mode == "split"
        assert plan.specs["dense"].dkv_mode == dkv


@pytest.mark.parametrize("placement", ("fold_ds", "convert", "dq_before_atomic"))
def test_split_scale_placements(placement):
    arch = "gfx950"
    knobs = {"scale_placement": placement}
    plan, general = _plan(arch, knobs), _general(arch, knobs)
    dq = plan.launches[2]
    gdq = next(s for s in general.launches if s.stage == "dq")
    assert dict(dq.scalars, batch=0, xcd_n=0, xcd_chunk=0) == dict(
        {k: gdq.scalars[k] for k in ("scale_log2", "ds_mult", "dq_mult")},
        batch=0, xcd_n=0, xcd_chunk=0,
    )  # fmt: skip
    assert plan.specs["dense_dq"].dq.static_scale == general.specs["dq"].static_scale


@pytest.mark.parametrize("arch", ARCHS)
def test_dq_knobs_are_routed_checked_and_refused(arch):
    plan = _plan(
        arch,
        {"dq_head_loop": "group", "dq_edge_tiles": True, "dq_block_m": 128,
         "dq_kv_prefetch": True},
    )  # fmt: skip
    s = plan.specs["dense_dq"].dq
    assert (s.head_loop, s.edge_tiles, s.block_m, s.kv_prefetch) == (
        "group", True, 128, True,
    )  # fmt: skip
    # G = 4 heads per CTA, kM0 / G rows each
    assert plan.launches[2].grid == (256 // (128 // 4), 2, 2)
    with pytest.raises(ValueError, match="need dq_mode='split'"):
        attn_dense_bwd_plan(
            _req(causal=True), arch,
            policy=config_policy("default", arch, {"dq_edge_tiles": True}),
        )  # fmt: skip
    with pytest.raises(ValueError, match="divide"):
        _plan(arch, {"dq_head_loop": "group"}, h_q=6, h_k=2, h_v=2)  # G = 3
    with pytest.raises(ValueError, match="num_xcds"):
        _plan(arch, {"dq_xcd_chunk": 64})
    plan = _plan(arch, {"dq_xcd_chunk": "gx"}, num_xcds=8)
    dq = plan.launches[2]
    assert dq.scalars["xcd_n"] == 8 and dq.scalars["xcd_chunk"] == dq.grid[0]
    assert plan.specs["dense_dq"].dq.xcd_remap
    assert plan.runtime["dq_xcd_n"] == 8


@pytest.mark.parametrize("arch", ARCHS)
def test_given_split_spec_takes_the_dq_values_in_runtime(arch):
    req = _req(causal=True)
    spec = dense_bwd_spec_for(req, arch, SPLIT)
    pol = config_policy("default", arch)
    plan = attn_dense_bwd_plan(
        req, arch, policy=pol, spec=spec, runtime={"dq_edge_tiles": True}
    )
    assert plan.specs["dense_dq"].dq.edge_tiles
    atomic = dense_bwd_spec_for(req, arch)
    with pytest.raises(ValueError, match="need dq_mode='split'"):
        attn_dense_bwd_plan(
            req, arch, policy=pol, spec=atomic, runtime={"dq_edge_tiles": True}
        )
    # an atomic plan is unchanged: no dense_dq launch or spec, no dq runtime
    plain = attn_dense_bwd_plan(req, arch, policy=pol)
    assert [s.stage for s in plain.launches] == ["prep", "dense", "convert"]
    assert "dense_dq" not in plain.specs and "dq_mode" not in plain.runtime


# ---------------------------------------------------------------------------
# run glue, compile cache, registry, sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_run_and_compile_hooks_serve_the_dense_dq_stage(arch):
    from kernels.common import attention_bwd_cache as cache
    from kernels.common import attention_bwd_run as run

    plan = _plan(arch)
    step = plan.launches[2]
    assert run._stage_params("dense_dq") == attn_dense_bwd_dq_params()
    tensors = {
        n: 0x1000 * (i + 1) for i, n in enumerate("q k v o do lse dq dk dv".split())
    }
    blob = run.pack_attn_bwd_args(
        step, tensors=tensors, workspace=1 << 20, workspace_layout=plan.workspace_layout
    )
    assert len(blob) == 7 * 8 + 3 * 4 + 3 * 4
    ptrs = struct.unpack_from("<7Q", blob, 0)
    assert ptrs[:5] == tuple(tensors[n] for n in ("q", "k", "v", "do", "dq"))
    assert ptrs[5] == (1 << 20) + plan.workspace_layout["WS_LSE2"][0]
    (batch,) = struct.unpack_from("<i", blob, 7 * 8 + 3 * 4)
    assert batch == 2
    kernel = run._BUILDERS["dense_dq"](plan.specs["dense_dq"], arch=arch)
    assert kernel.name == step.kernel
    assert bc._builder("dense_dq") is run._build_dense_dq
    assert "dense_dq" in cache.STAGES
    assert bc.spec_identity("dense_dq", plan.specs["dense_dq"]).startswith("dense_dq|")
    assert bc.not_built_reasons("dense_dq", plan.specs["dense_dq"]) == []


def test_dense_registry_builds_dq_mode_and_takes_the_dq_axes():
    from dispatch.attention_bwd import dense_tuning as dt
    from dispatch.attention_bwd import tuning_common as tc

    ax = dt.dense_axis("dq_mode")
    assert ax.choices == ("atomic", "split") and ax.not_built == ()
    assert ax.numerics == "reorders_fp"
    for name in DQ_KNOBS:
        assert dt.dense_axis(name) is tc.backward_axis(name)
    names = {a.name for a in dt.dense_tuning_axes("gfx950")}
    assert set(DQ_KNOBS) <= names and "dq_mode" in names
    # the split main kernel's couplings are the general family's
    assert not ax.applies("split", {"loop_sync": "counted"})
    assert ax.applies("split", {}) and ax.applies("atomic", {"loop_sync": "counted"})
    for arch in ARCHS:
        spec = _spec(arch, dq_mode="split")
        assert dt.dense_spec_legality(spec, arch) == "legal"


def test_a_split_tuned_entry_plans_on_its_key():
    """A ``DENSE_BWD_TUNED`` value the sweep emits for a split candidate (with
    its dQ values) routes dense and plans the dense dQ kernel."""
    from kernels.common.attention_bwd_plan import _dense_key

    from .dispatch.attention_bwd.test_dense_tuning_registry import _tuned_entry_plans

    sw = _sweep()
    gsl, sl = _dense_slice(sw)
    knobs = dict(sw.default_knobs(gsl), global_path="dma", ring_depth=2,
                 dq_mode="split", dq_edge_tiles=True)  # fmt: skip
    c = sw.Candidate.make(sl, knobs, {"g_split": 1, "route": "dense"})
    cell = sw.dense_cells(sl, sw.representative_cells(sl), 128)[0]
    key = _dense_key(sw.cell_request(cell))
    value = {
        k: tuple(v) if isinstance(v, list) else v
        for k, v in sw.dense_tuned_value(c).items()
    }
    (plan,) = _tuned_entry_plans({("gfx950", key): value})
    assert [s.stage for s in plan.launches][:3] == ["prep", "dense", "dense_dq"]
    assert plan.specs["dense_dq"].dq.edge_tiles


def _sweep():
    from benchmarks.common import attention_bwd_sweep as sw

    return sw


def _dense_slice(sw, arch="gfx950", d=128):
    return sw.primary_slice(arch, "general", d), sw.primary_slice(arch, "dense", d)


def test_sweep_dense_points_move_dq_mode():
    sw = _sweep()
    _gsl, sl = _dense_slice(sw)
    rel = sw.Release(sw.ALL_RELEASED)
    atomic = sw.Candidate.make(sl, {"waves": 8}, {"g_split": 1, "route": "dense"})
    pts = sw._dense_dq_points(atomic, rel)
    modes = [c.k.get("dq_mode", "atomic") for c in pts]
    assert modes[0] == "atomic" and "split" in modes
    split_pts = [c for c in pts if c.k.get("dq_mode") == "split"]
    # the split point alone, and with the dQ kernel's edge / interior loops
    assert {c.k.get("dq_edge_tiles", False) for c in split_pts} == {False, True}
    assert not any(
        k in DQ_KNOBS
        for c in pts
        if c.k.get("dq_mode", "atomic") == "atomic"
        for k in c.k
    )
    seed = sw.Candidate.make(
        sl, {"waves": 8, "dq_mode": "split", "dq_block_m": 128},
        {"g_split": 1, "route": "dense"},
    )  # fmt: skip
    pts = sw._dense_dq_points(seed, rel)
    assert pts[0] == seed.with_knobs(**sw.dense_points(seed, rel)[0])
    for c in pts:
        if c.k.get("dq_mode", "atomic") == "atomic":
            assert not any(k in DQ_KNOBS for k in c.k)  # dropped with the mode
        else:
            assert c.k["dq_block_m"] == 128  # the seed's dQ values kept


def test_sweep_dense_legality_accepts_split():
    sw = _sweep()
    _gsl, sl = _dense_slice(sw)
    c = sw.Candidate.make(sl, {"dq_mode": "split"}, {"g_split": 1, "route": "dense"})
    lg = sw.legality(c)
    assert lg.ok, lg
    assert lg.dq is not None
    # the dense plan of a cohort cell builds both dense kernels
    cell = sw.dense_cells(sl, sw.representative_cells(sl), 128)[0]
    p = sw.dense_plan(c, cell)
    assert [s.stage for s in p.launches][:3] == ["prep", "dense", "dense_dq"]
    # a split candidate with a dQ value the dQ validator refuses: a dq_ rule
    bad = c.with_knobs(dq_waves=8)  # eight waves on the 64-row start tile
    assert not sw.legality(bad).ok and sw.legality(bad).rule.startswith("dq_")
    # the dense keys of the two dQ modes differ (the dQ kernel is keyed in)
    a = sw.Candidate.make(sl, {}, {"g_split": 1, "route": "dense"})
    assert sw.effective_key(a) != sw.effective_key(c)
    assert sw.effective_key(c) != sw.effective_key(c.with_knobs(dq_edge_tiles=True))


@pytest.mark.parametrize("dq_rule", [None, "spill"])
def test_sweep_dense_compile_stage_gates_the_dense_dq_kernel(
    monkeypatch, tmp_path, dq_rule
):
    """The dense compile stage compiles each cell's dense dQ kernel beside its
    dense kernel and applies the dQ resource rules to it (its resolved dQ
    spec); a dQ prune fails the candidate as ``pruned:dq_<rule>``."""
    from collections import Counter

    from . import test_attention_bwd_sweep as ts

    sw = _sweep()
    sl = ts._dsl()
    c = sw.Candidate.make(sl, {"dq_mode": "split"}, {"g_split": 1, "route": "dense"})
    cells = sw.dense_cells(sl, sw.representative_cells(sl), 128)[:2]
    seen = []

    def fake_compile(items, **_kw):
        seen.extend(st for st, _ in items)
        return [ts._Res(True) for _ in items]

    class _Gate:
        calls: list = []

        def static_prune(self, arch, spec, hsaco):
            self.calls.append(("main", type(spec).__name__))
            return None, {}

        def static_prune_dq(self, arch, spec, hsaco):
            self.calls.append(("dq", type(spec).__name__))
            return dq_rule, {}

    monkeypatch.setattr(bc, "compile_batch", fake_compile)
    drv = ts._driver(tmp_path)
    gate = _Gate()
    drv._gate = gate
    path = tmp_path / "x.jsonl"
    counts = Counter()
    surv = drv._compile_and_prune_dense([c], cells, path, {}, counts)
    assert seen == ["dense", "dense_dq"] * len(cells)
    if dq_rule is None:
        assert [s[0] for s in surv] == [c]
        assert gate.calls == [("main", "AttnBwdSpec"), ("dq", "AttnBwdDqSpec")] * len(
            cells
        )
    else:
        assert surv == [] and counts["pruned"] == 1
        assert gate.calls == [("main", "AttnBwdSpec"), ("dq", "AttnBwdDqSpec")]
        assert "pruned:dq_spill" in path.read_text()
    # an atomic candidate compiles its dense kernels only
    seen.clear()
    gate.calls.clear()
    a = sw.Candidate.make(sl, {}, {"g_split": 1, "route": "dense"})
    drv._compile_and_prune_dense([a], cells, path, {}, Counter())
    assert seen == ["dense"] * len(cells)
