# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dense SDPA backward: spec, compile-time band classification, plan (CPU).

* spec: required fields, value sets, the band of every mask form, kernel
  names salted by every non-default knob, ``body_knobs`` canonical and
  limited to the shared body's tuning knobs, per-arch subclasses;
* validation: ragged / aligned tiles, the arch of a per-arch spec, the
  not-built knob values;
* the compile-time classification of :class:`DiagonalBand` /
  :class:`DenseNoMask` against the general policies, evaluated on the host
  over every kv tile of small shapes (every mask form, both diagonals);
* the dense plan: prep / dense / convert launches with the ABI's kernargs,
  the general family's workspace and multipliers, the runtime knobs and
  their refusals.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools

import pytest

from kernels.common._attention_bwd_band import (
    NoMaskKTail,
    RuntimeBand,
    HostInt,
    static_kv_tile_plan,
)
from kernels.common._attention_dense_bwd_shell import (
    DenseNoMask,
    DiagonalBand,
    dense_static_classes,
)
from kernels.common.attention_bwd_plan import (
    AttnBwdRequest,
    attn_bwd_plan,
    dense_bwd_eligibility,
)
from kernels.common.attention_dense_bwd_plan import (
    attn_dense_bwd_plan,
    dense_bwd_spec_class,
    dense_bwd_spec_for,
)
from kernels.common.attention_dense_bwd_spec import (
    DENSE_BODY_KNOBS,
    AttentionDenseBwdSpec,
    attn_dense_bwd_params,
    dense_bwd_not_built_reasons,
    dense_layout_strides,
    validate_attn_dense_bwd_spec,
)
from kernels.gfx942.attention_dense_bwd import (
    Gfx942AttentionDenseBwdSpec,
)
from kernels.gfx942.attention_dense_bwd import (
    build_attention_dense_bwd as build942,
)
from kernels.gfx950.attention_dense_bwd import (
    Gfx950AttentionDenseBwdSpec,
)
from kernels.gfx950.attention_dense_bwd import (
    build_attention_dense_bwd as build950,
)
from rocke.helpers.attention_band import AttnRuntimeBounds

from .sdpa.bwd_kernel_cases import config_policy

ARCHS = ("gfx942", "gfx950")
_SPEC = {"gfx942": Gfx942AttentionDenseBwdSpec, "gfx950": Gfx950AttentionDenseBwdSpec}
_BUILD = {"gfx942": build942, "gfx950": build950}


def _spec(arch="gfx942", **kw):
    base = dict(
        seqlen_q=256, seqlen_kv=256, num_query_heads=8, num_kv_heads=2, head_size=128
    )
    base.update(kw)
    return _SPEC[arch](**base)


def _req(**kw):
    base = dict(
        b=2, h_q=8, h_k=2, h_v=2, s_q=256, s_kv=256, d_qk=128, d_v=128, dtype="bf16"
    )
    base.update(kw)
    layout = base.pop("dense_layout", "bshd")
    req = AttnBwdRequest(**base)
    strides = {}
    for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
        sb, sh, st = dense_layout_strides(
            layout, req.tensor_heads(n), req.tensor_seq(n), req.d_qk
        )
        strides[n] = (sb, sh, st, 1)
    return dataclasses.replace(req, strides=strides)


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


def test_required_fields_are_the_problem_shape():
    for cls in (AttentionDenseBwdSpec, *_SPEC.values()):
        required = sorted(
            f.name
            for f in dataclasses.fields(cls)
            if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        )
        assert required == sorted(
            ["seqlen_q", "seqlen_kv", "num_query_heads", "num_kv_heads", "head_size"]
        )


@pytest.mark.parametrize("arch", ARCHS)
def test_builder_signature(arch):
    sig = inspect.signature(_BUILD[arch])
    assert list(sig.parameters) == ["spec", "arch"]
    assert sig.parameters["arch"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    "kw",
    [
        dict(head_size=32),
        dict(seqlen_q=0),
        dict(num_query_heads=6, num_kv_heads=4),
        dict(dtype="fp32"),
        dict(layout="thd"),
        dict(epilogue="vector"),
        dict(oob="wrap"),
        dict(dkv_mode="split"),
        dict(ws_layout="row"),
        dict(persist_decode="hkv_major"),  # needs persistent
        dict(persistent=True, xcd_remap=True),
        dict(causal=False, sliding_window=64),
        dict(causal=True, window_left=3),
        dict(causal=False, window_right=-2),
        dict(batch=0),
        dict(body_knobs={"head_size": 64}),
        dict(body_knobs={"ws_layout": "token_major"}),
        dict(body_knobs={"not_a_knob": 1}),
        dict(body_knobs={"block_m": 48}),
        dict(body_knobs={"xcd_remap": True}),  # the dense spec owns the remap
        dict(body_variant="trload"),  # gfx942 has no trload body
        dict(dq_mode="workspace"),  # not a dQ mode
    ],
)
def test_bad_spec_values_raise(kw):
    with pytest.raises(ValueError):
        _spec("gfx942", **kw)


def test_body_knobs_are_canonical():
    a = _spec(body_knobs={"waves": 4, "block_n": 64, "warp_grid_g4": [1, 4]})
    b = _spec(body_knobs=(("warp_grid_g4", (1, 4)), ("block_n", 64), ("waves", 4)))
    assert a == b and hash(a) == hash(b)
    assert a.body_knobs == (("block_n", 64), ("warp_grid_g4", (1, 4)), ("waves", 4))
    assert a.body_spec().waves == 4
    # an entry equal to the body default is dropped: one kernel, one spec
    plain = _spec()
    for knobs in ({"static_scale": "fold_ds"}, {"lb_interleave": False}):
        same = _spec(body_knobs=knobs)
        assert same == plain and same.kernel_name() == plain.kernel_name()
    assert _spec(body_knobs={"static_scale": "runtime"}).body_knobs == (
        ("static_scale", "runtime"),
    )
    assert set(DENSE_BODY_KNOBS).isdisjoint(
        {
            "head_size",
            "dtype",
            "dkv_mode",
            "ws_layout",
            "dkv_epilogue",
            "edge_tiles",
            "xcd_remap",
        }
    )
    # plan-derived body knobs ride in body_knobs, as in the general family
    assert {"static_scale", "lb_interleave"} <= set(DENSE_BODY_KNOBS)


@pytest.mark.parametrize(
    "kw,band",
    [
        (dict(causal=False), (-1, -1, False)),
        (dict(causal=True), (-1, 0, False)),
        (dict(causal=True, causal_bottom_right=True), (-1, 0, True)),
        (dict(causal=True, sliding_window=64), (63, 0, False)),
        (dict(causal=True, sliding_window=1, causal_bottom_right=True), (0, 0, True)),
        (dict(causal=False, window_left=7, window_right=3), (7, 3, False)),
        (dict(causal=False, window_right=5), (-1, 5, False)),
        (
            dict(causal=False, window_left=9, causal_bottom_right=True),
            (9, -1, True),
        ),
    ],
)
def test_band_of_every_mask_form(kw, band):
    s = _spec(**kw)
    assert s.band() == band
    assert s.masked == (band[:2] != (-1, -1))
    assert s.body_spec().mask_class == ("band" if s.masked else "none")


def test_kernel_name_salted_by_every_knob():
    base = _spec()
    names = {base.kernel_name()}
    variants = [
        dict(epilogue="direct"),
        dict(epilogue="lds_staged_pair"),
        dict(oob="clamp"),
        dict(edge_tiles=False),
        dict(ws_layout="token_major"),
        dict(dkv_mode="atomic"),
        dict(persistent=True),
        dict(persistent=True, persist_decode="hkv_major"),
        dict(xcd_remap=True),
        dict(body_knobs={"waves": 4}),
        dict(body_knobs={"stage_order": "g3g1g4"}),
        dict(name="rocke_attn_dense_bwd2"),
        dict(ragged=True),
        dict(batch=3),
        dict(layout="bhsd"),
        dict(causal=False),
        dict(causal_bottom_right=True),
        dict(sliding_window=128),
        dict(dtype="fp16"),
    ]
    for kw in variants:
        n = _spec(**kw).kernel_name()
        assert n not in names, kw
        names.add(n)
    assert base.kernel_name().startswith(
        "rocke_attn_dense_bwd_bf16_d128_bshd_sq256_sk256_hq8_hk2_causaltl"
    )


def test_per_arch_specs_differ_only_in_body_variant():
    f942 = {f.name for f in dataclasses.fields(Gfx942AttentionDenseBwdSpec)}
    f950 = {f.name for f in dataclasses.fields(Gfx950AttentionDenseBwdSpec)}
    base = {f.name for f in dataclasses.fields(AttentionDenseBwdSpec)}
    assert f942 == f950 == base | {"body_variant"}
    assert _spec("gfx950", body_variant="trload").body_variant == "trload"
    assert _spec("gfx942", body_variant="xt").body_variant == "xt"


# ---------------------------------------------------------------------------
# validation and not-built values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_aligned_needs_tile_multiples(arch):
    body = validate_attn_dense_bwd_spec(_spec(arch), arch)
    with pytest.raises(ValueError, match="ragged"):
        validate_attn_dense_bwd_spec(_spec(arch, seqlen_kv=body.block_n + 1), arch)
    validate_attn_dense_bwd_spec(
        _spec(arch, seqlen_kv=body.block_n + 1, ragged=True), arch
    )


def test_wrong_arch_raises():
    with pytest.raises(ValueError, match="gfx942 spec"):
        validate_attn_dense_bwd_spec(_spec("gfx942"), "gfx950")
    with pytest.raises(TypeError):
        build950(_spec("gfx942"), arch="gfx950")
    with pytest.raises(ValueError):
        validate_attn_dense_bwd_spec(_spec("gfx942"), "gfx1151")


@pytest.mark.parametrize("arch,variant", [("gfx942", "xt"), ("gfx950", "trload")])
def test_per_arch_bodies_are_registered_not_built(arch, variant):
    with pytest.raises(NotImplementedError, match="body_variant"):
        _BUILD[arch](_spec(arch, body_variant=variant), arch=arch)


def test_not_built_body_knob_raises():
    # the mirror pass (a second kv tile per CTA) is the general shell's only
    with pytest.raises(NotImplementedError, match="lb_mirror"):
        build942(_spec(body_knobs={"lb_mirror": True}), arch="gfx942")


def _ring_values(arch):
    """Every built (global_path, ring_depth) of ``arch`` but the synchronous
    staging, on the default tile."""
    out = []
    for path in ("vgpr", "dma"):
        for depth in (1, 2, 3):
            if (path, depth) == ("vgpr", 1):
                continue
            try:
                validate_attn_dense_bwd_spec(
                    _spec(arch, body_knobs={"global_path": path, "ring_depth": depth}),
                    arch,
                )
            except ValueError:
                continue
            out.append((path, depth))
    return out


def test_ring_values_per_arch():
    assert _ring_values("gfx942") == [("vgpr", 2)]
    assert _ring_values("gfx950") == [
        ("vgpr", 2), ("vgpr", 3), ("dma", 1), ("dma", 2), ("dma", 3),
    ]  # fmt: skip


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("oob", ["buffer_oob", "clamp"])
@pytest.mark.parametrize("ragged", [False, True])
def test_prefetch_ring_is_built_on_the_dense_shell(arch, oob, ragged):
    """The Q / dO prefetch ring (register and DMA paths, every depth of the
    arch) builds on the dense shell under both OOB modes; under buffer_oob
    Q and dO leave the buffer resources (their global-pointer view) while K
    and V keep theirs."""
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    shape = dict(seqlen_q=200, seqlen_kv=200, ragged=True) if ragged else {}
    plain = _BUILD[arch](_spec(arch, oob=oob, **shape), arch=arch)
    n_plain = _count_ops(plain, "tile.buffer_rsrc")
    for path, depth in _ring_values(arch):
        knobs = {"global_path": path, "ring_depth": depth}
        spec = _spec(arch, oob=oob, body_knobs=knobs, **shape)
        assert dense_bwd_not_built_reasons(spec) == []
        k = _BUILD[arch](spec, arch=arch)
        ir = _lower_kernel_to_llvm_python(k, arch=arch, llvm_flavor="llvm22")
        assert ("global.load.lds" in ir or "global_load_lds" in ir) == (path == "dma")
        n = _count_ops(k, "tile.buffer_rsrc")
        if oob == "clamp":
            assert n == 0
        else:
            assert 0 < n < n_plain, (path, depth, n, n_plain)


def _count_ops(kernel, name):
    def walk(ops):
        for op in ops:
            yield op
            for r in op.regions:
                yield from walk(r.ops)

    return sum(1 for op in walk(kernel.body.ops) if op.name == name)


@pytest.mark.parametrize("arch", ARCHS)
def test_merged_ring_needs_no_prefetch_ring(arch, monkeypatch):
    """barrier_merge = "on" stages its own 2-slot ring in the tile step: the
    dense shell builds no prefetch ring for it (the general shell's rule),
    and every other staging but (vgpr, 1) builds one."""
    from kernels.common import _attention_dense_bwd_shell as shell

    built = []
    real = shell.QRing

    def spy(*a, **kw):
        built.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(shell, "QRing", spy)
    _BUILD[arch](_spec(arch, body_knobs={"barrier_merge": "on"}), arch=arch)
    _BUILD[arch](_spec(arch), arch=arch)
    assert built == []
    _BUILD[arch](_spec(arch, body_knobs={"ring_depth": 2}), arch=arch)
    assert built == [1]


@pytest.mark.parametrize("arch", ARCHS)
def test_split_dq_is_built(arch):
    """``dq_mode = "split"`` is built on the dense shell (the main kernel
    computes dK / dV only; ``test_attention_dense_bwd_split.py`` covers the
    kernel, its dQ kernel and the plan)."""
    spec = _spec(arch, dq_mode="split")
    assert dense_bwd_not_built_reasons(spec) == []
    assert _BUILD[arch](spec, arch=arch).name == spec.kernel_name()


# ---------------------------------------------------------------------------
# compile-time band classification (host evaluation of the emitters)
# ---------------------------------------------------------------------------

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


def _host_policy(dense: bool, s_q, s_kv, left, right, br, km, kn):
    hb = HostInt()
    masked = not (left == -1 and right == -1)
    if dense:
        if masked:
            return DiagonalBand(
                hb, s_q=s_q, s_kv=s_kv, left=left, right=right, bottom_right=br,
                k_m0=km, k_n0=kn,
            )  # fmt: skip
        return DenseNoMask(hb, s_q=s_q, s_kv=s_kv, k_m0=km, k_n0=kn)
    if masked:
        off = s_kv - s_q if br else 0
        return RuntimeBand(
            hb, AttnRuntimeBounds(s_q, s_kv, off, left, right), k_m0=km, k_n0=kn
        )
    return NoMaskKTail(hb, s_q, s_kv, k_m0=km, k_n0=kn)


@pytest.mark.parametrize("mask", _MASKS)
def test_static_classification_matches_general_policy(mask):
    left, right, br = mask
    hb = HostInt()
    for km, kn in ((16, 16), (16, 64), (32, 128)):
        for s_q, s_kv in itertools.product((1, 16, 33, 64, 100), (16, 47, 64, 130)):
            dense = _host_policy(True, s_q, s_kv, left, right, br, km, kn)
            general = _host_policy(False, s_q, s_kv, left, right, br, km, kn)
            any_edge = any_interior = False
            for kt in range(-(-s_kv // kn)):
                k0 = kt * kn
                qs, nq = dense.q_tile_range(hb, k0)
                assert (qs, nq) == general.q_tile_range(hb, k0)
                # the same tiles in the same order: edge runs [qs, i0) and
                # [i1, qe), interior [i0, i1); a dropped run may move the split
                parts = []
                for pol in (dense, general):
                    i0, i1 = pol.interior_q_range(hb, k0, qs, nq)
                    assert qs <= i0 <= i1 <= qs + nq
                    parts.append(
                        [(t, True) for t in range(qs, i0)]
                        + [(t, False) for t in range(i0, i1)]
                        + [(t, True) for t in range(i1, qs + nq)]
                    )
                assert parts[0] == parts[1], (s_q, s_kv, km, kn, kt)
                plan_edges = static_kv_tile_plan(
                    s_q=s_q, s_kv=s_kv, k0=k0, k_m0=km, k_n0=kn, left=left,
                    right=right, bottom_right=br,
                    masked=not (left == -1 and right == -1),
                ).edge_tiles  # fmt: skip
                assert [t for t, e in parts[0] if e] == list(plan_edges)
                plan = static_kv_tile_plan(
                    s_q=s_q, s_kv=s_kv, k0=k0, k_m0=km, k_n0=kn, left=left,
                    right=right, bottom_right=br,
                    masked=not (left == -1 and right == -1),
                )  # fmt: skip
                any_edge |= bool(plan.edge_tiles)
                any_interior |= bool(plan.interior_tiles)
            for q in range(0, s_q + km, 7):
                for k in range(0, s_kv + kn, 5):
                    assert dense.cell_keep(hb, q, k) == general.cell_keep(hb, q, k)
            assert (dense.any_edge, dense.any_interior) == (any_edge, any_interior)
            assert dense_static_classes(
                s_q=s_q, s_kv=s_kv, k_m0=km, k_n0=kn, left=left, right=right,
                bottom_right=br, masked=not (left == -1 and right == -1),
            ) == (any_edge, any_interior)  # fmt: skip


def test_aligned_unmasked_has_no_edge_tile():
    assert dense_static_classes(
        s_q=256, s_kv=512, k_m0=16, k_n0=128, left=-1, right=-1, bottom_right=False,
        masked=False,
    ) == (False, True)  # fmt: skip
    assert dense_static_classes(
        s_q=256, s_kv=256, k_m0=16, k_n0=128, left=-1, right=0, bottom_right=False,
        masked=True,
    ) == (True, True)  # fmt: skip


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def _policy(arch):
    return config_policy("default", arch)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
def test_dense_plan_launches_and_kernargs(arch, layout):
    req = _req(dense_layout=layout, causal=True)
    assert dense_bwd_eligibility(req, arch) == (True, "dense_eligible")
    plan = attn_dense_bwd_plan(req, arch, policy=_policy(arch))
    general = attn_bwd_plan(req, arch, policy=_policy(arch))
    assert plan.route == "dense"
    assert [s.stage for s in plan.launches] == ["prep", "dense", "convert"]
    assert plan.workspace_bytes == general.workspace_bytes
    assert dict(plan.workspace_layout) == dict(general.workspace_layout)
    dense = plan.launches[1]
    names = [n for n, _ in attn_dense_bwd_params()]
    assert sorted(list(dense.pointers) + list(dense.scalars)) == sorted(names)
    gmain = general.launches[1]
    for k in ("scale_log2", "ds_mult", "dk_mult", "dq_mult"):
        assert dense.scalars[k] == gmain.scalars[k]
    spec = plan.specs["dense"]
    assert spec.layout == layout and spec.batch is None and not spec.ragged
    body = validate_attn_dense_bwd_spec(spec, arch)
    assert dense.grid == (256 // body.block_n, 2, 2)
    assert dense.scalars["batch"] == 2 and dense.scalars["g_split"] == 1
    assert dense.pointers["dK"] == "dk" and dense.pointers["WS_DK"] is None
    assert plan.launches[0] == general.launches[0]
    assert plan.launches[2] == general.launches[2]


@pytest.mark.parametrize("arch", ARCHS)
def test_dense_spec_for_request_masks_and_ragged(arch):
    cls = dense_bwd_spec_class(arch)
    for kw, band in (
        (dict(), (-1, -1, False)),
        (dict(causal=True), (-1, 0, False)),
        (dict(causal_bottom_right=True, s_q=128), (-1, 0, True)),
        (dict(left_bound=63, right_bound=0), (63, 0, False)),
        (dict(left_bound=7, right_bound=3, bottom_right=True), (7, 3, True)),
        (dict(left_bound=-1, right_bound=9), (-1, 9, False)),
    ):
        req = _req(**kw)
        spec = dense_bwd_spec_for(req, arch)
        assert isinstance(spec, cls)
        assert spec.band() == band == req.resolved_band()
        assert not spec.ragged
    spec = dense_bwd_spec_for(_req(s_kv=200, causal=True), arch)
    assert spec.ragged


@pytest.mark.parametrize("arch", ARCHS)
def test_dense_plan_runtime_knobs(arch):
    pol = _policy(arch)
    req = _req(causal=True, num_cus=304, num_xcds=8)
    base = dense_bwd_spec_for(req, arch)
    atomic = dataclasses.replace(base, dkv_mode="atomic")
    plan = attn_dense_bwd_plan(
        req, arch, policy=pol, spec=atomic, runtime={"g_split": 2}
    )
    d = plan.launches[1]
    assert d.scalars["g_split"] == 2 and d.grid[1] == 2 * 2
    assert [s.stage for s in plan.launches] == ["prep", "dense"] + ["convert"] * 3
    assert d.pointers["WS_DK"] == "WS_DK" and d.pointers["dK"] is None
    with pytest.raises(ValueError, match="atomic"):
        attn_dense_bwd_plan(req, arch, policy=pol, spec=base, runtime={"g_split": 2})
    with pytest.raises(ValueError, match="atomic"):
        attn_dense_bwd_plan(req, arch, policy=pol, spec=atomic)
    pers = dataclasses.replace(base, persistent=True)
    plan = attn_dense_bwd_plan(req, arch, policy=pol, spec=pers)
    assert plan.launches[1].grid == (304, 1, 1)
    assert plan.launches[1].scalars["num_persistent"] == 304
    plan = attn_dense_bwd_plan(
        req, arch, policy=pol, spec=pers, runtime={"num_persistent": 608}
    )
    assert plan.launches[1].grid == (608, 1, 1)
    with pytest.raises(ValueError, match="persistent"):
        attn_dense_bwd_plan(
            dataclasses.replace(req, num_cus=None), arch, policy=pol, spec=pers
        )
    plan = attn_dense_bwd_plan(
        req, arch, policy=pol, spec=base, runtime={"lb_order": "reverse"}
    )
    assert plan.launches[1].scalars["lb_order"] == 1
    with pytest.raises(ValueError, match="mirror"):
        attn_dense_bwd_plan(
            req, arch, policy=pol, spec=base, runtime={"lb_order": "mirror"}
        )
    gx = plan.launches[1].grid[0]
    with pytest.raises(ValueError, match="xcd_remap"):
        attn_dense_bwd_plan(req, arch, policy=pol, spec=base, runtime={"xcd_chunk": gx})
    xspec = dataclasses.replace(base, xcd_remap=True)
    plan = attn_dense_bwd_plan(
        req, arch, policy=pol, spec=xspec, runtime={"xcd_chunk": gx}
    )
    assert plan.launches[1].scalars["xcd_n"] == 8
    assert plan.launches[1].scalars["xcd_chunk"] == gx
    tok = dataclasses.replace(base, ws_layout="token_major")
    plan = attn_dense_bwd_plan(req, arch, policy=pol, spec=tok)
    assert plan.specs["prep"].ws_layout == "token_major"
    assert plan.specs["convert"].ws_layout == "token_major"


@pytest.mark.parametrize("arch", ARCHS)
def test_run_and_compile_hooks_serve_the_dense_stage(arch):
    import struct

    from benchmarks.common import attention_bwd_compile as bc
    from kernels.common import attention_bwd_run as run

    plan = attn_dense_bwd_plan(_req(causal=True), arch, policy=_policy(arch))
    step = plan.launches[1]
    assert run._stage_params("dense") == attn_dense_bwd_params()
    tensors = {
        n: 0x1000 * (i + 1) for i, n in enumerate("q k v o do lse dk dv".split())
    }
    blob = run.pack_attn_bwd_args(
        step, tensors=tensors, workspace=1 << 20, workspace_layout=plan.workspace_layout
    )
    assert len(blob) == 13 * 8 + 4 * 4 + 7 * 4
    q_ptr, k_ptr = struct.unpack_from("<QQ", blob, 0)
    assert (q_ptr, k_ptr) == (tensors["q"], tensors["k"])
    (batch,) = struct.unpack_from("<i", blob, 13 * 8 + 4 * 4)
    assert batch == 2
    kernel = run._BUILDERS["dense"](plan.specs["dense"], arch=arch)
    assert kernel.name == step.kernel
    assert bc._builder("dense") is run._build_dense
    assert bc.spec_identity("dense", plan.specs["dense"]).startswith("dense|")
    assert bc.not_built_reasons("dense", plan.specs["dense"]) == []
    variant = "xt" if arch == "gfx942" else "trload"
    spec = dataclasses.replace(plan.specs["dense"], body_variant=variant)
    assert bc.not_built_reasons("dense", spec)
    # a body knob value that is not built is a not-built dense spec too
    spec = dataclasses.replace(plan.specs["dense"], body_knobs={"lb_mirror": True})
    assert any("lb_mirror" in r for r in bc.not_built_reasons("dense", spec))
    # the prefetch ring is built
    spec = dataclasses.replace(plan.specs["dense"], body_knobs={"ring_depth": 2})
    assert bc.not_built_reasons("dense", spec) == []


@pytest.mark.parametrize("arch", ARCHS)
def test_hook_g_split_makes_dkv_atomic(arch):
    """A hook configuration with g_split > 1 gets atomic dK / dV (the general
    plan's rule); an explicit direct mode with a split is refused."""
    req = _req(causal=True)
    pol = config_policy("default", arch, overrides={"g_split": 2})
    plan = attn_dense_bwd_plan(req, arch, policy=pol)
    assert plan.specs["dense"].dkv_mode == "atomic"
    assert plan.launches[1].scalars["g_split"] == 2
    assert [s.stage for s in plan.launches].count("convert") == 3
    plan = attn_dense_bwd_plan(req, arch, policy=_policy(arch))
    assert plan.specs["dense"].dkv_mode == "direct"
    pol = config_policy("default", arch, overrides={"g_split": 2, "dkv_mode": "direct"})
    with pytest.raises(ValueError, match="atomic"):
        attn_dense_bwd_plan(req, arch, policy=pol)


@pytest.mark.parametrize("arch", ARCHS)
def test_head_pack_is_dropped_or_refused(arch):
    """head_pack = False (the general knob set's value) is what the dense
    body does; head_pack = True is not built on the dense route."""
    req = _req(causal=True)
    base = dense_bwd_spec_for(req, arch)
    for knobs in ({"head_pack": False}, {"body_knobs": {"head_pack": False}}):
        assert dense_bwd_spec_for(req, arch, knobs) == base
    plan = attn_dense_bwd_plan(
        req, arch, policy=config_policy("default", arch, {"head_pack": False})
    )
    assert plan.specs["dense"] == base
    for knobs in ({"head_pack": True}, {"body_knobs": {"head_pack": True}}):
        with pytest.raises(ValueError, match="head_pack=True is not built"):
            dense_bwd_spec_for(req, arch, knobs)
    with pytest.raises(ValueError, match="head_pack=True is not built"):
        attn_dense_bwd_plan(
            req, arch, policy=config_policy("default", arch, {"head_pack": True})
        )


@pytest.mark.parametrize("arch", ARCHS)
def test_dense_plan_refusals(arch):
    pol = _policy(arch)
    with pytest.raises(ValueError, match="dense-eligible"):
        attn_dense_bwd_plan(_req(h_v=1, h_k=2), arch, policy=pol)
    with pytest.raises(ValueError, match="dense-eligible"):
        attn_dense_bwd_plan(_req(d_qk=32, d_v=32), arch, policy=pol)
    other = dense_bwd_spec_for(_req(causal=True), arch)
    with pytest.raises(ValueError, match="band"):
        attn_dense_bwd_plan(_req(), arch, policy=pol, spec=other)
    with pytest.raises(ValueError, match="dense-eligible"):
        attn_dense_bwd_plan(_req(), "gfx1151", policy=_policy("gfx1151"))


@pytest.mark.parametrize("arch", ARCHS)
def test_dense_plan_derives_the_plan_owned_body_knobs(arch):
    pol = _policy(arch)
    req = _req(causal=True, num_cus=304, num_xcds=8)
    # static_scale follows the scale placement (the general plan's rule)
    for placement in ("fold_ds", "convert", "dq_before_atomic"):
        plan = attn_dense_bwd_plan(
            req, arch, policy=pol, runtime={"scale_placement": placement}
        )
        body = plan.specs["dense"].body_spec()
        assert body.static_scale == placement
        general = attn_bwd_plan(
            req,
            arch,
            policy=config_policy(
                "default", arch, overrides={"scale_placement": placement}
            ),
        )
        for k in ("ds_mult", "dk_mult", "dq_mult"):
            assert plan.launches[1].scalars[k] == general.launches[1].scalars[k]
    pinned = attn_dense_bwd_plan(
        req,
        arch,
        policy=config_policy("default", arch, overrides={"static_scale": "runtime"}),
        runtime={"scale_placement": "convert"},
    )
    assert pinned.specs["dense"].body_spec().static_scale == "runtime"
    base = dense_bwd_spec_for(req, arch)
    with pytest.raises(ValueError, match="static_scale"):
        attn_dense_bwd_plan(
            req, arch, policy=pol, spec=base, runtime={"scale_placement": "convert"}
        )
    # lb_order interleave: the decode switch is derived for a hook config and
    # required of a given spec
    plan = attn_dense_bwd_plan(
        req, arch, policy=pol, runtime={"lb_order": "interleave"}
    )
    assert plan.specs["dense"].body_spec().lb_interleave
    assert plan.launches[1].scalars["lb_order"] == 3
    with pytest.raises(ValueError, match="lb_interleave"):
        attn_dense_bwd_plan(
            req, arch, policy=pol, spec=base, runtime={"lb_order": "interleave"}
        )
    woven = dataclasses.replace(base, body_knobs={"lb_interleave": True})
    plan = attn_dense_bwd_plan(
        req, arch, policy=pol, spec=woven, runtime={"lb_order": "interleave"}
    )
    assert plan.launches[1].scalars["lb_order"] == 3
    # xcd_chunk from a hook config turns the remap on; symbolic chunks resolve
    gx = plan.launches[1].grid[0]
    for chunk, want in (("gx", gx), ("2gx", 2 * gx), ("4gx", 4 * gx), (64, 64)):
        plan = attn_dense_bwd_plan(req, arch, policy=pol, runtime={"xcd_chunk": chunk})
        assert plan.specs["dense"].xcd_remap
        assert plan.launches[1].scalars["xcd_n"] == 8
        assert plan.launches[1].scalars["xcd_chunk"] == want
    with pytest.raises(ValueError, match="xcd_chunk"):
        attn_dense_bwd_plan(req, arch, policy=pol, runtime={"xcd_chunk": 3})
    plan = attn_dense_bwd_plan(req, arch, policy=pol)
    assert not plan.specs["dense"].xcd_remap
    assert plan.launches[1].scalars["xcd_n"] == 0


def test_shape_pin_is_explicit_and_named():
    """The dense family pins the problem shape at compile time, explicitly:
    ``SHAPE_BINDING`` / ``SHAPE_FIELDS`` / ``shape_key`` (class-level, not
    spec fields, so descriptors and the knob registry are unchanged), one
    instance per shape key, the shape spelled in readable name tokens."""
    from kernels.common.attention_dense_bwd_spec import AttentionDenseBwdSpec

    assert AttentionDenseBwdSpec.SHAPE_BINDING == "compile_time"
    names = {f.name for f in dataclasses.fields(AttentionDenseBwdSpec)}
    assert set(AttentionDenseBwdSpec.SHAPE_FIELDS) <= names
    assert not {"SHAPE_BINDING", "SHAPE_FIELDS", "ARCH"} & names
    base = dict(seqlen_q=128, seqlen_kv=256, num_query_heads=8, num_kv_heads=2,
                head_size=64)  # fmt: skip
    s = AttentionDenseBwdSpec(**base)
    assert s.shape_key() == (128, 256, 8, 2, 64, "bshd", None)
    # the name of the default instance is unchanged (goldens, caches and
    # resource-gate rows are keyed by it)
    assert s.kernel_name() == (
        "rocke_attn_dense_bwd_bf16_d64_bshd_sq128_sk256_hq8_hk2_causaltl"
    )
    # every shape field moves the name; a knob that is not shape does not
    # move the shape key
    for field_name, value in (("seqlen_q", 64), ("seqlen_kv", 128),
                              ("num_query_heads", 4), ("num_kv_heads", 1),
                              ("head_size", 128), ("layout", "bhsd"), ("batch", 2)):  # fmt: skip
        other = dataclasses.replace(s, **{field_name: value})
        assert other.shape_key() != s.shape_key()
        assert other.kernel_name() != s.kernel_name(), field_name
    knob = dataclasses.replace(s, epilogue="direct")
    assert knob.shape_key() == s.shape_key() and knob.kernel_name() != s.kernel_name()
    for cls in (Gfx942AttentionDenseBwdSpec, Gfx950AttentionDenseBwdSpec):
        assert cls.SHAPE_BINDING == "compile_time" and cls.ARCH in ("gfx942", "gfx950")
        assert cls(**base).shape_key() == s.shape_key()
        assert cls(**base).variant() == "general"
    assert s.variant() == "general" and AttentionDenseBwdSpec.ARCH is None
