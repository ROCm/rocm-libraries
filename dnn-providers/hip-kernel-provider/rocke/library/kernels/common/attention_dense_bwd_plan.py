# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host plan of the dense SDPA backward (a route inside declared cells).

:func:`attn_dense_bwd_plan` plans a request the general predicate accepts and
:func:`~kernels.common.attention_bwd_plan.dense_bwd_eligibility` admits:
``prep -> dense main -> convert(dQ) [-> convert(dK), convert(dV)]`` on one
stream. Prep, the converts, the workspace layout and the scale placement are
the general family's (planned by
:func:`~kernels.common.attention_bwd_plan.attn_bwd_plan` with the same
``g_split``, ``ws_layout``, scale placement and aux knobs); only the main
launch is replaced by the dense kernel (``rocke.attn_dense_bwd.v1``).

Under ``dq_mode = "split"`` (a dense spec field) the plan is ``prep -> dense
main -> dense dQ [-> convert(dK), convert(dV)]``: the general plan in split
mode supplies prep without the dQ zero fill, the workspace without ``WS_DQ``
and the dQ kernel's multipliers (the dQ convert's multiplier folded into
``dq_mult``); its dQ launch is replaced by the dense dQ kernel
(:mod:`kernels.common.attention_dense_bwd_dq`, ``rocke.attn_dense_bwd_dq.v1``,
stage ``"dense_dq"``) on grid ``(ceil(s_q / R), h_q / H, B)``. The dQ
kernel's tuning values are the general plan's ``DQ_KNOBS`` (``dq_<field>``,
``dq_xcd_chunk``), from the hook or, with a given ``spec``, in ``runtime``;
they are refused without split.

:func:`dense_bwd_spec_for` builds the per-arch dense spec of a request:
problem fields from the request, the mask from its band, ``ragged`` when a
length is not a tile multiple, knobs from the tuning hook (``route =
"dense"``). Knob names the hook may return: every dense spec knob, the
shared body's knobs (merged into ``body_knobs``), the runtime plan values of
the general family (``g_split``, ``lb_order``, ``xcd_chunk``,
``scale_placement``; ``use_worklist`` must be False), the aux values, and
``num_persistent`` (persistent grids; default ``num_cus`` of the request).
``head_pack = False`` is accepted (and dropped: the dense body never packs
heads); ``head_pack = True`` is refused as not built (short-query GQA
requests stay on the general route). A hook configuration with ``g_split >
1`` gets ``dkv_mode = "atomic"`` unless it sets ``dkv_mode`` itself, as in
the general plan.
As in the general plan, the body knobs ``static_scale`` (from
``scale_placement``, unless pinned to ``"runtime"``) and ``lb_interleave``
(from ``lb_order = "interleave"``) and the dense ``xcd_remap`` (from a non-zero
``xcd_chunk`` with the request's ``num_xcds``) are derived for a hook's
configuration; a given ``spec`` must already carry them.

Routing among the families lives in the dispatch layer
(``dispatch.attention_bwd.routing``); this module never decides a route.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
from typing import Any, Mapping, Optional

from kernels.common.attention_bwd_plan import (
    _GRID_YZ_MAX,
    AUX_KNOBS,
    DQ_KNOBS,
    FEATURE_GROUPS,
    LB_ORDERS,
    RUNTIME_KNOBS,
    SCALE_PLACEMENTS,
    AttnBwdPlan,
    AttnBwdPolicy,
    AttnBwdRequest,
    LaunchStep,
    _dense_layout_of,
    _fill_bucket,
    _freeze,
    _Q_SIDE,
    attn_bwd_perf_class,
    attn_bwd_plan,
    attn_bwd_support,
    dense_bwd_eligibility,
    shipped_policy,
)
from kernels.common.attention_bwd import attn_bwd_arch_facts, static_unit_mults
from kernels.common.attention_bwd_dq import dq_static_unit_mults
from kernels.common.attention_dense_bwd_dq import (
    AttentionDenseBwdDqSpec,
    attn_dense_bwd_dq_grid,
    attn_dense_bwd_dq_params,
    validate_attn_dense_bwd_dq_spec,
)
from kernels.common.attention_dense_bwd_spec import (
    DENSE_BODY_KNOBS,
    AttentionDenseBwdSpec,
    attn_dense_bwd_params,
    validate_attn_dense_bwd_spec,
)

__all__ = [
    "DENSE_RUNTIME_KNOBS",
    "attn_dense_bwd_plan",
    "dense_bwd_spec_class",
    "dense_bwd_spec_for",
]

# Runtime values of the dense plan on top of the general family's.
DENSE_RUNTIME_KNOBS = ("num_persistent",)
_DENSE_SPEC_KNOBS = tuple(
    f.name
    for f in dataclasses.fields(AttentionDenseBwdSpec)
    if f.default is not dataclasses.MISSING
    and f.name
    not in (
        "batch",
        "causal",
        "causal_bottom_right",
        "dtype",
        "layout",
        "sliding_window",
        "window_left",
        "window_right",
        "ragged",
        "name",
        "body_knobs",
    )
) + ("body_variant",)


def dense_bwd_spec_class(arch: str):
    """The per-arch dense spec class (imported lazily: the per-arch modules
    import this family's common modules)."""
    if arch not in ("gfx942", "gfx950"):
        raise ValueError(f"no dense backward on {arch}")
    mod = importlib.import_module(f"kernels.{arch}.attention_dense_bwd")
    return getattr(mod, f"{arch.capitalize()}AttentionDenseBwdSpec")


def _mask_fields(req: AttnBwdRequest) -> dict:
    left, right, br = req.resolved_band()
    if left == -1 and right == -1:
        return dict(causal=False, causal_bottom_right=br)
    if right == 0 and left == -1:
        return dict(causal=True, causal_bottom_right=br)
    if right == 0 and left >= 0:
        return dict(causal=True, causal_bottom_right=br, sliding_window=left + 1)
    return dict(
        causal=False,
        causal_bottom_right=br,
        window_left=None if left == -1 else left,
        window_right=None if right == -1 else right,
    )


def _check_head_pack(value) -> None:
    """``head_pack`` in a dense knob set: ``False`` is what the dense body
    does (it is dropped); ``True`` is refused (not built: the dense shell
    stages one query head per q step and has no packed dQ sink or
    ``pack_heads`` kernarg; short-query GQA requests stay on the general
    route, whose plan packs them)."""
    if value is not False:
        raise ValueError(
            f"head_pack={value!r} is not built on the dense route (one query "
            "head per q step); short-query GQA requests stay on the general route"
        )


def dense_bwd_spec_for(
    req: AttnBwdRequest, arch: str, knobs: Optional[Mapping[str, Any]] = None
) -> AttentionDenseBwdSpec:
    """The dense spec serving ``req`` on ``arch`` (validated), with ``knobs``.

    ``knobs`` holds dense spec knobs and the shared body's knobs (the latter
    go to ``body_knobs``). ``ragged`` is chosen from the resolved tile:
    ``True`` exactly when a length is not a tile multiple.
    """
    ok, tag = dense_bwd_eligibility(req, arch)
    if not ok:
        raise ValueError(f"request is not dense-eligible: {tag}")
    knobs = dict(knobs or {})
    body = dict(knobs.pop("body_knobs", ()) or ())
    for k in list(knobs):
        if k in DENSE_BODY_KNOBS:
            body[k] = knobs.pop(k)
    for src in (knobs, body):
        if "head_pack" in src:
            _check_head_pack(src.pop("head_pack"))
    unknown = set(knobs) - set(_DENSE_SPEC_KNOBS)
    if unknown:
        raise ValueError(f"unknown dense knobs {sorted(unknown)}")
    cls = dense_bwd_spec_class(arch)
    base = dict(
        seqlen_q=req.s_q,
        seqlen_kv=req.s_kv,
        num_query_heads=req.h_q,
        num_kv_heads=req.h_k,
        head_size=req.d_qk,
        dtype=req.dtype,
        layout=_dense_layout_of(req, _Q_SIDE),
        body_knobs=tuple(sorted(body.items())),
        **_mask_fields(req),
        **knobs,
    )
    spec = cls(**base, ragged=False)
    body_spec = validate_attn_dense_bwd_spec(
        dataclasses.replace(spec, ragged=True), arch
    )
    aligned = req.s_q % body_spec.block_m == 0 and req.s_kv % body_spec.block_n == 0
    if not aligned:
        spec = dataclasses.replace(spec, ragged=True)
    validate_attn_dense_bwd_spec(spec, arch)
    return spec


def attn_dense_bwd_plan(
    req: AttnBwdRequest,
    arch: str,
    *,
    policy: Optional[AttnBwdPolicy] = None,
    spec: Optional[AttentionDenseBwdSpec] = None,
    runtime: Optional[Mapping[str, Any]] = None,
) -> AttnBwdPlan:
    """Plan ``req`` on the dense route of ``arch``.

    ``spec``: a dense spec to use instead of the tuning hook's (it must serve
    the request); ``runtime``: runtime and aux values (``RUNTIME_KNOBS``,
    ``AUX_KNOBS``, ``num_persistent``, and the dQ kernel's ``DQ_KNOBS`` of a
    split spec) used with ``spec`` (or on top of the hook's). Raises ``ValueError`` when the predicate declines, the request
    is not dense-eligible, or a value is illegal.
    """
    policy = shipped_policy() if policy is None else policy
    verdict = attn_bwd_support(req, arch, policy=policy)
    if not verdict.ok:
        raise ValueError(f"{verdict.code}: {verdict.detail}")
    ok, tag = dense_bwd_eligibility(req, arch)
    if not ok:
        raise ValueError(f"request is not dense-eligible: {tag}")
    pclass = attn_bwd_perf_class(req, arch, "dense")
    tuned = (
        {}
        if spec is not None
        else dict(policy.tuning(req, arch, "dense", pclass) or {})
    )
    tuned.update(runtime or {})
    runtime_in = {k: tuned.pop(k) for k in RUNTIME_KNOBS if k in tuned}
    aux_in = {k: tuned.pop(k) for k in AUX_KNOBS if k in tuned}
    dense_rt = {k: tuned.pop(k) for k in DENSE_RUNTIME_KNOBS if k in tuned}
    # the dense dQ kernel's values (dq_mode = "split" only), as in the general
    # plan; with a given spec they ride in ``runtime``
    dq_in = {k: tuned.pop(k) for k in DQ_KNOBS if k in tuned}
    placement = runtime_in.get("scale_placement", "fold_ds")
    if placement not in SCALE_PLACEMENTS:
        raise ValueError(f"scale_placement={placement!r} not in {SCALE_PLACEMENTS}")
    lb_name = runtime_in.get("lb_order", "natural")
    if lb_name not in LB_ORDERS:
        raise ValueError(f"lb_order={lb_name!r} not in {tuple(LB_ORDERS)}")
    if lb_name == "mirror":
        raise ValueError("mirror pairing is not built: one kv tile per CTA")
    from_hook = spec is None
    if from_hook:
        # Plan-derived body knobs, as in the general plan: the kernel is
        # compiled for the scale placement (its unit multipliers are not
        # applied) unless the hook pins "runtime", and the decode serves the
        # interleave order only when it is asked for.
        body_in = dict(tuned.pop("body_knobs", ()) or ())
        body_in.update({k: tuned.pop(k) for k in DENSE_BODY_KNOBS if k in tuned})
        body_in.setdefault("static_scale", placement)
        if lb_name == "interleave":
            body_in["lb_interleave"] = True
        tuned["body_knobs"] = tuple(sorted(body_in.items()))
        # dK / dV are atomic exactly when the head group is split (the
        # general plan's rule), so one knob set serves both routes
        if int(runtime_in.get("g_split", 1)) > 1:
            tuned.setdefault("dkv_mode", "atomic")
        spec = dense_bwd_spec_for(req, arch, tuned)
    elif tuned:
        raise ValueError(f"spec given: knobs {sorted(tuned)} are not runtime values")
    body = validate_attn_dense_bwd_spec(spec, arch)
    if body.static_scale not in (placement, "runtime"):
        raise ValueError(
            f"static_scale={body.static_scale!r} does not match "
            f"scale_placement={placement!r}"
        )
    if lb_name == "interleave" and not body.lb_interleave:
        raise ValueError("lb_order='interleave' needs a spec with lb_interleave=True")
    want = dense_bwd_spec_for(req, arch, {})
    for fname in (
        "seqlen_q",
        "seqlen_kv",
        "num_query_heads",
        "num_kv_heads",
        "head_size",
        "dtype",
        "layout",
    ):
        if getattr(spec, fname) != getattr(want, fname):
            raise ValueError(f"dense spec {fname} does not match the request")
    if spec.band() != want.band():
        raise ValueError(f"dense spec band {spec.band()} != request {want.band()}")
    if spec.batch is not None and spec.batch != req.b:
        raise ValueError(f"dense spec pins batch={spec.batch}, request has {req.b}")

    g_split = int(runtime_in.get("g_split", 1))
    group = req.h_q // req.h_k
    if g_split < 1 or group % g_split:
        raise ValueError(f"g_split={g_split} must divide the head group {group}")
    if (g_split > 1) != (spec.dkv_mode == "atomic"):
        raise ValueError(
            f"g_split={g_split} needs dkv_mode='atomic' exactly when g_split > 1 "
            f"(spec has {spec.dkv_mode!r})"
        )
    if runtime_in.get("use_worklist"):
        raise ValueError("the work-list role is not built")
    n_kv_tiles = math.ceil(req.s_kv / body.block_n)
    units = req.h_k * g_split
    xcd_n = xcd_chunk = 0
    chunk_in = runtime_in.get("xcd_chunk", 0)
    if chunk_in:
        if not req.num_xcds:
            raise ValueError("xcd_chunk needs num_xcds")
        gx = n_kv_tiles
        # symbolic multiples of the grid x (the registry's sweep values)
        chunk_in = {"gx": gx, "2gx": 2 * gx, "4gx": 4 * gx}.get(chunk_in, chunk_in)
        xcd_chunk = int(chunk_in)
        if xcd_chunk not in (gx, 2 * gx, 4 * gx, 64):
            raise ValueError(f"xcd_chunk={xcd_chunk} not in (gx, 2*gx, 4*gx, 64)")
        xcd_n = int(req.num_xcds)
        if not spec.xcd_remap:
            if not from_hook:
                raise ValueError("xcd_chunk needs a spec with xcd_remap=True")
            # only a kernel built with the remap reads xcd_n / xcd_chunk
            spec = dataclasses.replace(spec, xcd_remap=True)
            body = validate_attn_dense_bwd_spec(spec, arch)
    num_persistent = 0
    if spec.persistent:
        num_persistent = int(dense_rt.get("num_persistent") or req.num_cus or 0)
        if num_persistent < 1:
            raise ValueError("a persistent dense plan needs num_persistent or num_cus")
    elif dense_rt.get("num_persistent"):
        raise ValueError("num_persistent needs a spec with persistent=True")
    split = spec.dq_mode == "split"
    if dq_in and not split:
        raise ValueError(f"dQ kernel knobs {sorted(dq_in)} need dq_mode='split'")
    dq_spec = dq_grid = None
    dq_xcd_n = dq_xcd_chunk = 0
    if split:
        # The dense dQ kernel: the general dQ kernel's tuning values on this
        # problem (its scale placement follows the main kernel's).
        dq_tuning = {k[3:]: v for k, v in dq_in.items() if k != "dq_xcd_chunk"}
        dq_spec = AttentionDenseBwdDqSpec.of(spec, body.dq_spec(**dq_tuning))
        resolved = validate_attn_dense_bwd_dq_spec(dq_spec, arch)
        dq_spec = dataclasses.replace(dq_spec, dq=resolved)
        dq_grid = attn_dense_bwd_dq_grid(dq_spec, arch, batch=req.b)
        if dq_grid[1] > _GRID_YZ_MAX or dq_grid[0] >= 2**31:
            raise ValueError(f"GRID_LIMIT: dQ grid {dq_grid}")
        chunk_in = dq_in.get("dq_xcd_chunk", 0)
        if req.num_xcds:
            gxd = dq_grid[0]
            chunk_in = {"gx": gxd, "2gx": 2 * gxd, "4gx": 4 * gxd}.get(
                chunk_in, chunk_in
            )
            dq_xcd_chunk = int(chunk_in) or 0
            if dq_xcd_chunk and dq_xcd_chunk not in (gxd, 2 * gxd, 4 * gxd, 64):
                raise ValueError(
                    f"dq_xcd_chunk={dq_xcd_chunk} not in (gx, 2*gx, 4*gx, 64)"
                )
            dq_xcd_n = int(req.num_xcds) if dq_xcd_chunk else 0
        elif chunk_in:
            raise ValueError("dq_xcd_chunk needs num_xcds")
        if dq_xcd_n and not dq_spec.dq.xcd_remap:
            # only a kernel built with the remap reads xcd_n / xcd_chunk
            dq_spec = dataclasses.replace(
                dq_spec, dq=dataclasses.replace(dq_spec.dq, xcd_remap=True)
            )
            validate_attn_dense_bwd_dq_spec(dq_spec, arch)

    # The general plan supplies prep, the converts, the workspace and the
    # multipliers (its own main launch is replaced). The predicate verdict
    # above is the request's own; the inner plan only needs a permissive
    # policy with the same aux / runtime values.
    inner_knobs = dict(aux_in)
    inner_knobs["ws_layout"] = spec.ws_layout
    inner_knobs["g_split"] = g_split
    if "scale_placement" in runtime_in:
        inner_knobs["scale_placement"] = runtime_in["scale_placement"]
    if split:
        # prep without the dQ zero fill, no WS_DQ, no dQ convert; the general
        # dQ launch supplies the dQ kernel's multipliers (the convert's folded
        # into dq_mult) and is replaced by the dense one
        inner_knobs["dq_mode"] = "split"

    def inner_tuning(_req, _arch, _route, _pclass):
        return dict(inner_knobs)

    inner = AttnBwdPolicy(
        declared_groups=frozenset(FEATURE_GROUPS),
        waive_unverified=True,
        tuning=inner_tuning,
        not_validated_archs=frozenset(),
    )
    general = attn_bwd_plan(req, arch, policy=inner)
    main = next(s for s in general.launches if s.stage == "main")
    scalars = {
        k: main.scalars[k] for k in ("scale_log2", "ds_mult", "dk_mult", "dq_mult")
    }
    unit = static_unit_mults(body.static_scale, body.dkv_mode)
    if any(scalars[n] != 1.0 for n in unit):
        raise AssertionError("a multiplier the dense kernel does not apply is not 1.0")
    scalars.update(
        batch=req.b,
        g_split=g_split,
        lb_order=LB_ORDERS[lb_name],
        pair=0,
        xcd_n=xcd_n,
        xcd_chunk=xcd_chunk,
        num_persistent=num_persistent,
    )
    atomic = spec.dkv_mode == "atomic"
    layout = general.workspace_layout
    pointers = {
        "Q": "q",
        "K": "k",
        "V": "v",
        "O": "o",
        "dO": "do",
        "LSE": "lse",
        "dK": None if atomic else "dk",
        "dV": None if atomic else "dv",
        "WS_LSE2": "WS_LSE2",
        "WS_DSUM": "WS_DSUM",
        "WS_DQ": "WS_DQ" if "WS_DQ" in layout else None,
        "WS_DK": "WS_DK" if atomic else None,
        "WS_DV": "WS_DV" if atomic else None,
    }
    if atomic and ("WS_DK" not in layout or "WS_DV" not in layout):
        raise AssertionError("atomic dense plan without a kv workspace")
    facts = attn_bwd_arch_facts(arch)
    grid = (num_persistent, 1, 1) if spec.persistent else (n_kv_tiles, units, req.b)
    dense = LaunchStep(
        "dense",
        spec.kernel_name(),
        grid,
        body.waves * facts.wave_size,
        _freeze(scalars),
        _freeze(pointers),
    )
    names = [n for n, _ in attn_dense_bwd_params(spec)]
    if sorted(list(dense.pointers) + list(dense.scalars)) != sorted(names):
        raise AssertionError("dense kernargs out of sync with the ABI")
    swap = {"main": dense}
    if split:
        gdq = next(s for s in general.launches if s.stage == "dq")
        dq_scalars = {k: gdq.scalars[k] for k in ("scale_log2", "ds_mult", "dq_mult")}
        if any(
            dq_scalars[n] != 1.0 for n in dq_static_unit_mults(dq_spec.dq.static_scale)
        ):
            raise AssertionError("a multiplier the dQ kernel does not apply is not 1.0")
        dq_scalars.update(batch=req.b, xcd_n=dq_xcd_n, xcd_chunk=dq_xcd_chunk)
        dq_ptrs = {
            "Q": "q",
            "K": "k",
            "V": "v",
            "dO": "do",
            "dQ": "dq",
            "WS_LSE2": "WS_LSE2",
            "WS_DSUM": "WS_DSUM",
        }
        swap["dq"] = LaunchStep(
            "dense_dq",
            dq_spec.kernel_name(),
            dq_grid,
            dq_spec.dq.waves * facts.wave_size,
            _freeze(dq_scalars),
            _freeze(dq_ptrs),
        )
        got = sorted(list(dq_ptrs) + list(dq_scalars))
        if got != sorted(n for n, _ in attn_dense_bwd_dq_params(dq_spec)):
            raise AssertionError("dense dQ kernargs out of sync with the ABI")
        dq_convert = any(
            s.stage == "convert" and s.pointers["SRC"] == "WS_DQ"
            for s in general.launches
        )
        if "WS_DQ" in layout or dq_convert:
            raise AssertionError("a split dense plan with a dQ workspace or convert")
    launches = tuple(swap.get(s.stage, s) for s in general.launches)
    specs = {k: v for k, v in general.specs.items() if k not in ("main", "dq")}
    specs["dense"] = spec
    if split:
        specs["dense_dq"] = dq_spec
    rt = dict(general.runtime)
    rt.update(
        lb_order=lb_name,
        xcd_n=xcd_n,
        xcd_chunk=xcd_chunk,
        pack_heads=0,
        num_persistent=num_persistent,
    )
    if split:
        rt.update(dq_xcd_n=dq_xcd_n, dq_xcd_chunk=dq_xcd_chunk)
    return dataclasses.replace(
        general,
        route="dense",
        route_tag=tag,
        perf_class=pclass,
        perf_status=policy.status_of(arch, "dense", pclass),
        fill_bucket=_fill_bucket(req, grid[0] * grid[1] * grid[2]),
        specs=_freeze(specs),
        runtime=_freeze(rt),
        launches=launches,
    )
