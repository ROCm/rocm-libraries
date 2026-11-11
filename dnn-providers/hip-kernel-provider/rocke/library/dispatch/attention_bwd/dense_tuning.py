# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tuning registry and tuned table of the dense attention backward.

Every dense-only knob of
:class:`~kernels.common.attention_dense_bwd_spec.AttentionDenseBwdSpec` (and
of its per-arch subclasses) is one :class:`~.tuning_common.KnobAxis`, with
the same conventions as the general family's registry: ``choices[0]`` is the
default, values the emission check refuses are ``not_built`` (registered,
never emitted). The shared body's knobs ride in the dense spec's
``body_knobs`` and are the general registry's axes
(:data:`~.tuning_common.BWD_KNOB_AXES`); :func:`dense_body_axes` lists the
ones the dense family sweeps (every general axis except the fields the dense
spec owns itself).

:data:`DENSE_BWD_TUNED` is the tuned dense table: ``{(arch, dense key):
knobs}`` with the dense key of
:func:`kernels.common.attention_bwd_plan._dense_key`. A request routes to the
dense family only when its exact key is in this table (and passed its gate);
a key enters only after its performance gate passed, so the registry alone
changes no routing. It joins the per-arch tables (:mod:`.gfx942_tuning`,
:mod:`.gfx950_tuning`: ``DENSE_TUNED``).

``dq_mode = "split"`` is built on the dense route: the dense main kernel
computes dK / dV only and the dense split dQ kernel
(:mod:`kernels.common.attention_dense_bwd_dq`) takes the general dQ kernel's
values (:func:`dense_dq_axes`, the plan's ``DQ_KNOBS``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from kernels.common.attention_dense_bwd_spec import (
    DENSE_BODY_KNOBS,
    DENSE_DKV_MODES,
    DENSE_EPILOGUES,
    DENSE_OOB_MODES,
    DENSE_PERSIST_DECODES,
    AttentionDenseBwdSpec,
    _shell_not_built,
    dense_bwd_not_built,
    validate_attn_dense_bwd_spec,
)
from kernels.common.attention_bwd import WS_LAYOUTS, _not_built
from kernels.common.attention_bwd_plan import AUX_KNOBS, DQ_KNOBS, RUNTIME_KNOBS

from . import gfx942_tuning, gfx950_tuning
from .tuning_common import (
    BWD_KNOB_AXES,
    KIND_RUNTIME,
    KnobAxis,
    backward_axis,
    check_known_wrong,
    known_wrong_findings,
)

__all__ = [
    "DENSE_BWD_KNOB_AXES",
    "DENSE_BWD_TUNED",
    "dense_axis",
    "dense_body_axes",
    "dense_dq_axes",
    "dense_known_wrong",
    "dense_runtime_axes",
    "dense_spec_legality",
    "dense_tuning_axes",
]

_BODY_VARIANTS = {"gfx942": ("general", "xt"), "gfx950": ("general", "trload")}


def _persistent_only(value, knobs: Mapping[str, Any]) -> bool:
    return value == "kv_tile_major" or bool(knobs.get("persistent"))


def _req_dq_mode(value, knobs: Mapping[str, Any]) -> bool:
    # the dense main kernel is the shared body: the general family's split
    # couplings (looked up at call time, so a coupling change there holds here)
    return backward_axis("dq_mode").applies(value, knobs)


DENSE_BWD_KNOB_AXES: tuple[KnobAxis, ...] = (
    KnobAxis(
        "body_variant",
        ("general", "xt", "trload"),
        family=3,
        stage="structure",
        numerics="tile",
        not_built=("xt", "trload"),
        arch_values=lambda arch: _BODY_VARIANTS.get(arch, ("general",)),
        note="per-arch bodies (gfx942 XT, gfx950 trload) are registered, not built",
    ),
    KnobAxis(
        "oob",
        DENSE_OOB_MODES,
        family=3,
        stage="structure",
        note="buffer range check (zero rows) or row clamp; identical dK / dV",
    ),
    KnobAxis("edge_tiles", (True, False), family=6, stage="codegen"),
    KnobAxis("epilogue", DENSE_EPILOGUES, family=6, stage="codegen"),
    KnobAxis("ws_layout", WS_LAYOUTS, family=6, stage="codegen"),
    KnobAxis(
        "dkv_mode",
        DENSE_DKV_MODES,
        family=7,
        stage="runtime",
        numerics="reorders_fp",
        plan_owned=True,
        note="atomic exactly when the runtime g_split > 1",
    ),
    KnobAxis("persistent", (False, True), family=7, stage="runtime"),
    KnobAxis(
        "persist_decode",
        DENSE_PERSIST_DECODES,
        family=7,
        stage="runtime",
        requires=_persistent_only,
    ),
    KnobAxis(
        "xcd_remap",
        (False, True),
        family=7,
        stage="runtime",
        plan_owned=True,
        note="set by the plan when num_xcds and xcd_chunk give xcd_n > 0 "
        "(whole-grid remap of the dense grid)",
    ),
    KnobAxis(
        "dq_mode",
        ("atomic", "split"),
        family=8,
        stage="structure",
        numerics="reorders_fp",
        not_built=backward_axis("dq_mode").not_built,
        requires=_req_dq_mode,
        note="'split': the dense main kernel computes dK / dV only and the dense "
        "Q-parallel dQ kernel dQ (the general dQ kernel's dq_* values); the "
        "split main kernel's couplings are the general family's",
    ),
    KnobAxis(
        "num_persistent",
        (None,),
        family=7,
        stage="runtime",
        kind=KIND_RUNTIME,
        derived="device-fill multiples of num_cus (default num_cus)",
    ),
)

_BY_NAME = {ax.name: ax for ax in DENSE_BWD_KNOB_AXES}
# Runtime plan values and aux values the dense plan shares with the general
# family (the work list is general-only).
_SHARED_RUNTIME = tuple(k for k in RUNTIME_KNOBS if k != "use_worklist") + tuple(
    AUX_KNOBS
)
# The split dQ kernel's values (dq_mode = "split"): the general family's dQ
# axes, taken by the dense dQ kernel (its spec carries them).
_SHARED_DQ = tuple(DQ_KNOBS)

# The tuned dense table: the per-arch tables (dispatch.attention_bwd.<arch>
# _tuning; a dense key enters only after its performance gate passed).
DENSE_BWD_TUNED: Mapping[tuple, Mapping[str, Any]] = {
    **gfx942_tuning.DENSE_TUNED,
    **gfx950_tuning.DENSE_TUNED,
}


def dense_axis(name: str) -> KnobAxis:
    """The dense axis ``name`` (a dense-only knob), or the general axis of a
    body knob or of a shared runtime / aux value."""
    if name in _BY_NAME:
        return _BY_NAME[name]
    if name in DENSE_BODY_KNOBS or name in _SHARED_RUNTIME or name in _SHARED_DQ:
        return backward_axis(name)
    raise KeyError(f"no dense backward knob {name!r}")


def dense_body_axes() -> tuple[KnobAxis, ...]:
    """General registry axes the dense family sweeps through ``body_knobs``."""
    return tuple(ax for ax in BWD_KNOB_AXES if ax.name in DENSE_BODY_KNOBS)


def dense_runtime_axes() -> tuple[KnobAxis, ...]:
    """General runtime / aux axes the dense plan accepts (``g_split``,
    ``lb_order``, ``xcd_chunk``, ``scale_placement`` and the aux values)."""
    return tuple(backward_axis(n) for n in _SHARED_RUNTIME)


def dense_dq_axes() -> tuple[KnobAxis, ...]:
    """General dQ-kernel axes the dense split dQ kernel takes (``dq_*``;
    ``dq_mode = "split"`` only)."""
    return tuple(backward_axis(n) for n in _SHARED_DQ)


def dense_tuning_axes(arch: str, level: str = "production") -> tuple[KnobAxis, ...]:
    """Dense-only axes plus the body, shared runtime and dQ-kernel axes, each
    with values on ``arch``."""
    out = []
    axes = DENSE_BWD_KNOB_AXES + dense_body_axes() + dense_runtime_axes()
    for ax in axes + dense_dq_axes():
        if ax.level == "full" and level != "full":
            continue
        if ax.values(arch, level):
            out.append(ax)
    return tuple(out)


def dense_spec_legality(spec: AttentionDenseBwdSpec, arch: str) -> str:
    """``"legal"``, ``"not_built"`` or ``"illegal"`` (the kernel's own rules;
    a known-wrong value of the dense spec or of its resolved body,
    :data:`~.tuning_common.BWD_KNOWN_WRONG_KNOBS`, is illegal)."""
    try:
        body = validate_attn_dense_bwd_spec(spec, arch)
    except ValueError:
        return "illegal"
    if dense_known_wrong(spec, body, arch):
        return "illegal"
    if dense_bwd_not_built(spec) or _not_built(body) or _shell_not_built(body):
        return "not_built"
    return "legal"


def dense_known_wrong(spec: AttentionDenseBwdSpec, body, arch: str) -> tuple:
    """Known-wrong findings (``"knob=value"``) of a dense spec's own fields
    and of its resolved body spec ``body`` on ``arch``."""
    own = {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec)}
    own["body_variant"] = getattr(spec, "body_variant", None)
    shared = {f.name: getattr(body, f.name) for f in dataclasses.fields(body)}
    return tuple(
        sorted(
            set(known_wrong_findings(arch, own) + known_wrong_findings(arch, shared))
        )
    )


def _check_registry() -> None:
    """Every dense tuning field has exactly one axis (raised at import)."""
    problem = {
        "seqlen_q",
        "seqlen_kv",
        "num_query_heads",
        "num_kv_heads",
        "head_size",
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
    }
    spec_fields = {f.name for f in dataclasses.fields(AttentionDenseBwdSpec)}
    tuning = (spec_fields - problem) | {"body_variant"}
    compile_axes = {ax.name for ax in DENSE_BWD_KNOB_AXES if ax.kind != KIND_RUNTIME}
    if tuning != compile_axes:
        raise AssertionError(
            f"dense spec tuning fields {sorted(tuning)} != dense compile axes "
            f"{sorted(compile_axes)}"
        )
    for ax in DENSE_BWD_KNOB_AXES:
        for v in ax.not_built:
            if v not in ax.choices or v == ax.default:
                raise AssertionError(
                    f"{ax.name}: not_built value {v!r} is not a choice"
                )
    missing = set(DENSE_BODY_KNOBS) - {ax.name for ax in BWD_KNOB_AXES}
    if missing:
        raise AssertionError(f"body knobs without a general axis: {sorted(missing)}")
    check_known_wrong(_BY_NAME)


_check_registry()
