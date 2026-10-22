# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Hybrid stage functions as layer-framework stages (roadmap T9, path b).

A stage track writes its stage once, as a ``stage_fn`` body
(``stage_fn.register_stage(name, body, supported)``; ``body(b, a, layout,
spec)`` with ``a`` the stage-function ABI namespace: ``tag``, ``task``,
``step``, ``layer``, the 41 ``LAYER_ARGS`` as i64, ``l0..l3`` / ``ls``,
``tid``, ``lane``, ``wave``). The same body, unchanged, runs:

* in the FlyDSL mono kernel (hybrid path a): ``stage_fn`` lowers it as a
  device function;
* in a rocKE layer kernel (path b): :func:`stage_fn_factory` wraps it as a
  ``framework.StageSpec`` with ``per_cta=True`` (every CTA calls the body
  once with its round-0 task index, exactly the ABI's ``task``: ``>=`` the
  stage's tasks means no task, the body guards itself).

The ABI values come from the framework: ``tag`` = ``ctx.tag`` (the layer's
launch tag), ``step`` = the kernel's step pointer, ``layer`` = the runtime
layer id, the arguments = the kernel's (same 41 names and order),
``l0..l3`` = each QSA row's cache length (:func:`qsa_lens`, ATOM
``stages/qsa.qsa_len``; 0 in other kinds and past the rows, as the mono
passes them). In a kernel build the bodies allocate their LDS with
``smem_alloc`` (``lds="arg"`` applies only to device functions); the
lowering's liveness packing overlays the per-stage pools, and the framework
arena holds only the framework's own entries, so adapted builds use
:func:`adapted_knobs` (no one-CTA-per-CU padding of the framework arena:
check the lowered LDS size instead, :func:`lds_words`).

Placement: bodies in :data:`SELF_PLACED` derive their task from the block id
with ATOM's base (``moe.m4_task``) and ignore ``task``: they run only on
ATOM's placement (the factory refuses others).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import replace
from types import SimpleNamespace

from rocke.core.ir import I64, IRBuilder, Value
from rocke.helpers import device_ops as dops
from rocke.helpers.task_schedule import Stage

from .framework import FrameworkKnobs, LayerCtx, LayerShape, Params, StageSpec
from .mono_layout import LAYER_ARGS, MonoLayout

__all__ = [
    "SELF_PLACED",
    "adapted_knobs",
    "available_stage_fns",
    "import_errors",
    "lds_words",
    "resident_build",
    "qsa_lens",
    "register_stage_fns",
    "stage_fn_factory",
]

# stage-function bodies that place themselves (task from bid + ATOM's base)
SELF_PLACED = frozenset({"m3", "m4", "m5"})
# stage-function modules beyond ``stage_fn.STAGE_MODULES`` (the MoE one
# registers itself on import)
_EXTRA_MODULES = ("stage_moe",)
# stage module -> "ExcType: message" of its last failed import
_IMPORT_ERRORS: dict[str, str] = {}


def available_stage_fns() -> tuple[str, ...]:
    """Stage names with a registered stage-function body (imports every stage
    module). A module that fails to import (e.g. another track mid-edit) is
    left out, with a ``RuntimeWarning`` naming it and the error; the errors
    stay readable through :func:`import_errors` and are quoted by the
    ``KeyError`` of :func:`stage_fn_factory`."""
    import importlib

    from . import stage_fn as SF

    mods = sorted(set(SF.STAGE_MODULES.values())) + list(_EXTRA_MODULES)
    for m in mods:
        try:
            importlib.import_module(f"{__package__}.{m}")
        except Exception as e:  # noqa: BLE001 - another track's module
            msg = f"{type(e).__name__}: {e}"
            if _IMPORT_ERRORS.get(m) != msg:
                warnings.warn(f"stage module {m} failed to import, its stage functions are "
                              f"unavailable: {msg}", RuntimeWarning, stacklevel=2)  # fmt: skip
            _IMPORT_ERRORS[m] = msg
        else:
            _IMPORT_ERRORS.pop(m, None)
    return tuple(sorted(SF.STAGES))


def import_errors() -> dict[str, str]:
    """Stage modules whose last import (:func:`available_stage_fns`) failed:
    module -> ``"ExcType: message"``."""
    return dict(_IMPORT_ERRORS)


def qsa_lens(b: IRBuilder, ctx: LayerCtx, p: Params, layout: MonoLayout) -> list[Value]:
    """Each request row's QSA cache length (``l0..l3``), ATOM ``qsa_len``:
    row ``q``'s ``pos`` entry of the ``qsa`` state table (i64 entries,
    ``len(qsa_state)`` per row) holds the address of its length; seq builds
    read the request's row and add the token index. Loaded once per layer of
    the launch (cached on ``ctx``; the per-CTA stages run unguarded, so the
    values dominate the later stages)."""
    key = ("qsa_lens", ctx.li)
    if key in ctx.extra:
        return ctx.extra[key]
    rows, seq = ctx.rows, ctx.seq
    tbl = p.ptr(b, "qsa", pointee=I64)
    n = len(layout.qsa_state)
    pos = layout.state_index("pos")
    ls: list[Value] = []
    for q in range(1 if seq else rows):
        a64 = b.readfirstlane(b.global_load_i64(tbl, b.const_i32(pos + q * n)))
        ls.append(b.readfirstlane(b.global_load_i32(dops.ptr_from_i64(b, a64), b.const_i32(0))))
    if seq:
        ls += [b.add(ls[0], b.const_i32(q)) for q in range(1, rows)]
    ls += [b.const_i32(0)] * (4 - len(ls))
    ctx.extra[key] = ls
    return ls


def _abi(b: IRBuilder, ctx: LayerCtx, p: Params, t: Value, ls: list[Value]):
    vals = {"tag": ctx.tag, "task": t, "step": ctx.step_ptr, "layer": ctx.layer}
    for n in LAYER_ARGS:
        vals[n] = p.addr(n)
    for i in range(4):
        vals[f"l{i}"] = ls[i]
    vals["lds_base"] = b.const_i32(0)  # kernel builds: the body's own smem
    tid = ctx.tid
    return SimpleNamespace(
        **vals, tid=tid, lane=b.land(tid, b.const_i32(63)), wave=ctx.wave, ls=list(ls)
    )


def stage_fn_factory(name: str, layout: MonoLayout, *, knobs=None, ls=None):
    """A ``framework`` stage factory running stage function ``name``'s body
    unchanged. ``knobs``: ``stage_fn.StageKnobs`` (default the shipped ones).
    ``ls(b, ctx, p) -> [4 x i32]``: the QSA lengths (default :func:`qsa_lens`
    in ``qsa`` layers, zeros elsewhere)."""
    from . import stage_fn as SF

    if name not in SF.STAGES:
        available_stage_fns()
    if name not in SF.STAGES:
        errs = "".join(f"; {m} failed to import ({e})" for m, e in sorted(_IMPORT_ERRORS.items()))
        raise KeyError(f"no stage function {name!r}; have {sorted(SF.STAGES)}{errs}")
    body, supported = SF.STAGES[name]
    sk = knobs if knobs is not None else SF.StageKnobs()

    def factory(kind: str, shape: LayerShape, st: Stage) -> StageSpec:
        spec = SF.StageSpec(name, shape.rows, shape.seq, shape.group, sk)
        if not supported(spec, layout):
            raise ValueError(f"stage function {name} has no variant {spec.variant}")
        if name in SELF_PLACED:
            from .atom_layout import BASE

            if st.map != "affine" or st.base != BASE[name]:
                raise ValueError(f"{name} places itself (ATOM base): atom placement only")
        if st.tasks > layout.grid:
            raise ValueError(f"{name}: {st.tasks} tasks > grid {layout.grid}")

        def emit(b: IRBuilder, ctx: LayerCtx, p: Params, t: Value) -> None:
            if ls is not None:
                lv = list(ls(b, ctx, p))
            elif kind == "qsa":
                lv = qsa_lens(b, ctx, p, layout)
            else:
                lv = [b.const_i32(0)] * 4
            a = _abi(b, ctx, p, t, lv)
            old = b.fence_barriers
            b.fence_barriers = "compiler"  # as stage_fn.build_stage_kernel
            try:
                body(b, a, layout, spec)
            finally:
                b.fence_barriers = old

        return StageSpec(stage=st, emit=emit, per_cta=True)

    return factory


def register_stage_fns(layout: MonoLayout, names=None, **kw) -> tuple[str, ...]:
    """Register every stage function (or ``names``) as the framework's
    default emitter of its stage (``framework.register_stage``)."""
    from .framework import register_stage

    have = available_stage_fns()
    out = []
    for n in names or have:
        register_stage(n, stage_fn_factory(n, layout, **kw))
        out.append(n)
    return tuple(out)


def adapted_knobs(knobs: FrameworkKnobs = FrameworkKnobs()) -> FrameworkKnobs:  # noqa: B008
    """Framework knobs for builds with adapted stage functions: the framework
    arena is not padded to one CTA per CU (the bodies' own pools add to it)."""
    return replace(knobs, lds_min_words=0, stamps_lds="own")


def resident_build(build, knobs: FrameworkKnobs = FrameworkKnobs(), *,  # noqa: B008
                   arch: str = "gfx950", flavor: str | None = None):
    """``build(knobs) -> KernelDef`` (e.g. a ``build_layer`` partial) with the
    framework arena padded so the lowered kernel holds more than half a CU's
    LDS (one CTA per CU, the mono's residency) and at most all of it.

    The arena is one allocation whose live range is its first to last use;
    the bodies' pools overlay it when their ranges do not meet (no stamps)
    and add to it when they do, so the padding is searched: the largest
    ``lds_min_words`` <= the default that fits. Returns ``(kernel, words)``;
    raises when no padding fits."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.helpers.lds_arena import LDS_LIMIT_WORDS, ONE_CTA_PER_CU_WORDS

    def words(lmin):
        k = build(replace(knobs, lds_min_words=lmin, lds_touch=True, stamps_lds="own"))
        return k, lds_words(lower_kernel_to_llvm(k, arch=arch, llvm_flavor=flavor))

    k, w = words(knobs.lds_min_words)
    if ONE_CTA_PER_CU_WORDS <= w <= LDS_LIMIT_WORDS:
        return k, w
    k0, w0 = words(0)
    if w0 > LDS_LIMIT_WORDS:
        raise ValueError(f"{k0.name}: {w0} LDS words > {LDS_LIMIT_WORDS} without padding")
    if w0 >= ONE_CTA_PER_CU_WORDS:
        return k0, w0
    lo, hi, best = 0, knobs.lds_min_words, None  # largest fitting padding
    while lo <= hi:
        mid = (lo + hi) // 2 // 4 * 4
        km, wm = words(mid)
        if wm <= LDS_LIMIT_WORDS:
            if wm >= ONE_CTA_PER_CU_WORDS:
                best = (km, wm)
            lo = mid + 4
        else:
            hi = mid - 4
    if best is None:
        raise ValueError(f"{k0.name}: no arena padding gives one CTA per CU within the LDS")
    return best


def lds_words(ll: str) -> int:
    """LDS of a lowered kernel (``ll``: its LLVM IR text; one pool holding
    the framework arena and the bodies' liveness-packed allocations), in
    i32 words."""
    m = re.findall(r"@smem_pool\.\S+ = .*?\[(\d+) x i8\]", ll)
    return sum(int(x) for x in m) // 4
