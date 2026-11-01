# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A two-stage test chain for the persistent platform (WP1 hardware gate).

Exact int32 math, so the same layers run as Path B (``layers=1`` kernels
launched once per layer, :func:`build_static`) and as Path A (one launch of
the loop driver, :func:`build_loop`) must give bit-identical outputs, and both
must equal :func:`reference`.

Per layer ``l`` (arguments ``scratch``, ``w`` = ``W_l[16][64]`` i32,
``h_out`` = ``H_l[64]`` i32):

* stage ``g`` (16 tasks, CTAs 0..15): wave 0's lane ``j`` polls pair ``j`` of
  region ``hx`` (the previous layer's output, tag = this layer's tag - 1:
  ``launch_tag(step, l - 1)``), task ``t`` puts ``y_t = sum_j x_j W_l[t][j] +
  t`` (wrapping) to pair ``t`` of ``y``;
* stage ``r`` (1 task, CTA 16): lanes 0..15 poll ``y``, ``s = sum_t y_t (t +
  1)``; lane ``j`` puts ``h_j = s (j + 1) ^ j`` to pair ``j`` of ``hx`` and
  stores it to ``H_l[j]``.

The host seeds ``hx`` (the slot layer ``layer0`` reads: 0 for static
builds, 1 for the loop) with ``x0`` under ``launch_tag(step, layer0 - 1)``.
Mutant ``prevtag``: ``g`` polls with the current layer's tag (never written
before ``r`` runs: a broken tag invariant) and must time out through a
deadline edge (``EdgeKnobs(poll="deadline")``), setting the launch-fatal word.
Its numerics are NOT a reliable verdict in either path: a timed-out poll
returns the stale ``hx`` payload, the previous layer's correct output, so
the layer is wrong only when ``r`` times out before the ``g`` waves (a race
between equal deadlines; WP1 verification and fix round). The verdict is the
launch-fatal word. Mutant ``wrongw``: task ``t`` multiplies by weight row
``(t + 1) % 16`` (no protocol change, no timeout): numerics must fail
deterministically in both paths.

Kernel names start with ``q38pt_``; nothing here is imported by a served
build.
"""

from __future__ import annotations

import numpy as np

from rocke.core.ir import I32, Value
from rocke.helpers import device_ops as dops
from rocke.helpers import mailbox as mbx
from rocke.helpers.task_schedule import Dep, Stage, TaskTable

from ..framework import FrameworkKnobs, ScratchLayout, StageSpec, build_layer_kernel
from .driver import build_persistent_kernel

__all__ = ["ARGS", "G_TASKS", "HX", "build_loop", "build_static", "isa_builds_knob", "isa_builds_ref", "layout", "reference", "table"]

ARGS = ("scratch", "w", "h_out")
ARGS_ERR = ("scratch", "w", "h_out", "err")
G_TASKS = 16
HX = 64
R_CTA = 16
MUTANTS = ("none", "prevtag", "wrongw")


def layout(parity_slots: int) -> ScratchLayout:
    return ScratchLayout.of({"hx": HX, "y": G_TASKS}, parity_slots=parity_slots)


def table(stack: bool) -> TaskTable:
    g = Stage("g", G_TASKS, base=0, deps=(Dep("r", layer=-1),) if stack else (), reads=("hx",), writes=("y",))
    r = Stage("r", 1, base=R_CTA, deps=(Dep("g"),), reads=("y",), writes=("hx",))
    return TaskTable((g, r), grid=256, inputs=("hx",), parity=("hx", "y") if stack else (), kind="ptoy")


def _prev_slot(ctx):
    s = ctx.slot
    return ctx.b.xor(s, ctx.b.const_i32(1)) if isinstance(s, Value) else 0


def _wsum(b, v, n: int) -> Value:
    """sum over lanes 0..n-1 of ``v`` (uniform i32, wrapping)."""
    acc = b.readlane(v, b.const_i32(0))
    for j in range(1, n):
        acc = b.add(acc, b.readlane(v, b.const_i32(j)))
    return acc


def _emit_g(mutant: str):
    def emit(b, ctx, p, t):
        with b.scf_if(dops.is_wave0(b)):
            lane = b.lane_id()
            tag = ctx.tag if mutant == "prevtag" else b.sub(ctx.tag, b.const_i32(1))
            (xs,) = ctx.poll([(ctx.mb("hx", slot=_prev_slot(ctx)), lane, 1)], tag)
            x = xs[0]
            row = b.mod(b.add(t, b.const_i32(1)), b.const_i32(G_TASKS)) if mutant == "wrongw" else t
            w = b.global_load_i32(p.ptr(b, "w"), b.add(b.mul(row, b.const_i32(HX)), lane))
            y = b.add(_wsum(b, b.mul(x, w), HX), t)
            with b.scf_if(b.cmp_eq(lane, b.const_i32(0))):
                ctx.put(ctx.mb("y"), t, y)

    return emit


def _emit_r(b, ctx, p, t):
    with b.scf_if(dops.is_wave0(b)):
        lane = b.lane_id()
        li = b.select(b.cmp_lt(lane, b.const_i32(G_TASKS)), lane, b.const_i32(0))
        (ys,) = ctx.poll([(ctx.mb("y"), li, 1)])
        yv = b.select(b.cmp_lt(lane, b.const_i32(G_TASKS)), ys[0], b.const_i32(0))
        s = _wsum(b, b.mul(yv, b.add(lane, b.const_i32(1))), G_TASKS)
        h = b.xor(b.mul(s, b.add(lane, b.const_i32(1))), lane)
        ctx.put(ctx.mb("hx"), lane, h)
        b.global_store(p.ptr(b, "h_out"), lane, h)


def _specs(stack: bool, mutant: str) -> list[StageSpec]:
    tb = table(stack)
    return [StageSpec(tb.stage("g"), _emit_g(mutant)), StageSpec(tb.stage("r"), _emit_r)]


def build_static(knobs: FrameworkKnobs = FrameworkKnobs(), mutant: str = "none"):  # noqa: B008
    """The Path B unit: one layer per launch (``layers=1``)."""
    if mutant not in MUTANTS:
        raise ValueError(mutant)
    args = ARGS_ERR if knobs.err_arg else ARGS
    name = "q38pt_static" + (f"_mut{mutant}" if mutant != "none" else "")
    return build_layer_kernel(name, _specs(False, mutant), table(False), layout(1), knobs=knobs, args=args)


def build_loop(knobs: FrameworkKnobs | None = None, mutant: str = "none"):
    """The Path A unit: ``nlayers`` layers in one launch (the loop driver)."""
    if mutant not in MUTANTS:
        raise ValueError(mutant)
    name = "q38pt" + (f"_mut{mutant}" if mutant != "none" else "")
    return build_persistent_kernel(name, _specs(True, mutant), table(True), layout(2), knobs=knobs, args=ARGS)


def reference(x0: np.ndarray, ws: list[np.ndarray]) -> list[np.ndarray]:
    """Host twin: per layer ``H_l`` (int32, wrapping)."""
    x = x0.astype(np.int64)
    out = []
    for w in ws:
        y = ((w.astype(np.int64) * x[None, :]).sum(axis=1) + np.arange(G_TASKS)) & 0xFFFFFFFF
        s = int((y * (np.arange(G_TASKS) + 1)).sum()) & 0xFFFFFFFF
        j = np.arange(HX, dtype=np.int64)
        h = ((s * (j + 1)) & 0xFFFFFFFF) ^ j
        out.append(h.astype(np.uint32).view(np.int32))
        x = h.astype(np.int64)
    return out


def launch_tag(step: int, layer: int) -> int:
    """``mailbox.launch_tag`` on the host (i32 wrap)."""
    v = (step * mbx.LAYER_SLOTS + layer + 1) & 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


I32_ = I32  # re-export for the bench


def isa_builds_ref() -> dict:
    """``persist.isa_gate`` build set (REF): the static and loop units."""
    from .knobs import D0_DEFAULTS

    return {
        "static": ("ptoy_static", lambda: build_static()),
        "loop": ("ptoy_loop", lambda: build_loop(FrameworkKnobs(persist=D0_DEFAULTS))),
    }


def isa_builds_knob() -> dict:
    """The same names with a kernel-side knob changed in ``static`` (the ``irf``
    poll): the pooled ISA gate must FAIL exactly that build."""
    from .knobs import EdgeKnobs, EdgePlan

    out = isa_builds_ref()
    out["static"] = ("ptoy_static", lambda: build_static(FrameworkKnobs(edges=EdgePlan.of(default=EdgeKnobs(poll="irf")))))
    return out
