# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LDS access tracing and cross-wavefront hazard analysis for backward kernels.

The tracer instruments :class:`LdsView` while a backward kernel is built and
records the address interval, access direction, and lane-private status of each
LDS operation.  The hazard model then expands zero and two loop trips and
requires a workgroup-wide LDS-ordering barrier between overlapping accesses that can
race across wavefronts.

This is test support rather than a pytest module: focused tests import the model
without collecting or constructing the exhaustive backward-IR test matrices.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias

from rocke.core.ir import KernelDef, Op

from kernels.common.attention_bwd import AttnBwdSpec, build_attn_bwd_main

__all__ = ["LDS_BARRIERS", "is_lds_access", "lds_hazards", "traced_spec"]

_LdsTag: TypeAlias = tuple[int, int, bool, bool]
_LdsTags: TypeAlias = dict[int, _LdsTag]
_LdsHazard: TypeAlias = tuple[str, int, str, int, int]


class _KernelBody(Protocol):
    ops: Sequence[Op]


class _KernelLike(Protocol):
    body: _KernelBody


LDS_BARRIERS = ("tile.sync", "tile.sync_lds_only")
_LDS_PREFIXES = ("tile.smem_", "tile.ds_read")


def is_lds_access(op: Op) -> bool:
    """Whether ``op`` is an LDS access emitted by an ``LdsView`` method."""
    return op.name.startswith(_LDS_PREFIXES) and op.name != "tile.smem_alloc"


def traced_spec(
    spec: AttnBwdSpec, arch: str, *, structural: bool = False
) -> tuple[KernelDef, _LdsTags]:
    """Build ``spec`` and tag every LDS operation emitted through ``LdsView``.

    ``structural`` bypasses only the main kernel's recorded resource refusals,
    allowing tests to inspect emission structure for a spec that a production
    compiler refuses. Every static builder rule still applies.

    Instrumentation temporarily patches process-global emitter methods. Calls
    must not overlap within one process; pytest-xdist workers are separate
    processes and remain safe.
    """
    from kernels.common import _attention_bwd_body as body
    from kernels.common._attention_bwd_frag import LdsView
    from kernels.common.attention_bwd import _emit_main

    tags, state = {}, {"private": False}
    original = {name: getattr(LdsView, name) for name in ("load", "store", "tr16_load")}
    original_acc = {
        name: getattr(body.BwdTileStep, name)
        for name in ("_load_lds_acc", "_store_lds_acc")
    }

    def tagged(name: str):
        def call(self, builder, *args, **kwargs):
            ops = builder._region_stack[-1].ops
            first = len(ops)
            result = original[name](self, builder, *args, **kwargs)
            emitted = [op for op in ops[first:] if is_lds_access(op)]
            assert emitted, [op.name for op in ops[first:]]
            high = self.base + self.rows * self.pitch * self.scale
            for op in emitted:
                tags[id(op)] = (
                    self.base,
                    high,
                    name == "store",
                    state["private"],
                )
            return result

        return call

    def private(name: str):
        def call(self, *args, **kwargs):
            state["private"] = True
            try:
                return original_acc[name](self, *args, **kwargs)
            finally:
                state["private"] = False

        return call

    try:
        for name in original:
            setattr(LdsView, name, tagged(name))
        for name in original_acc:
            setattr(body.BwdTileStep, name, private(name))
        if structural:
            kernel = _emit_main(spec, arch, main_kernel_refusals=False)
        else:
            kernel = build_attn_bwd_main(spec, arch=arch)
    finally:
        for name, function in original.items():
            setattr(LdsView, name, function)
        for name, function in original_acc.items():
            setattr(body.BwdTileStep, name, function)
    return kernel, tags


def _events(
    ops: Sequence[Op],
    tags: _LdsTags,
    *,
    trips: int,
    barriers: Sequence[str],
    in_branch: bool = False,
) -> list[tuple[str, Op]]:
    """Return program-order LDS and barrier events for one loop expansion."""
    out = []
    for op in ops:
        if op.name in barriers:
            if not in_branch:
                out.append(("sync", op))
        elif is_lds_access(op):
            assert id(op) in tags, "an LDS access outside the LdsView methods"
            out.append(("lds", op))
        elif op.name == "scf.for":
            for _ in range(trips):
                out += _events(
                    op.regions[0].ops,
                    tags,
                    trips=trips,
                    barriers=barriers,
                    in_branch=in_branch,
                )
        else:
            for region in op.regions:
                out += _events(
                    region.ops,
                    tags,
                    trips=trips,
                    barriers=barriers,
                    in_branch=True,
                )
    return out


def lds_hazards(
    kernel: _KernelLike,
    tags: _LdsTags,
    *,
    barriers: Sequence[str] = LDS_BARRIERS,
) -> list[_LdsHazard]:
    """Return pairs of LDS accesses that can race across wavefronts.

    Accesses race when their views overlap, no barrier in ``barriers`` lies
    between them on a zero-trip path or a two-iteration wrap, they are not both
    lane-private accumulator updates, and at least one is a write through a
    distinct view.
    """
    found = []
    for trips in (0, 2):
        events = _events(kernel.body.ops, tags, trips=trips, barriers=barriers)
        open_accesses = []
        for kind, op in events:
            if kind == "sync":
                open_accesses = []
                continue
            low, high, store, private = tags[id(op)]
            for previous in open_accesses:
                prev_low, prev_high, prev_store, prev_private = tags[id(previous)]
                if not (low < prev_high and prev_low < high) or (
                    private and prev_private
                ):
                    continue
                if (store != prev_store) or (
                    store and (low, high) != (prev_low, prev_high)
                ):
                    found.append((previous.name, prev_low, op.name, low, trips))
            open_accesses.append(op)
    return found
