# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared stage-function plumbing for gfx950 mono families.

The device-function lowering is deliberately family-neutral.  A mono family
only has to provide its ABI, symbol namespace, the values that are uniform
when the function is inlined, and a layout loader.  Keeping this small layer
separate is important: changing the ML4 ABI must not silently change the
Qwen3.8 stage ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Mapping

from rocke.core.ir import I32, I64, IRBuilder


@dataclass(frozen=True)
class StageFamily:
    """Static inputs needed to build a family-specific stage function."""

    name: str
    abi_params: tuple[tuple[str, str], ...]
    symbol_prefix: str
    uniform_args: tuple[str, ...]
    layout_loader: Callable | None = None

    def symbol(self, stage: str, variant: str) -> str:
        return f"{self.symbol_prefix}{stage}_{variant}"

    def load_layout(self, layout=None, **kwargs):
        if layout is not None:
            return layout
        if self.layout_loader is None:
            raise ValueError(f"{self.name}: no default layout loader")
        return self.layout_loader(**kwargs)


def build_stage_kernel(
    spec,
    layout,
    *,
    family: StageFamily,
    stages: Mapping,
    load_stages: Callable[[str | None], None],
):
    """Build the common ABI prologue and invoke a registered stage body.

    ``stages`` maps a stage name to ``(body, supported)``.  Bodies receive the
    same compact namespace used by the original qwen38 implementation:
    ``tid``, ``lane``, ``wave`` and (where supplied by the family) ``ls``.
    """

    if spec.stage not in stages:
        load_stages(spec.stage)
    if spec.stage not in stages:
        raise KeyError(f"no rocKE stage function {spec.stage!r}; have {sorted(stages)}")
    body, supported = stages[spec.stage]
    if not supported(spec, layout):
        raise ValueError(f"{spec.stage} has no rocKE function for variant {spec.variant}")

    symbol = family.symbol(spec.stage, spec.variant)
    b = IRBuilder(symbol)
    b.kernel.attrs["max_workgroup_size"] = layout.threads
    b.fence_barriers = "compiler"
    vals = {}
    for name, typ in family.abi_params:
        vals[name] = b.param(name, I32 if typ == "i32" else I64)

    tid = b.thread_id_x()
    for name in family.uniform_args:
        if name in vals:
            vals[name] = b.readfirstlane(vals[name])

    a = SimpleNamespace(
        **vals,
        tid=tid,
        lane=b.land(tid, b.const_i32(63)),
        wave=b.readfirstlane(b.lshr(tid, b.const_i32(6))),
    )
    if all(name in vals for name in ("l0", "l1", "l2", "l3")):
        a.ls = [vals[f"l{i}"] for i in range(4)]
    body(b, a, layout, spec)
    b.ret()
    return b.kernel


__all__ = ["StageFamily", "build_stage_kernel"]
