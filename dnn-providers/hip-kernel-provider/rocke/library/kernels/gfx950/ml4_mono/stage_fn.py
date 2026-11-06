# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Family-parameterized ML4 rocKE stage-function ABI."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


from ..stage_fn_common import StageFamily, build_stage_kernel as _build_stage_kernel
from ..qwen38_mono.device_function import (
    DeviceFunctionKnobs,
    lower_device_function,
)
from .mono_layout import KERNEL_PARAMS, MonoLayout, from_aiter

ML4_FAMILY = StageFamily(
    name="ml4",
    abi_params=(
        ("tag", "i32"),
        ("task", "i32"),
        ("peer_slot", "i32"),
        *KERNEL_PARAMS,
    ),
    symbol_prefix="rocke_ml4_",
    uniform_args=("tag", "task", "peer_slot", "rank", "layer"),
    layout_loader=from_aiter,
)

# The lds_base parameter is added by lower_device_function for every
# DeviceFunctionKnobs(lds="arg") build, not by the stage family ABI itself.
ABI_PARAMS = ML4_FAMILY.abi_params
SYMBOL_PREFIX = ML4_FAMILY.symbol_prefix


def variant_name(rows: int, seq: bool = False, group: int = 0) -> str:
    return f"r{rows}" + ("s" if seq else "") + (f"g{group}" if group else "")


def parse_variant(value: str) -> tuple[int, bool, int]:
    m = re.fullmatch(r"r(\d+)(s?)(?:g(\d+))?", value)
    if not m:
        raise ValueError(f"bad variant {value!r} (r<rows>[s][g<S>])")
    return int(m.group(1)), bool(m.group(2)), int(m.group(3) or 0)


@dataclass(frozen=True)
class StageKnobs:
    mailbox_impl: str = "asm"
    fn: DeviceFunctionKnobs = field(
        default_factory=lambda: DeviceFunctionKnobs(lds="arg", inline="always")
    )

    def __post_init__(self):
        if self.mailbox_impl not in ("asm", "ir"):
            raise ValueError("mailbox_impl must be asm or ir")

    def suffix(self) -> str:
        parts = []
        if self.mailbox_impl != "asm":
            parts.append("mb" + self.mailbox_impl)
        defaults = DeviceFunctionKnobs(lds="arg", inline="always")
        if self.fn.lds != defaults.lds:
            parts.append("lds" + self.fn.lds)
        if self.fn.inline != defaults.inline:
            parts.append("inl" + self.fn.inline)
        if not self.fn.flat_wg_attr:
            parts.append("nowg")
        if not self.fn.convergent:
            parts.append("nocv")
        return "_".join(parts)


@dataclass(frozen=True)
class StageSpec:
    stage: str
    rows: int = 1
    seq: bool = False
    group: int = 0
    knobs: StageKnobs = field(default_factory=StageKnobs)

    @property
    def variant(self) -> str:
        return variant_name(self.rows, self.seq, self.group)

    @property
    def symbol(self) -> str:
        return ML4_FAMILY.symbol(self.stage, self.variant)


STAGES: dict = {}


def register_stage(name, body, supported):
    STAGES[name] = (body, supported)
    return body


def _load_stages(stage: str | None = None):
    import importlib

    modules = {"router": "moe", "ug": "moe", "down_gemv": "moe"}
    names = [modules[stage]] if stage in modules else sorted(set(modules.values()))
    for module in names:
        importlib.import_module(f"{__package__}.{module}")


def build_stage_kernel(spec: StageSpec, layout: MonoLayout | None = None):
    layout = ML4_FAMILY.load_layout(layout)
    return _build_stage_kernel(
        spec,
        layout,
        family=ML4_FAMILY,
        stages=STAGES,
        load_stages=_load_stages,
    )


def lower_stage(
    spec: StageSpec,
    layout: MonoLayout | None = None,
    *,
    arch="gfx950",
    flavor=None,
):
    return lower_device_function(
        build_stage_kernel(spec, layout),
        arch=arch,
        symbol=spec.symbol,
        llvm_flavor=flavor,
        knobs=spec.knobs.fn,
    )


DEFAULT_VARIANTS = ("r1", "r2", "r4", "r4s")


def _supported(spec, layout):
    return (
        spec.stage in ("router", "ug", "down_gemv")
        and spec.group == 0
        and 1 <= spec.rows <= layout.max_rows
        and (not spec.seq or spec.rows > 1)
        and spec.rows <= 4
    )


def _register():
    from . import moe

    register_stage("router", moe.emit_router, _supported)
    register_stage("ug", moe.emit_ug, _supported)
    register_stage("down_gemv", moe.emit_down_gemv, _supported)


_register()


__all__ = [
    "ABI_PARAMS",
    "DEFAULT_VARIANTS",
    "ML4_FAMILY",
    "STAGES",
    "StageKnobs",
    "StageSpec",
    "build_stage_kernel",
    "lower_stage",
    "parse_variant",
    "register_stage",
    "variant_name",
]
