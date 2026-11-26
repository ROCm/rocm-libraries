# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Source-only gradient execution plans; not a frozen replay format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rocke.core.ir import KernelDef


@dataclass
class Step:
    kernel: KernelDef
    signature: tuple | list
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    scalars: dict
    bindings: dict[str, str]
    llvm_flavor: str | None = None


@dataclass
class Plan:
    steps: list[Step]
    selection: str
    spec: object
    output_fill: int = 0
    immutable: dict[str, np.ndarray] = field(default_factory=dict)
    scratch_bytes: dict[str, int] = field(default_factory=dict)


def step(kernel, signature, grid, block, values, bindings, *, llvm_flavor=None) -> Step:
    """Separate pointer bindings from the scalar ABI for repeatable launches."""
    scalars = dict(values)
    for name in bindings:
        del scalars[name]
    return Step(kernel, signature, grid, block, scalars, bindings, llvm_flavor)
