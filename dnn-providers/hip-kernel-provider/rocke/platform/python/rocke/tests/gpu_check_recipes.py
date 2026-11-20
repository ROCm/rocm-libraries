# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: a kernel that checks a value with ``gpu_checked_value``, recorded as a recipe,
and the texts the doors that hand out a launch without reading its GPU-check word refuse it with.

The portable-IR launch-plan tests, the C++ plan comparison and the parameter-contract tests
all use it, so every door is tested on one kernel.
"""

from __future__ import annotations

from typing import Any

from rocke.core.ir import I32, IRBuilder, KernelDef, PtrType
from rocke.core.passes import optimize_kernel
from rocke.portable_ir.src import launch
from rocke.portable_ir.src.kerneldef_to_recipe import kerneldef_to_recipe

__all__ = [
    "GPU_CHECK_AND_FACTS_REFUSAL",
    "GPU_CHECK_REFUSAL",
    "gpu_check_recipe",
    "gpu_checked_kernel",
]

# What a recipe launch plan says for gpu_checked_kernel(), and for one that
# also states a parameter fact (the facts are named first).
GPU_CHECK_REFUSAL = (
    "gpu_checked: only rocke's runtime reads this kernel's GPU-check word, so a "
    "false gpu_checked_value fact would go unseen -- launch it through "
    "KernelLauncher.from_artifact"
)
GPU_CHECK_AND_FACTS_REFUSAL = (
    "gpu_checked: parameters carry facts, which a launch from a code object or "
    "from a launch plan does not check -- launch it through "
    "KernelLauncher.from_artifact"
)


def gpu_checked_kernel(
    *, with_facts: bool = False, result_used: bool = True
) -> KernelDef:
    """A kernel named ``gpu_checked`` that checks a loaded value with
    ``gpu_checked_value``. ``with_facts`` also states a fact on a parameter.
    With ``result_used=False`` nothing uses the checked value and the kernel is
    optimized, which drops the check's range marker and keeps its failure path."""
    b = IRBuilder("gpu_checked")
    out = b.param("out", PtrType(I32, "global"))
    if with_facts:
        b.param("n", I32, ge=1)
    loaded = b.readfirstlane(b.global_load(out, b.block_id_x(), I32))
    checked = b.gpu_checked_value(loaded, ge=0, le=100)
    if result_used:
        b.global_store(out, b.const_i32(0), checked)
    b.ret()
    if not result_used:
        optimize_kernel(b.kernel)
    return b.kernel


def gpu_check_recipe(
    *, with_facts: bool = False, result_used: bool = True
) -> dict[str, Any]:
    """``gpu_checked_kernel`` recorded as a recipe, with launch geometry."""
    kernel = gpu_checked_kernel(with_facts=with_facts, result_used=result_used)
    return launch.attach_launch(
        kerneldef_to_recipe(kernel), grid=[1, 1, 1], block=[64, 1, 1]
    )
