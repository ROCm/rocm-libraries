# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LoopModel frontend exports, loaded only when requested.

Importing ``Tensile.LoopModel.loopir`` does not initialize Theta, scheduling,
or target decoding.
"""

from importlib import import_module

__all__ = [
    "Axis", "Global", "Shared", "Fragment", "RegionLayout", "Trajectory", "Movement",
    "Operand", "AgentAxisAssignment", "AgentAssignment", "Theta",
    "build_S", "derive_S", "group_live_peak", "group_width",
    "MmaShape", "plan_transfers", "mma_shape",
    "frag_regs", "free_tiles", "k_tiles",
    "operand_buffer_regs", "load_regs", "operand_tile_bytes",
    "operand_footprint_regs",
    "render_geometry", "render_schedule", "render_operand_facts",
    "validate_loopir", "agent_assignment_consistency",
    "DecodedSchedule", "ThetaTargetProfile", "decode_theta",
    "traversal",
]


_MODULE_EXPORTS = {
    "theta": (
        "Axis", "Global", "Shared", "Fragment", "RegionLayout", "Trajectory",
        "Movement", "Operand", "AgentAxisAssignment", "AgentAssignment", "Theta",
    ),
    "schedule": ("build_S", "derive_S", "group_width"),
    "traversal": (
        "MmaShape", "plan_transfers", "mma_shape", "frag_regs", "free_tiles",
        "k_tiles", "group_live_peak", "group_ring_size", "operand_buffer_regs",
        "load_regs", "operand_tile_bytes", "operand_footprint_regs",
    ),
    "checks": ("validate_loopir", "agent_assignment_consistency"),
    "render": (
        "render_geometry", "render_schedule", "render_operand_facts",
    ),
    "decoder": ("DecodedSchedule", "ThetaTargetProfile", "decode_theta"),
}
_LAZY = {
    name: module
    for module, names in _MODULE_EXPORTS.items()
    for name in names
}


def __getattr__(name):
    if name == "traversal":
        value = import_module(f"{__name__}.traversal")
    elif name in _LAZY:
        value = getattr(import_module(f"{__name__}.{_LAZY[name]}"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
