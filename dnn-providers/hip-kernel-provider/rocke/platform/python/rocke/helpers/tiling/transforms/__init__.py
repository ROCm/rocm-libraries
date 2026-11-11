# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fragment-transform solver, MMA-safety observers, and the register-reorder verb.

Split by audience (see ``docs/tiling_api_contract.md``):
- ``_core``     -- shared solver primitives (the neutral delta solver + ``as_forward_map`` + value types).
- ``observers`` -- the read-only analysis TOOLBOX (classify/describe/soundness/derive-C).
- ``verb``      -- ``transform_fragment``, the front-door verb that realises a ``reorder`` plan.

This ``__init__`` re-exports the full public surface, so ``from ..transforms import X`` is unchanged.
"""

from __future__ import annotations

from ._core import (
    Diagnostic,
    ReorderPlan,
    TransformPlan,
    as_forward_map,
    interleave_idx,
    k_distribution,
    name_permutation,
)
from .observers import (
    classify_transform,
    derive_c_distribution,
    describe_edge,
    mma_accumulator_flow_consistent,
    mma_operand_layout_sound,
    mma_operand_repair_hint,
    mma_pair_compatible,
    mma_pair_k_aligned,
    reorder_between,
)
from .verb import transform_fragment

__all__ = [
    "TransformPlan",
    "interleave_idx",
    "k_distribution",
    "classify_transform",
    "mma_pair_k_aligned",
    "derive_c_distribution",
    "Diagnostic",
    "mma_accumulator_flow_consistent",
    "as_forward_map",
    "mma_operand_layout_sound",
    "mma_operand_repair_hint",
    "mma_pair_compatible",
    "transform_fragment",
    "describe_edge",
    "name_permutation",
    "reorder_between",
    "ReorderPlan",
]
