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
    diagnose_k_match,
    mma_compatible,
    mma_pair_compatible,
    operand_soundness,
    reorder_between,
    validate_operands,
)
from .verb import transform_fragment

__all__ = [
    "TransformPlan", "interleave_idx", "k_distribution", "classify_transform", "validate_operands",
    "derive_c_distribution", "Diagnostic", "diagnose_k_match", "as_forward_map",
    "operand_soundness", "mma_compatible", "mma_pair_compatible", "transform_fragment",
    "describe_edge", "name_permutation", "reorder_between", "ReorderPlan",
]
