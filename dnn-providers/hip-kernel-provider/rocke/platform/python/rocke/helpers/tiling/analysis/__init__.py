# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tiling ANALYSIS / modeling layer -- PURE calc, no matplotlib. The shared, tested backend that the
visualization package, the ``/coalescing`` skill, and agents all ground on so nobody re-rolls the math.

Public API:
- vectorization: ``vector_transactions`` (b128-capped hardware-transaction pattern), ``addr_fn_from_strides``.
- coalescing:    ``analyze_coalescing`` -> ``CoalescingReport`` (cross-lane cache-line fusion; fused vs scattered).
"""

from .coalescing import CoalescingReport, Instruction, analyze_coalescing, assert_asm_backed
from .vectorization import addr_fn_from_strides, vector_transactions

__all__ = ["vector_transactions", "addr_fn_from_strides",
           "analyze_coalescing", "CoalescingReport", "Instruction", "assert_asm_backed"]
