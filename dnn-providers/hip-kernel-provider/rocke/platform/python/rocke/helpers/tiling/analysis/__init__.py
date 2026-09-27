# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tiling ANALYSIS / modeling layer -- PURE calc, no matplotlib. The shared, tested backend that the
visualization package, the ``/coalescing`` skill, and agents all ground on so nobody re-rolls the math.

Public API:
- vectorization: ``vector_transactions`` (b128-capped hardware-transaction pattern), ``addr_fn_from_strides``.
- coalescing:    ``analyze_coalescing`` -> ``CoalescingReport`` (cross-lane cache-line fusion; fused vs scattered).
- geometry:      ``resolve_origin`` / ``resolve_value`` (symbolic-origin resolver).
- roundtrip:     ``verify_lds_roundtrip`` (gate 1: the per-half LDS addressing round-trip).
- soundness:     ``verify_mma_soundness`` (gate 2: the recorded-MMA sound-MAC check).
"""

from .coalescing import CoalescingReport, Instruction, analyze_coalescing, assert_asm_backed
from .geometry import OriginResolutionError, resolve_origin, resolve_value
from .roundtrip import RoundTripError, verify_lds_roundtrip
from .soundness import MmaSoundnessError, verify_mma_soundness
from .vectorization import addr_fn_from_strides, vector_transactions

__all__ = ["vector_transactions", "addr_fn_from_strides",
           "analyze_coalescing", "CoalescingReport", "Instruction", "assert_asm_backed",
           "resolve_origin", "resolve_value", "OriginResolutionError",
           "verify_lds_roundtrip", "RoundTripError",
           "verify_mma_soundness", "MmaSoundnessError"]
