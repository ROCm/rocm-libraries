# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LoopIR to GIR lowering and diagnostics."""

from .loopir_to_gir import gir_text, lower_to_gir, steady_chain

__all__ = ["gir_text", "lower_to_gir", "steady_chain"]
