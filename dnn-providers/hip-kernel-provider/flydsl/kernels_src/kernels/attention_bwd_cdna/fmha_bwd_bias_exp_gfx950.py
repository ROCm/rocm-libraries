# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""What the gfx950 backward kernels' bias exponent shares: its fast-math flags and the LSE floor.

A bias build forms `P = exp2((sm_scale*S + bias - lse) * log2e)`: the score, the bias and the LSE combine in natural units and
the `log2e` scale comes last. Scaling the bias first (`bias * log2e`) overflows for a bias near the lowest finite value, and a row
masked entirely by such a bias then has no finite exponent to carry. The forward's `c_neg_floor` is not used here because it
is `-3.0e38`: it sits above the `-3.3895e38` LSE of a bf16 row whose scores are all lowest-finite, and would turn that row's
`P = 1` into `P = 0`.
"""

import flydsl.expr as fx

__all__ = ["BWD_LSE_FLOOR", "FM_BIAS_EXP"]

# The floor a bias build puts under the row's natural-unit LSE: `-FLT_MAX`, the lowest finite f32.
BWD_LSE_FLOOR = -3.4028235e38

# Fast-math flags of a bias build's exponent chain. `contract` lets `sm_scale*S + bias` fuse; there is no `ninf`, `nnan` or
# `reassoc`, so a `-inf` bias and a `+inf` empty-row LSE reach `exp2` as `-inf` and the "combine, then scale last" order holds.
FM_BIAS_EXP = fx.arith.FastMathFlags.contract
