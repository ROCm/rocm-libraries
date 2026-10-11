# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Flash-style MFMA fixture: tl.dot -> tl.exp -> tl.dot.

Compiled unguarded with Triton 3.8.0 for gfx950 it carries v_pk_*_f32 beside
v_mfma (the erratum pattern); the erratum guard must remove it. Used only by
test_erratum_guard.py; never shipped.
"""

import triton
import triton.language as tl


@triton.jit
def mfma_fixture(
    Q, K, V, O, scale, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr
):
    m = tl.arange(0, BM)
    n = tl.arange(0, BN)
    d = tl.arange(0, D)
    q = tl.load(Q + m[:, None] * D + d[None, :])
    k = tl.load(K + n[None, :] * D + d[:, None])
    v = tl.load(V + n[:, None] * D + d[None, :])
    s = tl.dot(q, k)
    mx = tl.max(s, axis=1)
    p = tl.exp(s * scale - mx[:, None])
    l = tl.sum(p, axis=1)
    o = tl.dot(p.to(tl.bfloat16), v) / l[:, None]
    tl.store(O + m[:, None] * D + d[None, :], o.to(tl.bfloat16))


SIGNATURE = {
    "Q": "*bf16",
    "K": "*bf16",
    "V": "*bf16",
    "O": "*bf16",
    "scale": "fp32",
    "BM": "constexpr",
    "BN": "constexpr",
    "D": "constexpr",
}
# (BM, BN, D, num_warps): a tile config that, unguarded on gfx950, emits v_pk_*_f32 beside v_mfma.
CONFIG = (64, 64, 128, 4)
CONSTEXPRS = {"BM": CONFIG[0], "BN": CONFIG[1], "D": CONFIG[2]}
NUM_WARPS = CONFIG[3]
