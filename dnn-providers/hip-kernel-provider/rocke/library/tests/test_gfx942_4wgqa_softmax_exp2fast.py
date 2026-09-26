# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only emit test: the gfx942 4-warp GQA softmax lowers to ``exp2_fast``.

The 4-warp GQA kernel (``build_gfx942_4warp_gqa``) runs the online softmax as
``alpha = exp2(m_old - m_new)`` and ``P = exp2(S*scale - m_new)``. Both arguments
are ``<= 0`` by the running-max invariant, so the ``exp2`` overflow/underflow
range-reduction guard is unnecessary and the kernel emits ``math.exp2_fast``
(raw ``v_exp_f32``) rather than the guarded ``math.exp2``. This guards that
lowering for both bf16 and fp16 on the D128 wide-flash 4-warp cohort; it fails
against the pre-change kernel (which emitted the guarded ``math.exp2`` for these
sites). Arch is pinned so it runs on any host (no GPU / comgr).
"""

from __future__ import annotations

import re
import unittest

import kernels.common.attention_unified as au
from kernels.common.attention_unified import _tiled_spec_from_problem
from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa
from rocke.core.ir_print import print_ir


class _PinArch:
    """Pin ``_RESOLVED_ATTENTION_ARCH`` so spec resolution runs off-GPU."""

    def __init__(self, arch: str):
        self.arch = arch

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = self.arch
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


def _emit_4wgqa_ir(dtype: str) -> str:
    """Build the D128 SW 4-warp GQA kernel for ``dtype`` and return its IR text."""
    p = au.UnifiedAttentionProblem(
        total_q=8192,
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=8192,
        max_seqlen_k=8192,
        dtype=dtype,
        sliding_window=4096,
    )
    spec = _tiled_spec_from_problem(p)
    return print_ir(build_gfx942_4warp_gqa(spec, arch="gfx942"))


class TestGfx942_4wgqaSoftmaxExp2Fast(unittest.TestCase):
    def test_softmax_uses_exp2_fast_not_guarded_exp2(self):
        with _PinArch("gfx942"):
            for dt in ("bf16", "fp16"):
                ir = _emit_4wgqa_ir(dt)
                n_fast = len(re.findall(r"math\.exp2_fast\b", ir))
                n_slow = len(re.findall(r"math\.exp2(?!_fast)", ir))
                # alpha + P are the two online-softmax exp2 sites.
                self.assertGreaterEqual(
                    n_fast,
                    2,
                    msg=f"{dt}: expected >=2 math.exp2_fast (alpha + P), got {n_fast}",
                )
                # The exp2 -> exp2_fast swap must be complete (no guarded exp2 left).
                self.assertEqual(
                    n_slow,
                    0,
                    msg=f"{dt}: guarded math.exp2 must be gone from the softmax, "
                    f"got {n_slow}",
                )


if __name__ == "__main__":
    unittest.main()
