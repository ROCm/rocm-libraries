# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for gfx942 D128 sliding-window prefill routing.

D128 sliding-window (SWA) prefill on gfx942 is routed to the wide 32x32x8
non-ring flash path (single-pass fp32 online softmax) instead of the narrow
16x16 fallback. The wide-flash emitter is already SW-complete (windowed KV-skip
+ per-element window mask + all-masked -inf guard); SWA is gated only at
dispatch. These tests guard that routing:

  * fp16 + bf16 D128 SW take the flash path with the sliced-K ring forced OFF
    (the ring would need the windowed KV-skip / -inf rows composed with its
    per-slice softmax merge), and with the transposed VALU mask-limit stack OFF
    (it elides the per-element compare, invalid under a window),
  * the geometry is nw=2 + no-cfvst (nw=4 non-ring overflows the 64 KB LDS cap),
  * the D128 SW spec emits without a validator error, and
  * causal D128 (both dtypes) and D64 SW routing are unchanged (no regression).

Arch is pinned via ``_RESOLVED_ATTENTION_ARCH`` (memoized process-wide and
monkeypatched wholesale), so these run on any host without a gfx942 GPU.
"""

from __future__ import annotations

import unittest

import kernels.common.attention_unified as au
from kernels.common.attention_unified import _tiled_2d_impl, _tiled_spec_from_problem


class _PinArch:
    """Context-manager that pins ``_RESOLVED_ATTENTION_ARCH`` to ``arch``."""

    def __init__(self, arch: str):
        self.arch = arch

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = self.arch
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


def _d128_problem(**kw) -> "au.UnifiedAttentionProblem":
    """A Mistral-7B-like D128 GQA prefill problem (num_seqs=1, long context)."""
    base = dict(
        total_q=8192,
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=8192,
        max_seqlen_k=8192,
        dtype="bf16",
        sliding_window=4096,
    )
    base.update(kw)
    return au.UnifiedAttentionProblem(**base)


class TestGfx942D128SwRouting(unittest.TestCase):
    def test_sw_takes_flash_path(self):
        with _PinArch("gfx942"):
            self.assertTrue(au._enable_gfx942_bf16_flash(_d128_problem(dtype="bf16")))
            self.assertTrue(au._enable_gfx942_fp16_flash(_d128_problem(dtype="fp16")))

    def test_sw_forces_non_ring(self):
        with _PinArch("gfx942"):
            for dt in ("bf16", "fp16"):
                self.assertFalse(
                    au._enable_gfx942_flash_k_sliced_ring(_d128_problem(dtype=dt)),
                    msg=f"D128 SW {dt} must run non-ring",
                )

    def test_sw_disables_mask_limit(self):
        with _PinArch("gfx942"):
            for dt in ("bf16", "fp16"):
                self.assertFalse(
                    au._enable_gfx942_flash_mask_limit(_d128_problem(dtype=dt)),
                    msg=f"D128 SW {dt} must disable the mask-limit VALU stack",
                )

    def test_sw_geometry_is_nw2_no_cfvst(self):
        with _PinArch("gfx942"):
            for dt in ("bf16", "fp16"):
                p = _d128_problem(dtype=dt)
                self.assertEqual(au._select_gfx942_flash_num_warps(p), 2)
                self.assertFalse(au._gfx942_flash_use_cfvst(p))

    def test_sw_spec_emits(self):
        # Building the kernel exercises the spec __post_init__ validators, which
        # raise if an SW-incompatible sub-opt (mask-limit / register-pv) leaks in.
        _, build2d, _ = _tiled_2d_impl("gfx942")
        with _PinArch("gfx942"):
            for dt, sq in (
                ("bf16", 8192),
                ("bf16", 16384),
                ("fp16", 8192),
                ("fp16", 16384),
            ):
                spec = _tiled_spec_from_problem(
                    _d128_problem(dtype=dt, max_seqlen_q=sq)
                )
                self.assertEqual(spec.sliding_window, 4096)
                self.assertFalse(spec.use_k_sliced_ring)
                self.assertEqual(spec.num_warps, 2)
                build2d(spec, arch="gfx942")  # raises on an invalid flag combo

    def test_causal_unchanged(self):
        with _PinArch("gfx942"):
            # bf16 causal: flash on, ring off (its bs=64 optimum, unchanged).
            pb = _d128_problem(dtype="bf16", sliding_window=0)
            self.assertTrue(au._enable_gfx942_bf16_flash(pb))
            self.assertFalse(au._enable_gfx942_flash_k_sliced_ring(pb))
            # fp16 causal: flash on, depth-2 ring on (unchanged).
            pf = _d128_problem(dtype="fp16", sliding_window=0)
            self.assertTrue(au._enable_gfx942_fp16_flash(pf))
            self.assertTrue(au._enable_gfx942_flash_k_sliced_ring(pf))

    def test_d64_sw_stays_narrow(self):
        # Only D128 SW is opened to the flash path; D64 SW keeps the narrow path.
        with _PinArch("gfx942"):
            for dt in ("bf16", "fp16"):
                p = _d128_problem(dtype=dt, head_size=64)
                self.assertFalse(au._enable_gfx942_bf16_flash(p))
                self.assertFalse(au._enable_gfx942_fp16_flash(p))


if __name__ == "__main__":
    unittest.main()
