# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the gfx942 D256 bf16 prefill fast-path routing.

Covers the ``_d256_gfx942_fast`` dispatch gate (which routes the D256 gfx942
bf16 causal-prefill cohort to the 4-warp GQA builder) and guards the
productionization contract of that path:

  * the live builder ``build_gfx942_4warp_gqa`` is importable, and
  * the retired std-QK builder ``build_stdqk_attention_paged`` stays removed
    (0 call sites -- it was replaced at the seam by the 4-warp builder), and
  * ``supports_tiled_2d`` exposes the renamed ``use_d256_fast`` flag
    (never the old ``use_stdqk_paged`` name) that bypasses the generic
    staged-tile LDS-budget model for the fast path.

Arch is pinned via ``_RESOLVED_ATTENTION_ARCH`` (the resolver is memoized
process-wide and monkeypatched wholesale by tests), so these run on any host
without a gfx942 GPU.
"""

from __future__ import annotations

import inspect
import unittest

import kernels.common.attention_unified as au
import kernels.gfx942.attention_tiled_2d as t2d


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


def _d256_problem(**kw) -> "au.UnifiedAttentionProblem":
    """A D256 bf16 GQA causal-prefill problem the fast path accepts."""
    base = dict(
        total_q=4096,
        num_seqs=2,
        num_query_heads=16,
        num_kv_heads=2,
        head_size=256,
        block_size=32,
        max_seqlen_q=2048,
        max_seqlen_k=2048,
        dtype="bf16",
    )
    base.update(kw)
    return au.UnifiedAttentionProblem(**base)


class TestD256Gfx942FastGate(unittest.TestCase):
    def test_gate_accepts_d256_bf16_prefill(self):
        with _PinArch("gfx942"):
            self.assertTrue(au._d256_gfx942_fast(_d256_problem()))

    def test_gate_accepts_block_size_16(self):
        with _PinArch("gfx942"):
            self.assertTrue(au._d256_gfx942_fast(_d256_problem(block_size=16)))

    def test_gate_rejects_non_gfx942_arch(self):
        # Same problem must NOT take the gfx942 fast path on another arch.
        with _PinArch("gfx950"):
            self.assertFalse(au._d256_gfx942_fast(_d256_problem()))

    def test_gate_rejects_non_bf16(self):
        with _PinArch("gfx942"):
            self.assertFalse(au._d256_gfx942_fast(_d256_problem(dtype="fp16")))

    def test_gate_rejects_wrong_head_size(self):
        with _PinArch("gfx942"):
            self.assertFalse(au._d256_gfx942_fast(_d256_problem(head_size=128)))

    def test_gate_rejects_block_size_64(self):
        # block_size=64 cleanly falls back to the default builder (tile%block).
        with _PinArch("gfx942"):
            self.assertFalse(au._d256_gfx942_fast(_d256_problem(block_size=64)))

    def test_gate_rejects_decode(self):
        # max_seqlen_q == 1 is decode, not prefill.
        with _PinArch("gfx942"):
            self.assertFalse(au._d256_gfx942_fast(_d256_problem(max_seqlen_q=1)))

    def test_gate_rejects_feature_flags(self):
        with _PinArch("gfx942"):
            for kw in (
                dict(sliding_window=256),
                dict(softcap=30.0),
                dict(use_sinks=True),
                dict(use_alibi=True),
                dict(use_qq_bias=True),
                dict(use_fp8=True),
            ):
                self.assertFalse(
                    au._d256_gfx942_fast(_d256_problem(**kw)),
                    msg=f"fast path must exclude {kw}",
                )


class TestD256Gfx942BuilderContract(unittest.TestCase):
    def test_live_builder_present(self):
        self.assertTrue(callable(t2d.build_gfx942_4warp_gqa))

    def test_retired_stdqk_builder_removed(self):
        # Dead code guard: the std-QK paged builder had 0 call sites after the
        # 4-warp builder replaced it at the _d256_gfx942_fast seam; it must not
        # be reintroduced.
        self.assertFalse(hasattr(t2d, "build_stdqk_attention_paged"))

    def test_supports_flag_renamed(self):
        params = inspect.signature(t2d.supports_tiled_2d).parameters
        self.assertIn("use_d256_fast", params)
        self.assertNotIn("use_stdqk_paged", params)

    def test_supports_bypasses_lds_model_when_flagged(self):
        # The fast-path flag short-circuits the generic staged-tile LDS-budget
        # gate to (True, "supported"): the 4-warp builder manages its own LDS.
        ok, why = t2d.supports_tiled_2d(
            head_size=256,
            block_size=32,
            dtype="bf16",
            num_queries_per_kv=8,
            use_alibi=False,
            use_qq_bias=False,
            use_fp8=False,
            q_dtype=None,
            tile_size=32,
            num_warps=4,
            arch="gfx942",
            use_d256_fast=True,
        )
        self.assertTrue(ok, msg=why)

    def test_d256_entry_delegates_to_lean_natural_qk_body(self):
        """HD==256 must build via the lean natural-QK body
        (``_build_gfx942_4warp_gqa_d256_lean``): K/Q streamed direct-from-global,
        single-buffer ``V_lds`` staged once, one masked kv-loop. Regression guard for the
        develop D256 slowdown -- fails if the lean builder is absent or the ``HD==256``
        entry stops delegating to it (the emitted IR would then diverge)."""
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        self.assertTrue(hasattr(t2d, "_build_gfx942_4warp_gqa_d256_lean"))
        with _PinArch("gfx942"):
            spec = au._tiled_spec_from_problem(_d256_problem())
            via_entry = _lower_kernel_to_llvm_python(
                t2d.build_gfx942_4warp_gqa(spec, arch="gfx942"),
                arch="gfx942",
                llvm_flavor="llvm20",
            )
            via_lean = _lower_kernel_to_llvm_python(
                t2d._build_gfx942_4warp_gqa_d256_lean(spec, arch="gfx942"),
                arch="gfx942",
                llvm_flavor="llvm20",
            )
        self.assertEqual(
            via_entry, via_lean, "D256 entry must delegate to the lean natural-QK body"
        )


if __name__ == "__main__":
    unittest.main()
