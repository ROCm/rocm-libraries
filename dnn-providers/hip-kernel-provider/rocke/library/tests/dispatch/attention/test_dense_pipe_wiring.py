# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the gfx942 dense-pipe dispatcher wiring (fp16 + bf16).

Covers:
  - dense_pipe candidate is registered and discoverable
  - dense_pipe priority outranks unified_2d (priority 10) for eligible gfx942 fp16 shapes
  - MHA short-context (seqlen_q<=768) routes to dense_pipe, not narrow
  - GQA short-context routes to unified_2d narrow (dense_pipe rejected)
  - explicit algorithm="dense_pipe" forces the candidate
  - spec name and path are correct
  - bf16 flash gate is default-on with a small_q_narrow (q<=768) carve-out
"""

from __future__ import annotations

import os
import unittest

import kernels.common.attention_unified as au
from dispatch.attention import (
    AttentionRequest,
    attention_candidates,
    dispatch_attention,
)


def _gfx942_fp16(**kw) -> AttentionRequest:
    base = dict(
        batch=2,
        nhead_q=16,
        nhead_k=16,
        seqlen_q=512,
        seqlen_k=512,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        dtype="fp16",
    )
    base.update(kw)
    return AttentionRequest(**base)


class _Gfx942Arch:
    """Context-manager that pins _RESOLVED_ATTENTION_ARCH to gfx942."""

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = "gfx942"

        # Keep tests hermetic: clear any env overrides that would affect routing.
        self._env_keys = (
            "HIPDNN_GFX942_K_SLICED_RING",
            "HIPDNN_GFX942_FLASH_MLIM",
            "HIPDNN_GFX942_FLASH_WIDE",
            "HIPDNN_GFX942_BF16_CFVST",
            "HIPDNN_GFX942_BF16_WIDE",
        )
        self._old_env = {k: os.environ.get(k) for k in self._env_keys}
        for k in self._env_keys:
            os.environ.pop(k, None)

        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old
        for k, v in self._old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestDensePipeRegistration(unittest.TestCase):
    def test_candidate_is_registered(self):
        names = [c.name for c in attention_candidates()]
        self.assertIn("attention_gfx942_dense_pipe", names)

    def test_spec_id_is_gfx942_dense_pipe(self):
        candidate = next(
            c for c in attention_candidates() if c.name == "attention_gfx942_dense_pipe"
        )
        self.assertEqual(candidate.spec_id, "gfx942_dense_pipe")

    def test_algorithm_is_dense_pipe(self):
        candidate = next(
            c for c in attention_candidates() if c.name == "attention_gfx942_dense_pipe"
        )
        self.assertEqual(candidate.algorithm, "dense_pipe")

    def test_priority_beats_unified_2d(self):
        # The registry sorts ascending; dense_pipe must have a lower number than
        # unified_2d so it is selected first when both support the same request.
        dense = next(
            c for c in attention_candidates() if c.name == "attention_gfx942_dense_pipe"
        )
        unified_2d = next(
            c for c in attention_candidates() if c.name == "attention_unified_2d"
        )
        self.assertLess(dense.priority, unified_2d.priority)


class TestDensePipeSupportGates(unittest.TestCase):
    def test_rejects_non_gfx942_arch(self):
        with _Gfx942Arch():
            candidate = next(
                c
                for c in attention_candidates()
                if c.name == "attention_gfx942_dense_pipe"
            )
            ok, why = candidate.admits(_gfx942_fp16(arch="gfx950"))
            self.assertFalse(ok)
            self.assertIn("gfx942", why)

    def test_rejects_non_fp16_dtype(self):
        with _Gfx942Arch():
            candidate = next(
                c
                for c in attention_candidates()
                if c.name == "attention_gfx942_dense_pipe"
            )
            ok, why = candidate.admits(_gfx942_fp16(dtype="bf16"))
            self.assertFalse(ok)
            self.assertIn("fp16", why)

    def test_rejects_3d_problem(self):
        # decode (seqlen_q=1, long KV, small grid) routes to 3D — dense_pipe refuses
        with _Gfx942Arch():
            candidate = next(
                c
                for c in attention_candidates()
                if c.name == "attention_gfx942_dense_pipe"
            )
            req = _gfx942_fp16(
                batch=1, nhead_q=16, nhead_k=16, seqlen_q=1, seqlen_k=8192
            )
            ok, why = candidate.admits(req)
            self.assertFalse(ok)
            self.assertIn("3D", why)

    def test_admits_d128_sliding_window(self):
        # D128 sliding-window prefill rides the non-ring wide flash path (SW mask
        # + windowed KV-skip live in the emitter, and the sliced-K ring is forced
        # off for D128 SW), so ``_enable_gfx942_fp16_flash`` admits it.
        with _Gfx942Arch():
            candidate = next(
                c
                for c in attention_candidates()
                if c.name == "attention_gfx942_dense_pipe"
            )
            ok, why = candidate.admits(
                _gfx942_fp16(seqlen_q=512, seqlen_k=4096, sliding_window=256)
            )
            self.assertTrue(ok, why)

    def test_rejects_d64_sliding_window(self):
        # Only D128 gained the sliding-window carve-out; D64 SW has no emitter
        # path and must still be turned down by the fp16-flash gate.
        with _Gfx942Arch():
            candidate = next(
                c
                for c in attention_candidates()
                if c.name == "attention_gfx942_dense_pipe"
            )
            ok, why = candidate.admits(
                _gfx942_fp16(
                    seqlen_q=512,
                    seqlen_k=4096,
                    hdim_q=64,
                    hdim_v=64,
                    sliding_window=256,
                )
            )
            self.assertFalse(ok)
            self.assertIn("flash not eligible", why)


class TestDensePipeRouting(unittest.TestCase):
    """Verify that dispatch picks dense_pipe for eligible gfx942 fp16 shapes."""

    def test_mha_long_context_routes_dense_pipe(self):
        # MHA (nhead_q==nhead_k), long context -> dense_pipe wins over unified_2d
        with _Gfx942Arch():
            r = dispatch_attention(
                _gfx942_fp16(
                    batch=2, nhead_q=16, nhead_k=16, seqlen_q=2048, seqlen_k=2048
                )
            )
        self.assertEqual(r.candidate.spec_id, "gfx942_dense_pipe")
        self.assertEqual(r.spec.path, "2d")
        self.assertEqual(r.spec.name, "rocke_attention_gfx942_dense_pipe")

    def test_mha_short_context_routes_dense_pipe_not_narrow(self):
        # MHA short-context (seqlen_q=512 <= 768): fp16-flash gate now allows it,
        # so dense_pipe wins instead of routing to the narrow path via unified_2d.
        with _Gfx942Arch():
            r = dispatch_attention(
                _gfx942_fp16(
                    batch=2, nhead_q=16, nhead_k=16, seqlen_q=512, seqlen_k=512
                )
            )
        self.assertEqual(r.candidate.spec_id, "gfx942_dense_pipe")

    def test_gqa_short_context_does_not_route_dense_pipe(self):
        # GQA (nhead_q > nhead_k) short-context -> narrow path still wins;
        # dense_pipe support() must reject because fp16-flash is disabled for
        # GQA + small_q_narrow.
        with _Gfx942Arch():
            candidate = next(
                c
                for c in attention_candidates()
                if c.name == "attention_gfx942_dense_pipe"
            )
            req = _gfx942_fp16(
                batch=2, nhead_q=32, nhead_k=8, seqlen_q=512, seqlen_k=512
            )
            ok, _ = candidate.admits(req)
        self.assertFalse(ok)

    def test_spec_records_correct_dims(self):
        with _Gfx942Arch():
            r = dispatch_attention(
                _gfx942_fp16(
                    batch=1,
                    nhead_q=8,
                    nhead_k=8,
                    seqlen_q=1024,
                    seqlen_k=1024,
                    hdim_q=64,
                    hdim_v=64,
                    kv_block_size=32,
                )
            )
        self.assertEqual(r.spec.head_size, 64)
        self.assertEqual(r.spec.block_size, 32)
        self.assertEqual(r.spec.dtype, "fp16")

    def test_explicit_algorithm_forces_dense_pipe(self):
        with _Gfx942Arch():
            r = dispatch_attention(
                _gfx942_fp16(
                    seqlen_q=2048,
                    seqlen_k=2048,
                    algorithm="dense_pipe",
                )
            )
        self.assertEqual(r.candidate.algorithm, "dense_pipe")

    def test_explicit_algorithm_unified_2d_skips_dense_pipe(self):
        with _Gfx942Arch():
            r = dispatch_attention(
                _gfx942_fp16(
                    seqlen_q=2048,
                    seqlen_k=2048,
                    algorithm="auto",
                    spec_id="unified_2d",
                )
            )
        self.assertEqual(r.candidate.spec_id, "unified_2d")


class TestBf16FlashGate(unittest.TestCase):
    """Verify the bf16 flash gate (default-on, small_q_narrow carve-out)."""

    def _bf16_problem(self, seqlen_q=2048, seqlen_k=2048, nhead_q=16, nhead_k=16):
        return au.UnifiedAttentionProblem(
            total_q=2 * seqlen_q,
            num_seqs=2,
            num_query_heads=nhead_q,
            num_kv_heads=nhead_k,
            head_size=128,
            block_size=16,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            dtype="bf16",
        )

    def test_bf16_flash_on_by_default_long_context(self):
        with _Gfx942Arch():
            self.assertTrue(au._enable_gfx942_bf16_flash(self._bf16_problem()))

    def test_bf16_flash_short_context_disabled_mha(self):
        # MHA short-context (q=512<=768): small_q_narrow applies -> bf16 flash off.
        # Unlike fp16, bf16 D128 ring is not yet validated so MHA also uses narrowpath.
        with _Gfx942Arch():
            p = self._bf16_problem(seqlen_q=512, seqlen_k=512)
            self.assertFalse(au._enable_gfx942_bf16_flash(p))

    def test_bf16_flash_short_context_disabled_gqa(self):
        # GQA short-context: small_q_narrow -> bf16 flash off.
        with _Gfx942Arch():
            p = self._bf16_problem(seqlen_q=512, seqlen_k=512, nhead_q=32, nhead_k=8)
            self.assertFalse(au._enable_gfx942_bf16_flash(p))

    def test_bf16_flash_long_context_gqa_enabled(self):
        # GQA long-context (q>768): small_q_narrow off -> bf16 flash on.
        with _Gfx942Arch():
            p = self._bf16_problem(seqlen_q=2048, seqlen_k=2048, nhead_q=32, nhead_k=8)
            self.assertTrue(au._enable_gfx942_bf16_flash(p))

    def test_bf16_ring_enabled_for_d64_prefill(self):
        # D64 ring is default-on for bf16 prefill.
        with _Gfx942Arch():
            p = au.UnifiedAttentionProblem(
                total_q=2 * 2048,
                num_seqs=2,
                num_query_heads=16,
                num_kv_heads=16,
                head_size=64,
                block_size=16,
                max_seqlen_q=2048,
                max_seqlen_k=2048,
                dtype="bf16",
            )
            self.assertTrue(au._enable_gfx942_flash_k_sliced_ring(p))

    def test_bf16_ring_disabled_for_d128(self):
        # D128 bf16 stays OFF the ring, and the reason is performance rather
        # than correctness. The depth-3 slot-reuse bug that originally excluded
        # D128 was fixed two ways (the drain-on-reuse fence, now unconditional,
        # and the depth-2 ring), so a correct bf16 D128 ring exists -- it is
        # just slower than the non-ring T=64 flash path at the production
        # block_size=64. fp16 D128 takes the depth-2 ring; bf16 does not.
        with _Gfx942Arch():
            p = self._bf16_problem(seqlen_q=2048, seqlen_k=2048)
            self.assertFalse(au._enable_gfx942_flash_k_sliced_ring(p))

    def test_fp16_ring_enabled_for_d128_at_depth_2(self):
        # The other half of the same routing decision, pinned so the two
        # dtypes cannot silently converge: fp16 D128 does take the ring, and
        # specifically the depth-2 schedule (k%2 avoids slot reuse in the
        # k_groups=4 live set).
        with _Gfx942Arch():
            p = au.UnifiedAttentionProblem(
                total_q=2 * 2048,
                num_seqs=2,
                num_query_heads=16,
                num_kv_heads=16,
                head_size=128,
                block_size=16,
                max_seqlen_q=2048,
                max_seqlen_k=2048,
                dtype="fp16",
            )
            self.assertTrue(au._enable_gfx942_flash_k_sliced_ring(p))
            self.assertEqual(au._select_gfx942_flash_ring_depth(p), 2)

    def test_bf16_mask_limit_enabled_for_prefill(self):
        with _Gfx942Arch():
            p = self._bf16_problem(seqlen_q=2048, seqlen_k=2048)
            self.assertTrue(au._enable_gfx942_flash_mask_limit(p))

    def test_bf16_q_direct_enabled_for_d64(self):
        # q_direct is unblocked for bf16 D64.
        with _Gfx942Arch():
            p = au.UnifiedAttentionProblem(
                total_q=2 * 2048,
                num_seqs=2,
                num_query_heads=16,
                num_kv_heads=16,
                head_size=64,
                block_size=16,
                max_seqlen_q=2048,
                max_seqlen_k=2048,
                dtype="bf16",
            )
            self.assertTrue(au._enable_gfx942_flash_q_direct(p))

    def test_bf16_q_direct_disabled_for_d128(self):
        # q_direct is D64-only.
        with _Gfx942Arch():
            p = self._bf16_problem(seqlen_q=2048, seqlen_k=2048)
            self.assertFalse(au._enable_gfx942_flash_q_direct(p))


class TestFp16FlashGate(unittest.TestCase):
    """Direct unit tests for _enable_gfx942_fp16_flash routing logic."""

    def _fp16_problem(self, seqlen_q=2048, seqlen_k=2048, nhead_q=16, nhead_k=16):
        return au.UnifiedAttentionProblem(
            total_q=2 * seqlen_q,
            num_seqs=2,
            num_query_heads=nhead_q,
            num_kv_heads=nhead_k,
            head_size=128,
            block_size=16,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            dtype="fp16",
        )

    def test_mha_long_context_flash_enabled(self):
        with _Gfx942Arch():
            self.assertTrue(au._enable_gfx942_fp16_flash(self._fp16_problem()))

    def test_mha_short_context_flash_enabled(self):
        # Key behaviour from the commit: MHA short-context no longer excluded.
        with _Gfx942Arch():
            p = self._fp16_problem(seqlen_q=512, seqlen_k=512)
            self.assertTrue(au._enable_gfx942_fp16_flash(p))

    def test_gqa_short_context_flash_disabled(self):
        # GQA (nhead_q > nhead_k) at short context: narrow wins -> flash off.
        with _Gfx942Arch():
            p = self._fp16_problem(seqlen_q=512, seqlen_k=512, nhead_q=32, nhead_k=8)
            self.assertFalse(au._enable_gfx942_fp16_flash(p))

    def test_gqa_long_context_flash_enabled(self):
        # Long context: narrow path not triggered -> flash on for GQA too.
        with _Gfx942Arch():
            p = self._fp16_problem(seqlen_q=2048, seqlen_k=2048, nhead_q=32, nhead_k=8)
            self.assertTrue(au._enable_gfx942_fp16_flash(p))

    def test_flash_off_for_non_gfx942(self):
        old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = "gfx950"
        try:
            self.assertFalse(au._enable_gfx942_fp16_flash(self._fp16_problem()))
        finally:
            au._RESOLVED_ATTENTION_ARCH = old

    def test_flash_off_for_bf16(self):
        with _Gfx942Arch():
            p = au.UnifiedAttentionProblem(
                total_q=4096,
                num_seqs=2,
                num_query_heads=16,
                num_kv_heads=16,
                head_size=128,
                block_size=16,
                max_seqlen_q=2048,
                max_seqlen_k=2048,
                dtype="bf16",
            )
            self.assertFalse(au._enable_gfx942_fp16_flash(p))


if __name__ == "__main__":
    unittest.main()
