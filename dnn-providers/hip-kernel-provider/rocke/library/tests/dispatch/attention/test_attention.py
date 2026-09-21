# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Path-selection + coverage tests for the attention dispatcher family."""

from __future__ import annotations

import unittest

from dispatch.attention import (
    AttentionRequest,
    attention_candidates,
    dispatch_attention,
)
from dispatch.attention.common import ATTENTION_FEATURES

# Frozen map of every registered candidate to the feature set its Capability
# declares. Generated once from the registry; a new candidate (or a features
# edit on an existing one) fails ``test_declared_features_are_frozen`` until it
# is listed here, so a widening like fp8 can never silently reach a path that
# does not implement it.
EXPECTED_FEATURES = {
    "attention_gfx942_dense": {"causal"},
    "attention_gfx950_dense": {"causal", "sinks", "sliding_window"},
    "attention_d256_decode": {"causal"},
    "attention_gfx1250_wmma": {"causal"},
    "attention_gfx942_dense_pipe": {"causal", "sinks", "sliding_window"},
    "attention_gfx950_d256": {"causal"},
    "attention_unified_2d": {"causal", "fp8", "sinks", "sliding_window"},
    "attention_unified_3d": {"causal", "fp8", "sinks", "sliding_window"},
}

# Request kwargs that turn each attention feature on. Keyed by the vocabulary so
# a feature added to ATTENTION_FEATURES forces an entry here (KeyError until
# listed) rather than inheriting silence.
_ENABLE_FEATURE = {
    "causal": {"mask_type": 1},
    "sliding_window": {"sliding_window": 256},
    "sinks": {"use_sinks": True},
    "fp8": {"use_fp8": True},
}

# Features the dispatch spec (hence kernel_name / spec hash) actually encodes.
# ``fp8`` is carried on AttentionSpec; ``sliding_window`` reroutes 3d->2d and
# path is in the name. ``causal`` and ``sinks`` are a PRE-EXISTING gap: the spec
# is built from (path, head_size, block_size, dtype, gqa, use_fp8) only, so
# toggling the mask or sinks does not change kernel_name today. That fix belongs
# on the consumers that key a compile cache on kernel_name() and is out of scope
# here; the gap is frozen below so a future fix trips the test instead of
# passing silently.
_IDENTITY_ENCODED = {"fp8", "sliding_window"}


def _attn(arch="gfx950", **kw):
    base = dict(
        batch=2,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=512,
        seqlen_k=512,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
    )
    base.update(kw)
    return AttentionRequest(**base)


class TestAttentionDispatch(unittest.TestCase):
    def test_rejects_unsupported_head_size(self):
        with self.assertRaises(ValueError):
            dispatch_attention(_attn(hdim_q=96, hdim_v=96))

    def test_rejects_unsupported_dtype(self):
        with self.assertRaises(ValueError):
            dispatch_attention(_attn(dtype="fp8"))

    def test_rejects_non_divisible_gqa(self):
        with self.assertRaises(ValueError):
            dispatch_attention(_attn(nhead_q=30, nhead_k=8))

    def test_rejects_unknown_arch(self):
        with self.assertRaises(ValueError):
            dispatch_attention(_attn(arch="gfx000"))

    def test_short_kv_routes_2d(self):
        r = dispatch_attention(_attn(seqlen_q=512, seqlen_k=512))
        self.assertEqual(r.spec.path, "2d")
        self.assertEqual(r.candidate.spec_id, "unified_2d")

    def test_sliding_window_routes_2d(self):
        r = dispatch_attention(_attn(seqlen_q=128, seqlen_k=4096, sliding_window=256))
        self.assertEqual(r.spec.path, "2d")

    def test_long_kv_small_grid_routes_3d(self):
        # decode (q=1) long kv, small grid -> 3d split-KV.
        r = dispatch_attention(
            _attn(batch=1, nhead_q=16, nhead_k=16, seqlen_q=1, seqlen_k=8192)
        )
        self.assertEqual(r.spec.path, "3d")
        self.assertEqual(r.candidate.spec_id, "unified_3d")

    def test_fp8_decode_gfx950_ocp_routes_3d(self):
        # bf16 compute + OCP fp8 K/V decode on gfx950 (OCP-native) -> 3d fp8 path.
        r = dispatch_attention(
            _attn(
                arch="gfx950",
                batch=1,
                nhead_q=16,
                nhead_k=16,
                seqlen_q=1,
                seqlen_k=8192,
                use_fp8=True,
                fp8_fnuz=False,
            )
        )
        self.assertEqual(r.spec.path, "3d")
        self.assertEqual(r.candidate.spec_id, "unified_3d")

    def test_fp8_decode_gfx942_fnuz_routes_3d(self):
        # bf16 compute + fnuz fp8 K/V decode on gfx942 (fnuz-native) -> 3d fp8 path.
        r = dispatch_attention(
            _attn(
                arch="gfx942",
                batch=1,
                nhead_q=16,
                nhead_k=16,
                seqlen_q=1,
                seqlen_k=8192,
                use_fp8=True,
                fp8_fnuz=True,
            )
        )
        self.assertEqual(r.spec.path, "3d")
        self.assertEqual(r.candidate.spec_id, "unified_3d")

    def test_fp8_decode_rejects_format_arch_mismatch(self):
        # OCP fp8 on gfx942 and fnuz fp8 on gfx950 both mis-decode -> no candidate.
        # Match on the "fnuz" reason so the format guard -- not some unrelated
        # future narrowing -- is what has to keep this red.
        for arch, fnuz in (("gfx942", False), ("gfx950", True)):
            with self.assertRaisesRegex(ValueError, "fnuz"):
                dispatch_attention(
                    _attn(
                        arch=arch,
                        batch=1,
                        nhead_q=16,
                        nhead_k=16,
                        seqlen_q=1,
                        seqlen_k=8192,
                        use_fp8=True,
                        fp8_fnuz=fnuz,
                    )
                )

    def test_declared_features_are_frozen(self):
        # A new candidate, or a features edit on an existing one, must be listed
        # in EXPECTED_FEATURES -- it cannot inherit a widened set unnoticed.
        actual = {
            c.name: set(c.capability.supports_features) for c in attention_candidates()
        }
        self.assertEqual(actual, EXPECTED_FEATURES)

    def test_feature_changes_spec_identity(self):
        # Derive one case per feature from the vocabulary: toggling a feature the
        # spec encodes must change kernel_name; the frozen causal/sinks gap must
        # not (until the consumer-side fix lands, which will flip these).
        self.assertEqual(set(_ENABLE_FEATURE), set(ATTENTION_FEATURES))
        base = _attn(batch=1, nhead_q=16, nhead_k=16, seqlen_q=1, seqlen_k=8192)
        base_name = dispatch_attention(base).spec.kernel_name()
        for feature in sorted(ATTENTION_FEATURES):
            with self.subTest(feature=feature):
                variant = _attn(
                    batch=1,
                    nhead_q=16,
                    nhead_k=16,
                    seqlen_q=1,
                    seqlen_k=8192,
                    **_ENABLE_FEATURE[feature],
                )
                variant_name = dispatch_attention(variant).spec.kernel_name()
                if feature in _IDENTITY_ENCODED:
                    self.assertNotEqual(base_name, variant_name)
                else:
                    self.assertEqual(base_name, variant_name)

    def test_large_grid_routes_2d(self):
        # many seqs/heads -> num_2d > target -> 2d even with long kv.
        r = dispatch_attention(
            _attn(batch=8, nhead_q=32, nhead_k=8, seqlen_q=1024, seqlen_k=1024)
        )
        self.assertEqual(r.spec.path, "2d")

    def test_path_candidates_are_mutually_exclusive(self):
        # Exactly one of (2d, 3d) supports any given problem.
        req = _attn(batch=1, nhead_q=16, nhead_k=16, seqlen_q=1, seqlen_k=8192)
        supported = [c for c in attention_candidates() if c.admits(req)[0]]
        self.assertEqual(len(supported), 1)

    def test_spec_records_dims(self):
        r = dispatch_attention(_attn(hdim_q=64, hdim_v=64, kv_block_size=32))
        self.assertEqual(r.spec.head_size, 64)
        self.assertEqual(r.spec.block_size, 32)

    def test_block_size_coverage(self):
        with self.assertRaises(ValueError):
            dispatch_attention(_attn(kv_block_size=128))  # not in {16,32,64}

    def test_unique_candidate_names(self):
        names = [c.name for c in attention_candidates()]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
