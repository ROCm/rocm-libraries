# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Path-selection + coverage tests for the attention dispatcher family."""

from __future__ import annotations

import unittest
from fractions import Fraction

from dispatch.attention import (
    AttentionMaskType,
    AttentionRequest,
    attention_candidates,
    dispatch_attention,
)
from dispatch.attention.common import ATTENTION_FEATURES

# Request kwargs that turn each attention feature on. Keyed by the vocabulary so
# a feature added to ATTENTION_FEATURES forces an entry here (KeyError until
# listed) rather than inheriting silence.
_ENABLE_FEATURE = {
    "causal": {"mask_type": 1},
    "causal_bottom_right": {"mask_type": AttentionMaskType.BOTTOM_RIGHT_CAUSAL},
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
_IDENTITY_ENCODED = {"causal_bottom_right", "fp8", "sliding_window"}


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


class _IntegerLike:
    """Minimal graph-frontend integer scalar (``operator.index`` protocol)."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class TestAttentionMaskType(unittest.TestCase):
    def test_public_enum_exports_canonical_ordinals(self):
        import dispatch.attention as attention

        self.assertIs(attention.AttentionMaskType, AttentionMaskType)
        self.assertIn("AttentionMaskType", attention.__all__)
        self.assertEqual(
            [(member.name, member.value) for member in AttentionMaskType],
            [
                ("NO_MASK", 0),
                ("TOP_LEFT_CAUSAL", 1),
                ("BOTTOM_RIGHT_CAUSAL", 2),
                ("SLIDING_WINDOW", 3),
            ],
        )

    def test_enum_and_raw_int_share_normalized_and_request_hash_identity(self):
        enum_req = _attn(mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL)
        int_req = _attn(mask_type=2)

        self.assertEqual(enum_req.normalized(), int_req.normalized())
        self.assertEqual(enum_req.features(), frozenset({"causal"}))
        enum_result = dispatch_attention(enum_req)
        int_result = dispatch_attention(int_req)
        self.assertEqual(enum_result.candidate.name, "attention_unified_2d")
        self.assertEqual(enum_result.candidate.name, int_result.candidate.name)
        self.assertEqual(enum_result.spec, int_result.spec)
        self.assertEqual(
            enum_result.kernel_id.request_hash,
            int_result.kernel_id.request_hash,
        )

    def test_integer_like_exact_ordinals_are_accepted(self):
        for ordinal in range(4):
            with self.subTest(ordinal=ordinal):
                req = _attn(mask_type=_IntegerLike(ordinal))
                self.assertEqual(req.normalized()["mask_type"], ordinal)
                dispatch_attention(req)

    def test_non_integer_and_unknown_ordinals_are_rejected_clearly(self):
        invalid = (
            "2",
            2.0,
            1.5,
            Fraction(2, 1),
            Fraction(3, 2),
            -1,
            4,
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, r"mask_type.*exact integer ordinal"
                ):
                    dispatch_attention(_attn(mask_type=value))


class TestAttentionBottomRightRouting(unittest.TestCase):
    def test_moving_bottom_right_is_a_distinct_request_feature(self):
        moving = _attn(
            seqlen_q=512,
            seqlen_k=1024,
            mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        )
        equal_length = _attn(mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL)

        self.assertEqual(
            moving.features(), frozenset({"causal", "causal_bottom_right"})
        )
        self.assertEqual(equal_length.features(), frozenset({"causal"}))

    def test_bottom_right_auto_preserves_unified_prefill_and_decode(self):
        for sq, sk, path in ((256, 512, "2d"), (1, 8192, "3d")):
            with self.subTest(path=path):
                req = _attn(
                    batch=1,
                    nhead_q=8,
                    nhead_k=1,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
                    num_cus=120,
                )
                result = dispatch_attention(req)
                self.assertEqual(result.candidate.name, f"attention_unified_{path}")
                self.assertEqual(result.spec.path, path)

    def test_bottom_right_still_rejects_top_left_only_standalone_kernels(self):
        for arch, algorithm in (
            ("gfx942", "attention_dense"),
            ("gfx1250", "wmma_attention_fwd"),
        ):
            with self.subTest(arch=arch):
                req = _attn(
                    arch=arch,
                    seqlen_k=1024,
                    mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
                    algorithm=algorithm,
                )
                with self.assertRaises(ValueError):
                    dispatch_attention(req)


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

    def test_feature_changes_spec_identity(self):
        # Derive one case per feature from the vocabulary: toggling a feature the
        # spec encodes must change kernel_name; the frozen causal/sinks gap must
        # not (until the consumer-side fix lands, which will flip these).
        self.assertEqual(set(_ENABLE_FEATURE), set(ATTENTION_FEATURES))
        for feature in sorted(ATTENTION_FEATURES):
            with self.subTest(feature=feature):
                common = dict(
                    batch=1,
                    nhead_q=16,
                    nhead_k=16,
                    seqlen_q=1,
                    seqlen_k=8192,
                )
                base_kw = {}
                if feature == "causal_bottom_right":
                    # Dense kernels bake the diagonal; unified kernels obtain
                    # their shifted diagonal from runtime sequence lengths.
                    common.update(
                        seqlen_q=512,
                        seqlen_k=1024,
                        algorithm="attention_dense",
                        dense_persistent="off",
                    )
                    base_kw["mask_type"] = AttentionMaskType.TOP_LEFT_CAUSAL
                base = _attn(**common, **base_kw)
                variant = _attn(**common, **_ENABLE_FEATURE[feature])
                base_name = dispatch_attention(base).spec.kernel_name()
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
