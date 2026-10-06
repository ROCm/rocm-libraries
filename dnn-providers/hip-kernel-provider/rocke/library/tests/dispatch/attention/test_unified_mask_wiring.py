# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Attention-band wiring on the unified (paged) attention candidates.

CPU-only. The unified kernels hard-wired the bottom-right causal diagonal. The mask is
now the cuDNN / hipDNN band ``(left = sliding_window, right = right_bound,
diagonal_alignment)``: top-left alignment, full attention (NO_MASK, the SDPA default in
cuDNN / FlashAttention / PyTorch) and non-causal windows / lookahead are distinct kernel
bodies on gfx950 and gfx942 and must never share a kernel name / compile-cache key with the
bottom-right causal one. Covers every unified candidate: ``unified_2d``, ``unified_3d``,
``gfx950_d256`` (2D D256 prefill) and ``d256_decode`` (3D D256 decode).
"""

from __future__ import annotations

import unittest
from dataclasses import replace

import kernels.common.attention_unified as au
from dispatch.attention import (
    AttentionMaskType,
    AttentionRequest,
    _problem,
    dispatch_attention,
)
from dispatch.attention.common import _attention_band

NO_MASK = AttentionMaskType.NO_MASK
TOP_LEFT = AttentionMaskType.TOP_LEFT_CAUSAL
BOTTOM_RIGHT = AttentionMaskType.BOTTOM_RIGHT_CAUSAL
SLIDING_WINDOW = AttentionMaskType.SLIDING_WINDOW
TL_ALIGN, BR_ALIGN = 0, 1  # hipDNN DiagonalAlignment ordinals


def _req(**kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=2048,
        seqlen_k=4096,
        hdim_q=128,
        hdim_v=128,
        arch="gfx950",
        dtype="bf16",
        num_cus=120,
    )
    base.update(kw)
    return AttentionRequest(**base)


# (expected spec_id, request kwargs). The prefill ones are 2D, the decode ones 3D.
_CASES = {
    "unified_2d": ("unified_2d", dict()),
    "unified_3d": ("unified_3d", dict(seqlen_q=1, seqlen_k=4096)),
    "gfx950_d256": (
        "gfx950_d256",
        dict(
            nhead_q=32,
            nhead_k=8,
            hdim_q=256,
            hdim_v=256,
            seqlen_q=4096,
            seqlen_k=8192,
        ),
    ),
    "d256_decode": (
        "d256_decode",
        dict(nhead_q=16, nhead_k=2, hdim_q=256, hdim_v=256, seqlen_q=1, seqlen_k=8192),
    ),
}


class _GfxArch:
    """Pin the memoized device arch the tiled selectors consult."""

    def __init__(self, arch):
        self._arch = arch

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = self._arch

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


def _problem_kw(**over):
    base = dict(
        total_q=1024,
        num_seqs=1,
        num_query_heads=16,
        num_kv_heads=2,
        head_size=128,
        block_size=16,
        max_seqlen_q=1024,
        max_seqlen_k=1024,
        dtype="fp16",
    )
    base.update(over)
    return base


class TestBandNormalization(unittest.TestCase):
    """request -> (causal_top_left, right_bound) the unified kernels compute."""

    def _band(self, **kw):
        return _attention_band(_req(**kw))

    def test_causal_kinds(self):
        self.assertEqual(self._band(mask_type=BOTTOM_RIGHT), (False, 0))
        self.assertEqual(self._band(mask_type=TOP_LEFT), (True, 0))
        # Equal length: the diagonals coincide, keep the existing kernel.
        eq = dict(seqlen_q=1024, seqlen_k=1024)
        self.assertEqual(self._band(mask_type=TOP_LEFT, **eq), (False, 0))

    def test_no_mask_is_full_attention(self):
        self.assertEqual(self._band(mask_type=NO_MASK), (False, -1))
        # Alignment is moot without any bound.
        self.assertEqual(
            self._band(mask_type=NO_MASK, diagonal_alignment=BR_ALIGN), (False, -1)
        )

    def test_no_mask_with_a_window_is_a_left_only_band(self):
        # cuDNN: a left bound with the right bound unset is a non-causal window.
        self.assertEqual(self._band(mask_type=NO_MASK, sliding_window=256), (True, -1))
        self.assertEqual(
            self._band(
                mask_type=NO_MASK, sliding_window=256, diagonal_alignment=BR_ALIGN
            ),
            (False, -1),
        )

    def test_two_sided_window_and_lookahead(self):
        for mask in (NO_MASK, SLIDING_WINDOW):
            with self.subTest(mask=mask):
                self.assertEqual(
                    self._band(mask_type=mask, sliding_window=256, right_bound=64),
                    (True, 64),
                )
                self.assertEqual(
                    self._band(mask_type=mask, right_bound=16, diagonal_alignment=1),
                    (False, 16),
                )

    def test_right_bound_zero_is_a_causal_window(self):
        self.assertEqual(
            self._band(mask_type=SLIDING_WINDOW, sliding_window=256, right_bound=0),
            (True, 0),
        )

    def test_equal_length_alignment_is_moot(self):
        eq = dict(seqlen_q=1024, seqlen_k=1024)
        self.assertEqual(
            self._band(mask_type=NO_MASK, sliding_window=128, right_bound=8, **eq),
            (False, 8),
        )

    def test_problem_carries_the_band(self):
        p = _problem(_req(mask_type=NO_MASK, sliding_window=256, right_bound=64))
        self.assertTrue(p.causal_top_left)
        self.assertEqual((p.right_bound, p.sliding_window), (64, 256))
        d = _problem(_req(mask_type=BOTTOM_RIGHT))
        self.assertTrue(d.default_mask)

    def test_invalid_bands_are_rejected_with_a_reason(self):
        bad = {
            "names a band": dict(mask_type=SLIDING_WINDOW),
            "right_bound must be": dict(mask_type=NO_MASK, right_bound=-2),
            "diagonal_alignment must be": dict(mask_type=NO_MASK, diagonal_alignment=2),
            "applies only to NO_MASK": dict(mask_type=BOTTOM_RIGHT, right_bound=8),
        }
        for needle, kw in bad.items():
            with self.subTest(needle=needle):
                with self.assertRaisesRegex(ValueError, needle):
                    dispatch_attention(_req(**kw))

    def test_causal_kinds_accept_an_explicit_zero_right_bound(self):
        for mask in (TOP_LEFT, BOTTOM_RIGHT):
            self.assertEqual(self._band(mask_type=mask, right_bound=0)[1], 0)

    def test_features(self):
        f = lambda **kw: _req(**kw).features()  # noqa: E731
        self.assertEqual(f(mask_type=NO_MASK), frozenset())
        self.assertEqual(
            f(mask_type=NO_MASK, sliding_window=256), {"band", "sliding_window"}
        )
        self.assertEqual(f(mask_type=NO_MASK, right_bound=16), {"band"})
        self.assertEqual(
            f(mask_type=NO_MASK, sliding_window=256, right_bound=0),
            {"causal", "sliding_window"},
        )
        self.assertEqual(
            f(
                mask_type=NO_MASK,
                sliding_window=256,
                right_bound=0,
                diagonal_alignment=BR_ALIGN,
            ),
            {"causal", "causal_bottom_right", "sliding_window"},
        )
        self.assertEqual(f(mask_type=TOP_LEFT), {"causal"})


class TestBandDistinctIdentity(unittest.TestCase):
    def test_every_unified_candidate_separates_top_left_from_bottom_right(self):
        for label, (spec_id, kw) in _CASES.items():
            with self.subTest(candidate=label):
                tl = dispatch_attention(_req(mask_type=TOP_LEFT, **kw))
                br = dispatch_attention(_req(mask_type=BOTTOM_RIGHT, **kw))
                self.assertEqual(tl.candidate.spec_id, spec_id)
                self.assertEqual(br.candidate.spec_id, spec_id)
                self.assertTrue(tl.spec.causal_top_left)
                self.assertFalse(br.spec.causal_top_left)
                self.assertNotEqual(tl.spec.kernel_name(), br.spec.kernel_name())
                self.assertIn("tl", tl.spec.kernel_name().split("_"))
                self.assertNotIn("tl", br.spec.kernel_name().split("_"))

    def test_top_left_at_equal_length_is_the_unchanged_kernel(self):
        for label, (_, kw) in _CASES.items():
            kw = dict(kw, seqlen_k=kw.get("seqlen_q", 2048))
            with self.subTest(candidate=label):
                try:
                    tl = dispatch_attention(_req(mask_type=TOP_LEFT, **kw))
                    br = dispatch_attention(_req(mask_type=BOTTOM_RIGHT, **kw))
                except ValueError:
                    continue  # shape leaves this cohort once sq == sk
                self.assertFalse(tl.spec.causal_top_left)
                self.assertEqual(tl.spec.kernel_name(), br.spec.kernel_name())

    def test_no_mask_is_its_own_kernel_on_every_unified_candidate(self):
        # Full attention must be neither aliased onto the causal kernels nor onto
        # top-left, even at equal length (where top-left and bottom-right coincide
        # but full attention does not).
        for label, (_, kw) in _CASES.items():
            for equal in (False, True):
                kw2 = dict(kw)
                if equal:
                    kw2["seqlen_k"] = kw2.get("seqlen_q", 2048)
                with self.subTest(candidate=label, equal_length=equal):
                    try:
                        nm = dispatch_attention(_req(mask_type=NO_MASK, **kw2))
                        br = dispatch_attention(_req(mask_type=BOTTOM_RIGHT, **kw2))
                        tl = dispatch_attention(_req(mask_type=TOP_LEFT, **kw2))
                    except ValueError:
                        continue  # shape leaves this cohort at equal length
                    self.assertEqual(nm.spec.right_bound, -1)
                    self.assertTrue(br.spec.causal)
                    self.assertIn("rbu", nm.spec.kernel_name().split("_"))
                    self.assertNotEqual(nm.spec.kernel_name(), br.spec.kernel_name())
                    self.assertNotEqual(nm.spec.kernel_name(), tl.spec.kernel_name())

    def test_each_band_is_its_own_kernel(self):
        bands = {
            "causal_window": dict(mask_type=TOP_LEFT, sliding_window=256),
            "left_only_tl": dict(mask_type=NO_MASK, sliding_window=256),
            "left_only_br": dict(
                mask_type=NO_MASK, sliding_window=256, diagonal_alignment=BR_ALIGN
            ),
            "two_sided_64": dict(mask_type=NO_MASK, sliding_window=256, right_bound=64),
            "two_sided_32": dict(mask_type=NO_MASK, sliding_window=256, right_bound=32),
            "lookahead_tl": dict(mask_type=NO_MASK, right_bound=16),
            "lookahead_br": dict(
                mask_type=NO_MASK, right_bound=16, diagonal_alignment=BR_ALIGN
            ),
            "full": dict(mask_type=NO_MASK),
            "causal_br": dict(mask_type=BOTTOM_RIGHT),
        }
        with _GfxArch("gfx950"):
            problems = {k: _problem(_req(**v)) for k, v in bands.items()}
            for key in (au._tiled_cache_key, au._cache_key, au._cheap_2d_sig):
                self.assertEqual(
                    len({key(p) for p in problems.values()}), len(bands), key.__name__
                )
        names = {
            k: dispatch_attention(_req(**v)).spec.kernel_name()
            for k, v in bands.items()
        }
        self.assertEqual(len(set(names.values())), len(bands), names)
        # Same band spelled two ways (kind vs trio) is the same kernel.
        self.assertEqual(
            names["causal_window"],
            dispatch_attention(
                _req(mask_type=SLIDING_WINDOW, sliding_window=256, right_bound=0)
            ).spec.kernel_name(),
        )

    def test_3d_cache_key_separates_masks(self):
        with _GfxArch("gfx950"):
            keys = {
                au._tiled_3d_cache_key(_problem(_req(seqlen_q=1, **kw)))
                for kw in (
                    dict(mask_type=BOTTOM_RIGHT),
                    dict(mask_type=TOP_LEFT),
                    dict(mask_type=NO_MASK),
                    dict(mask_type=NO_MASK, right_bound=8),
                )
            }
        self.assertEqual(len(keys), 4)

    def test_dense_serves_the_bands_it_implements_and_rejects_the_rest(self):
        # Dense implements the (left, right) band on the top-left diagonal and full
        # attention (including ragged cross-length). Anything it cannot compute exactly
        # is rejected rather than served as a different mask.
        for kw in (
            dict(mask_type=NO_MASK, sliding_window=256),
            dict(mask_type=NO_MASK, right_bound=16),
            dict(mask_type=NO_MASK, sliding_window=256, right_bound=64),
        ):
            with self.subTest(**kw):
                r = dispatch_attention(_req(algorithm="attention_dense", **kw))
                self.assertTrue(r.candidate.name.startswith("attention_gfx950_dense"))
        with self.subTest("bottom-right alignment + window"):
            with self.assertRaisesRegex(ValueError, "bottom-right alignment"):
                dispatch_attention(
                    _req(
                        algorithm="attention_dense",
                        mask_type=NO_MASK,
                        sliding_window=256,
                        diagonal_alignment=BR_ALIGN,
                    )
                )
        # Auto still routes the bands to the unified kernels (dense is opt-in).
        self.assertEqual(
            dispatch_attention(
                _req(mask_type=NO_MASK, sliding_window=256)
            ).candidate.spec_id,
            "unified_2d",
        )

    def test_default_cache_keys_are_unchanged_by_the_new_fields(self):
        # The suffix only appears for a non-default band, so existing keys are
        # identical to before the fields existed.
        with _GfxArch("gfx950"):
            p = _problem(_req(mask_type=BOTTOM_RIGHT))
            self.assertTrue(p.default_mask)
            for marker in ("top-left", "right_bound"):
                self.assertNotIn(marker, au._tiled_cache_key(p))
                self.assertNotIn(marker, au._cache_key(p))


class TestBandLowering(unittest.TestCase):
    def _lower(self, kernel):
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        text = _lower_kernel_to_llvm_python(kernel, arch="gfx950")
        return text.replace(kernel.name, "KERNEL")

    def test_every_band_is_a_distinct_body_on_2d_3d_and_scalar(self):
        from builders.common import attention_spec_builder as bld
        from kernels.common.attention_unified import (
            UnifiedAttention2DSpec,
            UnifiedAttention3DSpec,
            build_unified_attention_2d,
            build_unified_attention_3d,
        )
        from kernels.gfx950.attention_tiled_2d import build_unified_attention_2d_tiled
        from kernels.gfx950.attention_tiled_3d import build_unified_attention_3d_tiled

        bands = [
            dict(mask_type=BOTTOM_RIGHT),
            dict(mask_type=TOP_LEFT),
            dict(mask_type=NO_MASK),
            dict(mask_type=NO_MASK, right_bound=8),
            dict(mask_type=NO_MASK, right_bound=16),
            dict(mask_type=TOP_LEFT, sliding_window=128),
            dict(mask_type=NO_MASK, sliding_window=128),
            dict(mask_type=NO_MASK, sliding_window=128, right_bound=16),
        ]
        with _GfxArch("gfx950"):
            p2 = [_problem(_req(**b)) for b in bands]
            p3 = [_problem(_req(seqlen_q=1, **b)) for b in bands]
            s2 = [bld._spec_gfx950_generic(p) for p in p2]
            s3 = [bld._spec_generic_3d(p) for p in p3]
        builders = [
            (s2, lambda sp: build_unified_attention_2d_tiled(sp, arch="gfx950")),
            (s3, lambda sp: build_unified_attention_3d_tiled(sp, arch="gfx950")),
            (
                [UnifiedAttention2DSpec(p) for p in p2],
                lambda sp: build_unified_attention_2d(sp, arch="gfx950"),
            ),
            (
                [UnifiedAttention3DSpec(p, num_segments=8) for p in p3],
                lambda sp: build_unified_attention_3d(sp, arch="gfx950"),
            ),
        ]
        for specs, build in builders:
            names = [sp.kernel_name() for sp in specs]
            bodies = [self._lower(build(sp)) for sp in specs]
            self.assertEqual(len(set(names)), len(bands), names)
            self.assertEqual(len(set(bodies)), len(bands), names)

    def test_spec_rejects_right_bound_with_alibi_or_qq_bias(self):
        from builders.common import attention_spec_builder as bld

        with _GfxArch("gfx950"):
            spec = bld._spec_gfx950_generic(_problem(_req(mask_type=NO_MASK)))
        self.assertEqual(spec.right_bound, -1)
        for over in (dict(use_alibi=True), dict(use_qq_bias=True)):
            with self.subTest(**over), self.assertRaises(ValueError):
                replace(spec, **over)
        with self.assertRaises(ValueError):
            replace(spec, right_bound=-2)
        # A causal spec is not rejected on the right-bound rule, and a window is fine
        # with a right bound.
        for over in (dict(right_bound=0, use_alibi=True), dict(sliding_window=128)):
            try:
                replace(spec, **over)
            except ValueError as exc:  # other (transposed-VALU) rules may still apply
                self.assertNotIn("right_bound", str(exc))


class TestMaskGates(unittest.TestCase):
    def test_gates_reject_right_bound_with_alibi_or_qq_bias(self):
        for over in (dict(use_alibi=True), dict(use_qq_bias=True)):
            problem = au.UnifiedAttentionProblem(**_problem_kw(right_bound=-1, **over))
            ok, why = au._reject_mask_semantics(problem)
            self.assertFalse(ok)
            self.assertIn("causal masks", why)

    def test_a_window_with_any_right_bound_is_valid(self):
        for right in (-1, 0, 8):
            problem = au.UnifiedAttentionProblem(
                **_problem_kw(sliding_window=64, right_bound=right)
            )
            self.assertIsNone(au._reject_mask_semantics(problem))

    def test_invalid_problem_right_bound_is_rejected(self):
        with self.assertRaises(ValueError):
            au.UnifiedAttentionProblem(**_problem_kw(right_bound=-2))

    def test_tiled_gates_accept_bands_on_gfx950(self):
        with _GfxArch("gfx950"):
            for kw in (
                dict(mask_type=NO_MASK),
                dict(mask_type=NO_MASK, sliding_window=256, right_bound=32),
                dict(mask_type=TOP_LEFT, sliding_window=256),
            ):
                self.assertTrue(
                    au.supports_native_unified_attention_tiled(_problem(_req(**kw)))[0]
                )
                self.assertTrue(
                    au.supports_native_unified_attention_3d_tiled(
                        _problem(_req(seqlen_q=1, **kw)), arch="gfx950"
                    )[0]
                )


class TestBandOnGfx942(unittest.TestCase):
    """gfx942's tiled 2D/3D kernels implement the band as gfx950's do: NO_MASK is full
    attention, top-left / lookahead / non-causal windows are distinct bodies. The
    4-warp GQA cohorts keep top-left but decline a right bound other than 0, so such
    a problem builds the generic tiled 2D spec instead."""

    def _gfx942(self, **kw):
        return _req(**{"arch": "gfx942", "dtype": "fp16", **kw})

    def test_no_mask_is_full_attention(self):
        p = _problem(self._gfx942(mask_type=NO_MASK))
        self.assertFalse(p.default_mask)
        self.assertEqual((p.causal_top_left, p.right_bound), (False, -1))
        # cuDNN: a window on NO_MASK is a left-only (non-causal) band.
        p = _problem(self._gfx942(mask_type=NO_MASK, sliding_window=256))
        self.assertEqual((p.causal_top_left, p.right_bound), (True, -1))
        self.assertEqual(p.sliding_window, 256)
        r = dispatch_attention(self._gfx942(mask_type=NO_MASK))
        self.assertIn("rbu", r.spec.kernel_name().split("_"))

    def test_top_left_cross_length_routes_to_tiled(self):
        with _GfxArch("gfx942"):
            for kw in (dict(), dict(seqlen_q=1, seqlen_k=4096)):
                with self.subTest(**kw):
                    req = self._gfx942(mask_type=TOP_LEFT, **kw)
                    r = dispatch_attention(req)
                    self.assertIn(
                        r.candidate.spec_id,
                        ("unified_2d", "unified_3d", "gfx942_dense_pipe"),
                    )
                    self.assertTrue(r.spec.causal_top_left)
                    self.assertIn("tl", r.spec.kernel_name().split("_"))
                    p = _problem(req)
                    gate = (
                        au.supports_native_unified_attention_tiled(p)
                        if r.spec.path == "2d"
                        else au.supports_native_unified_attention_3d_tiled(
                            p, arch="gfx942"
                        )
                    )
                    self.assertTrue(gate[0], gate[1])

    def test_bands_dispatch_with_their_own_kernel_names(self):
        bands = {
            "rbu_window": (dict(mask_type=NO_MASK, sliding_window=256), "rbu"),
            "rb8": (dict(mask_type=NO_MASK, right_bound=8), "rb8"),
            "rb8_window": (
                dict(mask_type=SLIDING_WINDOW, sliding_window=256, right_bound=8),
                "rb8",
            ),
        }
        names = set()
        with _GfxArch("gfx942"):
            for label, (kw, token) in bands.items():
                with self.subTest(band=label):
                    r = dispatch_attention(self._gfx942(**kw))
                    self.assertIn(token, r.spec.kernel_name().split("_"))
                    names.add(r.spec.kernel_name())
        self.assertEqual(len(names), len(bands))

    def test_tiled_gates_accept_bands_on_gfx942(self):
        with _GfxArch("gfx942"):
            for kw in (
                dict(mask_type=NO_MASK),
                dict(mask_type=NO_MASK, sliding_window=256, right_bound=32),
                dict(mask_type=NO_MASK, right_bound=8),
                dict(mask_type=TOP_LEFT, sliding_window=256),
            ):
                with self.subTest(**kw):
                    ok, why = au.supports_native_unified_attention_tiled(
                        _problem(self._gfx942(**kw))
                    )
                    self.assertTrue(ok, why)
                    ok, why = au.supports_native_unified_attention_3d_tiled(
                        _problem(self._gfx942(seqlen_q=1, **kw)), arch="gfx942"
                    )
                    self.assertTrue(ok, why)

    def test_spec_builders_carry_the_band(self):
        from builders.common import attention_spec_builder as bld

        with _GfxArch("gfx942"):
            for kw, band in (
                (dict(mask_type=TOP_LEFT), (True, 0)),
                (dict(mask_type=NO_MASK), (False, -1)),
                (dict(mask_type=NO_MASK, sliding_window=256, right_bound=8), (True, 8)),
            ):
                for dtype in ("fp16", "bf16"):
                    with self.subTest(dtype=dtype, **kw):
                        req = _req(arch="gfx942", dtype=dtype, **kw)
                        s2 = bld._tiled_spec_from_problem(_problem(req))
                        self.assertEqual((s2.causal_top_left, s2.right_bound), band)
                        s3 = bld._tiled_3d_spec_from_problem(
                            _problem(replace(req, seqlen_q=1))
                        )
                        self.assertEqual((s3.causal_top_left, s3.right_bound), band)
            # The default mask builds the exact pre-band spec (fields at default).
            s = bld._tiled_spec_from_problem(
                _problem(self._gfx942(mask_type=BOTTOM_RIGHT))
            )
            self.assertEqual((s.causal_top_left, s.right_bound), (False, 0))

    def test_4warp_cohorts_keep_top_left_and_decline_a_right_bound(self):
        d256 = dict(
            dtype="bf16",
            nhead_q=16,
            nhead_k=2,
            hdim_q=256,
            hdim_v=256,
            seqlen_q=4096,
            seqlen_k=8192,
        )
        d128_swa = dict(dtype="fp16", sliding_window=256)
        with _GfxArch("gfx942"):
            for label, shape in (("d256", d256), ("d128_swa", d128_swa)):
                with self.subTest(cohort=label):
                    base = dict(_req(arch="gfx942").__dict__, **shape)
                    tl = _problem(AttentionRequest(**dict(base, mask_type=TOP_LEFT)))
                    br = _problem(
                        AttentionRequest(**dict(base, mask_type=BOTTOM_RIGHT))
                    )
                    self.assertTrue(au._gfx942_4warp_fast(br))
                    self.assertTrue(au._gfx942_4warp_fast(tl))
                    for right in (-1, 8):
                        rb = _problem(
                            AttentionRequest(
                                **dict(base, mask_type=NO_MASK, right_bound=right)
                            )
                        )
                        self.assertFalse(au._gfx942_4warp_fast(rb))
                        self.assertNotEqual(
                            au._tiled_cache_key(rb), au._tiled_cache_key(br)
                        )

    def test_fp8_band_routes_to_the_unified_kernels(self):
        with _GfxArch("gfx942"):
            req = self._gfx942(
                dtype="bf16",
                use_fp8=True,
                fp8_fnuz=True,
                mask_type=NO_MASK,
                sliding_window=256,
                right_bound=8,
            )
            r = dispatch_attention(req)
            self.assertEqual(r.candidate.spec_id, "unified_2d")
            name = r.spec.kernel_name().split("_")
            self.assertIn("fp8fnuz", name)
            self.assertIn("rb8", name)

    def test_dense_pipe_serves_the_band(self):
        from dispatch.attention import attention_candidates

        cand = next(
            c for c in attention_candidates() if c.name == "attention_gfx942_dense_pipe"
        )
        shape = dict(batch=2, nhead_q=16, nhead_k=16, seqlen_q=512, seqlen_k=1024)
        with _GfxArch("gfx942"):
            for kw, token in (
                (dict(mask_type=TOP_LEFT), "tl"),
                (dict(mask_type=NO_MASK), "rbu"),
                (dict(mask_type=NO_MASK, sliding_window=256, right_bound=16), "rb16"),
            ):
                with self.subTest(**kw):
                    req = self._gfx942(**shape, **kw)
                    ok, why = cand.admits(req)
                    self.assertTrue(ok, why)
                    r = dispatch_attention(req)
                    self.assertEqual(r.candidate.spec_id, "gfx942_dense_pipe")
                    self.assertIn(token, r.spec.kernel_name().split("_"))
            # The default mask keeps its pre-band kernel name.
            r = dispatch_attention(self._gfx942(**shape, mask_type=BOTTOM_RIGHT))
            for token in ("tl", "rbu"):
                self.assertNotIn(token, r.spec.kernel_name().split("_"))

    def test_a_causal_window_via_the_trio_is_accepted(self):
        req = self._gfx942(mask_type=SLIDING_WINDOW, sliding_window=256, right_bound=0)
        self.assertTrue(dispatch_attention(req).spec.causal)


class TestBandOnGfx1250(unittest.TestCase):
    """The gfx1250 tiled kernels are causal-only. NO_MASK is the request default, so
    (KNOWN GAP, pinned here) it keeps being served by the existing causal body there;
    every other band is rejected explicitly."""

    def test_no_mask_requests_keep_the_legacy_body(self):
        self.assertTrue(_problem(_req(mask_type=NO_MASK, arch="gfx1250")).default_mask)
        # Legacy convention: a window on the default mask is a causal window.
        p = _problem(_req(mask_type=NO_MASK, sliding_window=256, arch="gfx1250"))
        self.assertTrue(p.default_mask)
        self.assertEqual(p.sliding_window, 256)

    def test_non_causal_bands_are_rejected_with_a_reason(self):
        for kw in (
            dict(mask_type=NO_MASK, right_bound=16),
            dict(mask_type=NO_MASK, sliding_window=256, right_bound=64),
            dict(mask_type=SLIDING_WINDOW, sliding_window=256),
        ):
            with self.subTest(**kw):
                with self.assertRaisesRegex(ValueError, "gfx950 and gfx942 only"):
                    dispatch_attention(_req(arch="gfx1250", dtype="fp16", **kw))

    def test_direct_non_default_problems_are_rejected_by_tiled_gates(self):
        for over in (
            dict(causal_top_left=True),
            dict(right_bound=-1),
            dict(right_bound=8, sliding_window=64),
        ):
            with self.subTest(**over), _GfxArch("gfx1250"):
                problem = au.UnifiedAttentionProblem(**_problem_kw(**over))
                for gate in (
                    au.supports_native_unified_attention_tiled,
                    lambda p: au.supports_native_unified_attention_3d_tiled(
                        p, arch="gfx1250"
                    ),
                ):
                    ok, why = gate(problem)
                    self.assertFalse(ok)
                    self.assertIn("does not implement", why)


if __name__ == "__main__":
    unittest.main()
