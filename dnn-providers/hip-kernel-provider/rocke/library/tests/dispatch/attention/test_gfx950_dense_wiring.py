# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for gfx950 dense prefill dispatch wiring.

Covers:
  - Selection: a dense variant is reached only by pinning ``algorithm`` +
    ``spec_id``; ``tuning_id`` names one configuration of it
    (``auto``: the variant's default spec) and replays through dispatch
  - Knobs (waves_per_eu, persist_decode, ...) are candidate knobs, not request
    fields, and the kernel validates them
  - Sliding-window pass-through from AttentionRequest to the dense spec
  - Kernel name includes swa<W> token for sliding window
  - Ragged and sliding_window mutual exclusion constraint
  - Sinks support and capability metadata
  - SWA-sink composition (both features together)
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from itertools import islice

from dispatch.attention import (
    AttentionMaskType,
    AttentionRequest,
    AttentionTuningSpec,
    attention_candidates,
    attention_tuning_spec,
    dispatch_attention,
    iter_registered_attention_combos,
    tuning_spec_with_knobs,
)
from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES
from kernels.gfx942.attention_dense import Gfx942AttentionDenseSpec
from kernels.gfx950.attention_dense import (
    AttentionDenseSpec,
    GFX950_DENSE_LAYOUTS,
    Gfx950AttentionDenseSpec,
    attention_dense_grid,
    build_attention_dense,
    supports_attention_dense,
)
from rocke.dispatch.core import opt_in_probe

_GRID = "gfx950_dense_grid_default"
_PERSIST = "gfx950_dense_persist_default"
_WIDE = "gfx950_dense_persist_widedma_default"
_WIDE_NAME = "attention_gfx950_dense_persist_widedma_default"


def _gfx950_dense_req(**kw) -> AttentionRequest:
    """Helper to build a gfx950 dense attention request with common defaults."""
    base = dict(
        batch=2,
        nhead_q=8,
        nhead_k=1,  # GQA-8 (common for dense D64 cohort)
        seqlen_q=2048,
        seqlen_k=2048,
        hdim_q=64,
        hdim_v=64,
        arch="gfx950",
        dtype="bf16",
        mask_type=1,  # causal
    )
    base.update(kw)
    return AttentionRequest(**base)


def _spec(req: AttentionRequest, spec_id: str = _GRID, **knobs):
    """The kernel spec of dense variant ``spec_id`` for ``req``, ``knobs`` applied."""
    if knobs:
        return tuning_spec_with_knobs(req, spec_id, knobs).kernel_spec
    return attention_tuning_spec(req, spec_id).kernel_spec


def _admitting(req: AttentionRequest) -> set[str]:
    """Dense variants that admit ``req`` when pinned (what a sweep probes)."""
    return {
        c.name
        for c in attention_candidates()
        if c.algorithm == "attention_dense" and c.admits(opt_in_probe(req, c))[0]
    }


_LLAMA_8K = dict(
    batch=1,
    nhead_q=32,
    nhead_k=8,
    seqlen_q=8192,
    seqlen_k=8192,
    hdim_q=128,
    hdim_v=128,
    dtype="fp16",
)


class TestDenseSelection(unittest.TestCase):
    def test_pinned_spec_id_selects_the_concrete_gfx950_spec(self):
        result = dispatch_attention(
            _gfx950_dense_req(algorithm="attention_dense", spec_id=_GRID)
        )
        self.assertEqual(result.candidate.name, "attention_gfx950_dense_grid_default")
        self.assertIsInstance(result.spec, AttentionTuningSpec)
        self.assertEqual(result.spec.path, "dense")
        self.assertIsInstance(result.spec.kernel_spec, Gfx950AttentionDenseSpec)

    def test_gfx942_spec_id_selects_the_concrete_gfx942_spec(self):
        spec = _spec(_gfx950_dense_req(arch="gfx942"), "gfx942_dense")
        self.assertIsInstance(spec, Gfx942AttentionDenseSpec)

    def test_unpinned_requests_route_to_unified(self):
        self.assertEqual(
            dispatch_attention(_gfx950_dense_req()).candidate.name,
            "attention_unified_2d",
        )
        for kw in (
            dict(algorithm="attention_dense"),
            dict(spec_id=_GRID),
        ):
            with self.subTest(**kw), self.assertRaises(ValueError):
                dispatch_attention(_gfx950_dense_req(**kw))

    def test_pinned_spec_id_on_the_wrong_arch_is_refused(self):
        with self.assertRaises(ValueError):
            attention_tuning_spec(_gfx950_dense_req(arch="gfx942"), _GRID)

    def test_tuning_ids_replay_through_dispatch(self):
        req = _gfx950_dense_req(**_LLAMA_8K)
        specs = [
            s
            for _c, s in islice(
                iter_registered_attention_combos(req, candidate_prefix=_WIDE_NAME),
                24,
            )
        ]
        self.assertEqual(attention_tuning_spec(req, _WIDE), specs[0])
        for spec in specs[::4]:
            with self.subTest(tuning_id=spec.tuning_id):
                self.assertEqual(
                    attention_tuning_spec(req, _WIDE, spec.tuning_id), spec
                )
        with self.assertRaises(ValueError):
            attention_tuning_spec(req, _WIDE, "grid_default_wpe2@0000")


class TestDenseWavesPerEuWiring(unittest.TestCase):
    def test_default_policy_and_knob_override(self):
        default = _spec(_gfx950_dense_req())
        overridden = _spec(_gfx950_dense_req(), waves_per_eu=4)
        self.assertEqual(default.waves_per_eu, 2)
        self.assertEqual(overridden.waves_per_eu, 4)
        self.assertNotIn("wpe", default.kernel_name())
        self.assertIn("wpe4", overridden.kernel_name())
        self.assertNotEqual(default.kernel_name(), overridden.kernel_name())
        self.assertEqual(
            build_attention_dense(overridden, arch="gfx950").attrs["waves_per_eu"],
            4,
        )

    def test_invalid_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "waves_per_eu"):
            _spec(_gfx950_dense_req(), waves_per_eu=9)

    def _swept_waves(self, level: str) -> set[int]:
        req = _gfx950_dense_req(hdim_q=128, hdim_v=128)
        # The full level streams millions of knob settings; the WPE loop runs
        # inside each one, so the first knob set already carries every WPE.
        return {
            spec.kernel_spec.waves_per_eu
            for _candidate, spec in islice(
                iter_registered_attention_combos(
                    req,
                    candidate_prefix="attention_gfx950_dense_grid_default",
                    sweep_level=level,
                ),
                64,
            )
        }

    def test_production_and_full_sweeps_expand_wpe(self):
        self.assertEqual(self._swept_waves("production"), {2, 4})
        self.assertEqual(self._swept_waves("full"), {1, 2, 3, 4})


class TestDenseGqaPairWiring(unittest.TestCase):
    """Invariant-compatible shapes select a balanced GQA-local decode."""

    def test_exact_llama3_8b_prefill_auto_selects_gqa_pair(self):
        spec = _spec(_gfx950_dense_req(**_LLAMA_8K), _WIDE)
        self.assertEqual(spec.resolved_persist_decode, "gqa_pair")
        self.assertIn("gqapair", spec.kernel_name())
        self.assertTrue(spec.wide_lds_dma)
        self.assertIn("wdma", spec.kernel_name())
        self.assertEqual(spec.num_persistent, 256)

    def test_explicit_gqa_pair_knob_canonicalizes_to_the_default(self):
        """Pinning the decode auto already picks is the default kernel: the
        knob is dropped, so the id is the default's and no duplicate is minted."""
        req = _gfx950_dense_req(**_LLAMA_8K)
        pinned = tuning_spec_with_knobs(req, _WIDE, {"persist_decode": "gqa_pair"})
        self.assertEqual(pinned, attention_tuning_spec(req, _WIDE))
        self.assertEqual(pinned.knobs, ())
        self.assertEqual(pinned.kernel_spec.resolved_persist_decode, "gqa_pair")
        self.assertIn("gqapair", pinned.kernel_name())

    def test_invalid_dense_decode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "persist_decode"):
            _spec(_gfx950_dense_req(), _PERSIST, persist_decode="not-a-decode")

    def test_s4096_shape_selects_two_phase_pair_and_wide_dma(self):
        req = _gfx950_dense_req(**{**_LLAMA_8K, "seqlen_q": 4096, "seqlen_k": 4096})
        spec = _spec(req, _WIDE)
        self.assertEqual(spec.resolved_persist_decode, "gqa_pair_2phase")
        self.assertTrue(spec.wide_lds_dma)
        self.assertIn("gqapair2", spec.kernel_name())
        self.assertIn("wdma", spec.kernel_name())

    def test_bf16_shape_uses_invariant_compatible_fast_path(self):
        spec = _spec(_gfx950_dense_req(**{**_LLAMA_8K, "dtype": "bf16"}), _WIDE)
        self.assertEqual(spec.resolved_persist_decode, "gqa_pair")
        self.assertTrue(spec.wide_lds_dma)

    def test_s2048_h64_selects_two_phase_pair(self):
        req = _gfx950_dense_req(
            **{**_LLAMA_8K, "nhead_q": 64, "seqlen_q": 2048, "seqlen_k": 2048}
        )
        spec = _spec(req, _WIDE)
        self.assertEqual(spec.resolved_persist_decode, "gqa_pair_2phase")
        self.assertTrue(spec.wide_lds_dma)

    def test_explicit_two_phase_pair_knob_canonicalizes_to_the_default(self):
        req = _gfx950_dense_req(**{**_LLAMA_8K, "seqlen_q": 4096, "seqlen_k": 4096})
        pinned = tuning_spec_with_knobs(
            req, _WIDE, {"persist_decode": "gqa_pair_2phase"}
        )
        self.assertEqual(pinned, attention_tuning_spec(req, _WIDE))
        self.assertEqual(pinned.kernel_spec.resolved_persist_decode, "gqa_pair_2phase")
        # A decode auto does not pick is kept, and gets its own id.
        other = tuning_spec_with_knobs(req, _WIDE, {"persist_decode": "hkv_major"})
        self.assertEqual(dict(other.knobs), {"persist_decode": "hkv_major"})
        self.assertNotEqual(other.tuning_id, pinned.tuning_id)

    def test_exact_shape_with_sinks_keeps_gqa_pair(self):
        req = _gfx950_dense_req(**_LLAMA_8K, use_sinks=True)
        spec = _spec(req, _PERSIST)
        self.assertEqual(spec.resolved_persist_decode, "gqa_pair")
        self.assertTrue(spec.use_sinks)
        self.assertFalse(spec.wide_lds_dma)
        # No dispatch gate stricter than the kernel: wide DMA admits sinks too.
        self.assertTrue(_spec(req, _WIDE).use_sinks)

    def test_sliding_window_falls_back_to_qb_major(self):
        req = _gfx950_dense_req(**_LLAMA_8K, sliding_window=512)
        for spec_id in (_PERSIST, _WIDE):
            with self.subTest(spec_id=spec_id):
                self.assertEqual(
                    _spec(req, spec_id).resolved_persist_decode, "qb_major"
                )

    def test_mha_falls_back_to_qb_major(self):
        spec = _spec(_gfx950_dense_req(**{**_LLAMA_8K, "nhead_k": 32}), _WIDE)
        self.assertEqual(spec.resolved_persist_decode, "qb_major")
        self.assertTrue(spec.wide_lds_dma)
        self.assertNotIn("gqapair", spec.kernel_name())


class TestDenseBottomRightWiring(unittest.TestCase):
    @staticmethod
    def _moving_req(mask_type, **kw):
        base = dict(
            batch=1,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=8192,
            seqlen_k=12288,
            hdim_q=128,
            hdim_v=128,
            dtype="fp16",
            mask_type=mask_type,
        )
        base.update(kw)
        return _gfx950_dense_req(**base)

    def test_grid_variant_runs_aligned_and_ragged_moving_masks(self):
        shapes = (
            ("aligned", 8192, 12288, False),
            ("ragged", 8180, 12270, True),
        )
        for mask_type in (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 2):
            for name, sq, sk, ragged in shapes:
                with self.subTest(mask_type=mask_type, shape=name):
                    spec = _spec(self._moving_req(mask_type, seqlen_q=sq, seqlen_k=sk))
                    self.assertTrue(spec.causal)
                    self.assertTrue(spec.causal_bottom_right)
                    self.assertEqual(spec.ragged, ragged)
                    self.assertFalse(spec.persistent)
                    self.assertFalse(spec.wide_lds_dma)
                    self.assertIn("br", spec.kernel_name().split("_"))
                    self.assertNotIn("persist", spec.kernel_name())
                    self.assertNotIn("wdma", spec.kernel_name())

    def test_persistent_variants_refuse_moving_bottom_right(self):
        for mask_type in (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 2):
            for spec_id in (_PERSIST, _WIDE):
                with self.subTest(mask_type=mask_type, spec_id=spec_id):
                    with self.assertRaisesRegex(ValueError, "bottom.right"):
                        _spec(self._moving_req(mask_type), spec_id)

    def test_equal_length_bottom_right_preserves_gqa_pair_and_wide_dma(self):
        mask_pairs = (
            (AttentionMaskType.TOP_LEFT_CAUSAL, 2),
            (1, AttentionMaskType.BOTTOM_RIGHT_CAUSAL),
        )
        for top_left, bottom_right in mask_pairs:
            with self.subTest(top_left=top_left, bottom_right=bottom_right):
                top_left_spec = _spec(
                    _gfx950_dense_req(mask_type=top_left, **_LLAMA_8K), _WIDE
                )
                bottom_right_spec = _spec(
                    _gfx950_dense_req(mask_type=bottom_right, **_LLAMA_8K), _WIDE
                )
                self.assertEqual(bottom_right_spec, top_left_spec)
                self.assertFalse(bottom_right_spec.causal_bottom_right)
                self.assertTrue(bottom_right_spec.persistent)
                self.assertEqual(bottom_right_spec.resolved_persist_decode, "gqa_pair")
                self.assertTrue(bottom_right_spec.wide_lds_dma)


class TestDenseGeometrySpec(unittest.TestCase):
    """Tile geometry is explicit, validated, and kernel-identity-safe."""

    @staticmethod
    def _spec(**kw) -> AttentionDenseSpec:
        base = dict(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=32,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            dtype="fp16",
        )
        base.update(kw)
        return AttentionDenseSpec(**base)

    def test_dispatch_captures_named_default_geometry(self):
        spec = _spec(
            _gfx950_dense_req(
                batch=1,
                nhead_q=32,
                nhead_k=8,
                hdim_q=128,
                hdim_v=128,
                dtype="fp16",
            )
        )
        defaults = DENSE_TILE_GEOMETRIES["default"]
        layout = GFX950_DENSE_LAYOUTS["default"]
        self.assertEqual(spec.block_m, defaults["block_m"])
        self.assertEqual(spec.block_n, defaults["block_n"])
        self.assertEqual(spec.lds_v_row_pad, layout["lds_v_row_pad"])

    def test_block_m_controls_grid_and_kernel_identity(self):
        default = self._spec()
        bm128 = replace(default, block_m=128)
        self.assertEqual(attention_dense_grid(default), (2, 32, 1))
        self.assertEqual(attention_dense_grid(bm128), (4, 32, 1))
        self.assertNotEqual(default.kernel_name(), bm128.kernel_name())
        self.assertIn("bm128", bm128.kernel_name())
        ok, why = supports_attention_dense(bm128)
        self.assertTrue(ok, why)

    def test_v_row_pad_controls_kernel_identity(self):
        default = self._spec()
        vpad16 = replace(default, lds_v_row_pad=16)
        self.assertNotEqual(default.kernel_name(), vpad16.kernel_name())
        self.assertIn("vpad16", vpad16.kernel_name())

    def test_unknown_block_m_is_rejected_by_gfx950_support(self):
        spec = self._spec(seqlen_q=384, block_m=192)
        ok, why = supports_attention_dense(spec)
        self.assertFalse(ok)
        self.assertIn("block_m", why)

    def test_wide_dma_requires_default_v_row_pad(self):
        with self.assertRaisesRegex(ValueError, "K/V slab padding"):
            self._spec(
                persistent=True,
                num_persistent=16,
                persist_decode="gqa_pair",
                wide_lds_dma=True,
                lds_v_row_pad=16,
            )


class TestDenseSlidingWindowWiring(unittest.TestCase):
    """Verify sliding_window passes through dispatch to the dense spec."""

    def test_sliding_window_zero_by_default(self):
        """Without sliding_window in request, spec gets 0 (full causal)."""
        spec = _spec(_gfx950_dense_req())
        self.assertEqual(spec.sliding_window, 0)
        self.assertNotIn("swa", spec.kernel_name())

    def test_sliding_window_passes_through_to_spec(self):
        spec = _spec(_gfx950_dense_req(sliding_window=128))
        self.assertEqual(spec.sliding_window, 128)

    def test_sliding_window_appears_in_kernel_name(self):
        spec = _spec(_gfx950_dense_req(sliding_window=256))
        self.assertIn("swa256", spec.kernel_name())

    def test_different_window_sizes(self):
        for window in [64, 128, 256, 512]:
            with self.subTest(window=window):
                spec = _spec(_gfx950_dense_req(sliding_window=window))
                self.assertEqual(spec.sliding_window, window)
                self.assertIn(f"swa{window}", spec.kernel_name())

    def test_sliding_window_with_persistent_mode(self):
        """Sliding window works with persistent mode (both appear in kernel_name)."""
        spec = _spec(_gfx950_dense_req(sliding_window=128, seqlen_q=4096), _PERSIST)
        self.assertEqual(spec.sliding_window, 128)
        self.assertTrue(spec.persistent)
        kname = spec.kernel_name()
        self.assertIn("swa128", kname)
        self.assertIn("persist", kname)

    def test_sliding_window_with_ragged_shape_rejected_at_dispatch(self):
        """Ragged + sliding_window is rejected by the spec validator, so no
        dense variant admits the request."""
        req = _gfx950_dense_req(seqlen_q=500, seqlen_k=500, sliding_window=128)
        with self.assertRaises(ValueError) as cm:
            _spec(req)
        err_msg = str(cm.exception).lower()
        self.assertIn("ragged", err_msg)
        self.assertIn("sliding_window", err_msg)

    def test_sliding_window_without_ragged_accepted(self):
        spec = _spec(
            _gfx950_dense_req(seqlen_q=2048, seqlen_k=2048, sliding_window=256)
        )
        self.assertFalse(spec.ragged)
        self.assertEqual(spec.sliding_window, 256)


class TestDenseCapabilitySlidingWindow(unittest.TestCase):
    """Verify dense candidate capability metadata includes sliding_window."""

    @staticmethod
    def _wide():
        return next(c for c in attention_candidates() if c.name == _WIDE_NAME)

    def test_sliding_window_in_supports_features(self):
        self.assertIn("sliding_window", self._wide().capability.supports_features)

    def test_sinks_also_in_supports_features(self):
        self.assertIn("sinks", self._wide().capability.supports_features)

    def test_causal_in_supports_features(self):
        self.assertIn("causal", self._wide().capability.supports_features)

    def test_every_d128_variant_admits_sinks(self):
        """Admission is the kernel's: wide DMA runs sinks, so its variant admits."""
        names = _admitting(
            _gfx950_dense_req(
                use_sinks=True, hdim_q=128, hdim_v=128, nhead_q=32, nhead_k=8
            )
        )
        self.assertIn(_WIDE_NAME, names)
        self.assertIn("attention_gfx950_dense_persist_default", names)

    def test_every_d128_variant_admits_sliding_window(self):
        names = _admitting(
            _gfx950_dense_req(
                sliding_window=256, hdim_q=128, hdim_v=128, nhead_q=32, nhead_k=8
            )
        )
        self.assertIn(_WIDE_NAME, names)
        self.assertIn("attention_gfx950_dense_persist_default", names)


class TestSWASinkComposition(unittest.TestCase):
    """Verify SWA-sink (sliding_window + sinks) composes correctly."""

    def test_swa_sink_both_flags_pass_through(self):
        spec = _spec(_gfx950_dense_req(sliding_window=256, use_sinks=True))
        self.assertEqual(spec.sliding_window, 256)
        self.assertTrue(spec.use_sinks)

    def test_swa_sink_kernel_name_has_both_tokens(self):
        kname = _spec(
            _gfx950_dense_req(sliding_window=128, use_sinks=True)
        ).kernel_name()
        self.assertIn("swa128", kname)
        self.assertIn("sinks", kname)


class TestSinksValidation(unittest.TestCase):
    """Verify run_attention_dense_torch validates sinks parameter correctly."""

    def test_sinks_rejected_when_use_sinks_false(self):
        """Providing sinks when spec.use_sinks=False raises ValueError."""
        from types import SimpleNamespace

        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=8,
            num_kv_heads=8,
            head_size=64,
            dtype="bf16",
            use_sinks=False,  # Sinks disabled
        )

        qshape = (spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size)
        kvshape = (spec.batch, spec.seqlen_kv, spec.num_kv_heads, spec.head_size)
        q = SimpleNamespace(shape=qshape)
        k = SimpleNamespace(shape=kvshape)
        v = SimpleNamespace(shape=kvshape)
        out = SimpleNamespace(shape=qshape)
        # Minimal mock for sinks (validation checks if it's not None)
        sinks = [0.0] * spec.num_query_heads

        with self.assertRaises(ValueError) as cm:
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=0.125,
                sinks=sinks,  # Should be rejected
            )

        self.assertIn("sinks provided but spec.use_sinks is False", str(cm.exception))

    def test_sinks_required_when_use_sinks_true(self):
        """Not providing sinks when spec.use_sinks=True raises ValueError."""
        from types import SimpleNamespace

        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=8,
            num_kv_heads=8,
            head_size=64,
            dtype="bf16",
            use_sinks=True,  # Sinks enabled
        )

        qshape = (spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size)
        kvshape = (spec.batch, spec.seqlen_kv, spec.num_kv_heads, spec.head_size)
        q = SimpleNamespace(shape=qshape)
        k = SimpleNamespace(shape=kvshape)
        v = SimpleNamespace(shape=kvshape)
        out = SimpleNamespace(shape=qshape)

        with self.assertRaises(ValueError) as cm:
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=0.125,
                sinks=None,  # Missing sinks
            )

        self.assertIn("spec.use_sinks=True requires sinks", str(cm.exception))

    def test_sinks_wrong_shape_rejected(self):
        """Sinks with incorrect shape raises ValueError."""
        from types import SimpleNamespace

        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=8,
            num_kv_heads=8,
            head_size=64,
            dtype="bf16",
            use_sinks=True,
        )

        qshape = (spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size)
        kvshape = (spec.batch, spec.seqlen_kv, spec.num_kv_heads, spec.head_size)
        q = SimpleNamespace(shape=qshape, dtype="bfloat16")
        k = SimpleNamespace(shape=kvshape)
        v = SimpleNamespace(shape=kvshape)
        out = SimpleNamespace(shape=qshape)
        # Mock sinks with wrong shape (16 instead of spec.num_query_heads=8)
        wrong_size = spec.num_query_heads * 2
        sinks = SimpleNamespace(
            shape=(wrong_size,),
            dtype="bfloat16",
            is_contiguous=lambda: True,
            is_cuda=True,
        )

        with self.assertRaises(ValueError) as cm:
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=0.125,
                sinks=sinks,
            )

        err_msg = str(cm.exception)
        self.assertIn("sinks must have shape", err_msg)
        self.assertIn(f"({spec.num_query_heads},)", err_msg)
        self.assertIn(f"({wrong_size},)", err_msg)

    def test_sinks_wrong_dtype_rejected(self):
        """Sinks with dtype mismatch raises ValueError."""
        from types import SimpleNamespace

        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=8,
            num_kv_heads=8,
            head_size=64,
            dtype="bf16",
            use_sinks=True,
        )

        qshape = (spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size)
        kvshape = (spec.batch, spec.seqlen_kv, spec.num_kv_heads, spec.head_size)
        q = SimpleNamespace(shape=qshape, dtype="bfloat16")
        k = SimpleNamespace(shape=kvshape)
        v = SimpleNamespace(shape=kvshape)
        out = SimpleNamespace(shape=qshape)
        # Mock sinks with wrong dtype (float16 instead of bfloat16)
        sinks = SimpleNamespace(
            shape=(spec.num_query_heads,),
            dtype="float16",
            is_contiguous=lambda: True,
            is_cuda=True,
        )

        with self.assertRaises(ValueError) as cm:
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=0.125,
                sinks=sinks,
            )

        err_msg = str(cm.exception)
        self.assertIn("sinks dtype", err_msg)
        self.assertIn("must match q dtype", err_msg)

    def test_sinks_non_contiguous_rejected(self):
        """Non-contiguous sinks raises ValueError."""
        from types import SimpleNamespace

        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=8,
            num_kv_heads=8,
            head_size=64,
            dtype="bf16",
            use_sinks=True,
        )

        qshape = (spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size)
        kvshape = (spec.batch, spec.seqlen_kv, spec.num_kv_heads, spec.head_size)
        q = SimpleNamespace(shape=qshape, dtype="bfloat16")
        k = SimpleNamespace(shape=kvshape)
        v = SimpleNamespace(shape=kvshape)
        out = SimpleNamespace(shape=qshape)
        # Mock non-contiguous sinks
        sinks = SimpleNamespace(
            shape=(spec.num_query_heads,),
            dtype="bfloat16",
            is_contiguous=lambda: False,
            is_cuda=True,
        )

        with self.assertRaises(ValueError) as cm:
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=0.125,
                sinks=sinks,
            )

        self.assertIn("sinks must be contiguous", str(cm.exception))

    def test_sinks_cpu_tensor_rejected(self):
        """CPU sinks tensor raises ValueError (would silently produce garbage)."""
        from types import SimpleNamespace

        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=8,
            num_kv_heads=8,
            head_size=64,
            dtype="bf16",
            use_sinks=True,
        )

        qshape = (spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size)
        kvshape = (spec.batch, spec.seqlen_kv, spec.num_kv_heads, spec.head_size)
        q = SimpleNamespace(shape=qshape, dtype="bfloat16")
        k = SimpleNamespace(shape=kvshape)
        v = SimpleNamespace(shape=kvshape)
        out = SimpleNamespace(shape=qshape)
        # Mock CPU tensor (is_cuda=False)
        sinks = SimpleNamespace(
            shape=(spec.num_query_heads,),
            dtype="bfloat16",
            is_contiguous=lambda: True,
            is_cuda=False,
        )

        with self.assertRaises(ValueError) as cm:
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=0.125,
                sinks=sinks,
            )

        self.assertEqual(str(cm.exception), "sinks must be a CUDA tensor")


class TestGfx950DenseVariants(unittest.TestCase):
    """One registered candidate per frozen (tile x persist x wide-DMA) combo."""

    def test_six_dense_candidates_are_registered(self):
        from dispatch.attention.gfx950_dense import GFX950_DENSE_VARIANTS

        names = [c.name for c in attention_candidates()]
        expected = [v.candidate_name for v in GFX950_DENSE_VARIANTS]
        self.assertEqual(len(expected), 6)
        for variant in GFX950_DENSE_VARIANTS:
            with self.subTest(variant=variant.variant_id):
                self.assertIn(variant.candidate_name, names)
                self.assertEqual(variant.candidate_name, f"attention_{variant.spec_id}")

    def test_d128_admits_all_six_dense_variants(self):
        from dispatch.attention.gfx950_dense import GFX950_DENSE_VARIANTS

        req = _gfx950_dense_req(
            hdim_q=128, hdim_v=128, nhead_q=32, nhead_k=8, seqlen_q=2048, seqlen_k=2048
        )
        self.assertEqual(
            _admitting(req), {v.candidate_name for v in GFX950_DENSE_VARIANTS}
        )

    def test_d64_refuses_wide_dma_variants(self):
        names = _admitting(_gfx950_dense_req())
        self.assertIn("attention_gfx950_dense_grid_default", names)
        self.assertNotIn(_WIDE_NAME, names)
        self.assertNotIn("attention_gfx950_dense_persist_widedma_bm128", names)

    def test_spec_id_selects_the_tile(self):
        req = _gfx950_dense_req(hdim_q=128, hdim_v=128, nhead_q=32, nhead_k=8)
        spec = _spec(req, "gfx950_dense_persist_widedma_bm128")
        self.assertEqual(spec.block_m, 128)
        self.assertTrue(spec.persistent)
        self.assertTrue(spec.wide_lds_dma)
        self.assertEqual(_spec(req, _WIDE).block_m, 256)

    def test_llama3_8b_s8192_needs_an_explicit_pin(self):
        req = _gfx950_dense_req(**_LLAMA_8K)
        self.assertEqual(dispatch_attention(req).candidate.name, "attention_unified_2d")
        pinned = dispatch_attention(
            replace(req, algorithm="attention_dense", spec_id=_WIDE)
        )
        self.assertEqual(pinned.candidate.name, _WIDE_NAME)
        spec = pinned.spec.kernel_spec
        self.assertEqual(spec.block_m, 256)
        self.assertTrue(spec.persistent)
        self.assertTrue(spec.wide_lds_dma)
        self.assertEqual(spec.resolved_persist_decode, "gqa_pair")

    def test_registered_combos_include_dense_not_unified_2d(self):
        from dispatch.attention import registered_attention_combos

        req = _gfx950_dense_req(
            hdim_q=128,
            hdim_v=128,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=2048,
            seqlen_k=2048,
        )
        names = {
            c.name for c, _spec in registered_attention_combos(req, tuning_sample=2)
        }
        self.assertIn(_WIDE_NAME, names)
        self.assertIn("attention_gfx950_dense_grid_bm128", names)
        self.assertNotIn("attention_unified_2d", names)

    def test_d256_combos_exclude_dense_and_routing_labels(self):
        from dispatch.attention import registered_attention_combos

        req = _gfx950_dense_req(
            hdim_q=256,
            hdim_v=256,
            nhead_q=16,
            nhead_k=2,
            seqlen_q=2048,
            seqlen_k=2048,
            dtype="bf16",
        )
        combos = registered_attention_combos(req, tuning_sample=2)
        names = {c.name for c, _spec in combos}
        self.assertNotIn("attention_gfx950_d256", names)
        self.assertFalse(
            any(c.algorithm == "attention_dense" for c, _spec in combos),
            names,
        )


if __name__ == "__main__":
    unittest.main()
