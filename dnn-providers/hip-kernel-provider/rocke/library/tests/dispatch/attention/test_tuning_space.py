# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dependency-valid, unique, complete attention tuning space."""

from __future__ import annotations

import dataclasses
import unittest
from dataclasses import replace
from itertools import islice

from dispatch.attention import (
    AttentionRequest,
    attention_candidates,
    attention_execution_candidates,
    attention_tuning_spec,
    dispatch_attention,
    dispatch_attention_all,
    iter_registered_attention_combos,
    tuning_spec_with_knobs,
)
from dispatch.attention.axes import (
    _CODEPATH_KNOBS,
    DENSE_BASE_RELATIVE_KNOBS,
    DENSE_LOOP_FIELDS,
    DENSE_PROBLEM_FIELDS,
    DENSE_UNTUNABLE_KNOBS,
    DENSE_VARIANT_FIELDS,
    KNOWN_WRONG_KNOBS,
    tuning_axes,
)
from dispatch.attention.common import _problem
from dispatch.attention.dense_rules import resolve_dense_num_persistent
from dispatch.attention.gfx950_unified import GFX950_TUNING_VARIANTS
from dispatch.attention.tuning_specs import _SEMANTIC_FIELDS
from dispatch.attention.unified_rules import (
    _gfx950_2d_lds_bytes,
    canonicalize_tuning_spec,
    unified_space,
)
from rocke.core.arch import ArchTarget
from kernels.common.attention_unified import _tiled_2d_impl, _tiled_3d_impl
from kernels.gfx942.attention_dense import (
    Gfx942AttentionDenseSpec,
    supports_attention_dense as supports_gfx942,
)
from kernels.gfx942.attention_tiled_2d import UnifiedAttention2DTiledSpec as Gfx942Spec
from kernels.gfx950.attention_dense import (
    Gfx950AttentionDenseSpec,
    supports_attention_dense,
)


def _request(arch="gfx950", **kw):
    base = dict(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=1024,
        seqlen_k=1024,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        dtype="bf16" if arch == "gfx950" else "fp16",
    )
    base.update(kw)
    return AttentionRequest(**base)


def _specs_for(prefix, n=200, **req_kw):
    """The first ``n`` specs of the (possibly million-spec) sweep stream."""
    candidate = next(c for c in attention_candidates() if c.name.startswith(prefix))
    req = replace(
        _request(**req_kw),
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
    )
    return candidate, req, tuple(islice(candidate.sweep_space(req), n))


def _sampled(prefix, n, seed=0, **req_kw):
    candidate = next(c for c in attention_candidates() if c.name.startswith(prefix))
    req = replace(
        _request(**req_kw),
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
    )
    return tuple(candidate.sample_space(req, n, seed))


_GEOMETRY_FIELDS = frozenset(
    {
        "num_warps",
        "block_m_per_warp",
        "tile_size",
        "waves_per_eu",
        "num_segments",
        "tile_size_override",
    }
)


class TestTuningSpace(unittest.TestCase):
    def test_every_kernel_tuning_field_is_swept(self):
        for arch in ("gfx942", "gfx950"):
            for path, spec_type in (
                ("2d", _tiled_2d_impl(arch)[0]),
                ("3d", _tiled_3d_impl(arch)[0]),
            ):
                with self.subTest(arch=arch, path=path):
                    swept = {
                        name
                        for axis in tuning_axes(arch, path)
                        for choice in axis.choices
                        for name, _value in choice
                    }
                    codepath = {
                        name
                        for (owner, _cp), knobs in _CODEPATH_KNOBS.items()
                        if owner == arch
                        for name in knobs
                    }
                    fields = {f.name for f in dataclasses.fields(spec_type)}
                    missing = (
                        fields
                        - _SEMANTIC_FIELDS
                        - _GEOMETRY_FIELDS
                        - swept
                        - codepath
                        - KNOWN_WRONG_KNOBS[arch]
                    )
                    self.assertFalse(missing, sorted(missing))

    def test_sampling_draws_distinct_reproducible_specs(self):
        prefix = "attention_gfx942_u2d_transposed_x8_nw2_mw32_t4xb_llvm"
        first = _sampled(prefix, 32, seed=3, arch="gfx942")
        again = _sampled(prefix, 32, seed=3, arch="gfx942")
        other = _sampled(prefix, 32, seed=4, arch="gfx942")
        ids = [s.tuning_id for s in first]
        self.assertEqual(len(ids), 32)
        self.assertEqual(len(set(ids)), 32)
        self.assertEqual(ids, [s.tuning_id for s in again])
        self.assertNotEqual(ids, [s.tuning_id for s in other])

    def test_sampling_stops_at_a_small_space(self):
        specs = _sampled(
            "attention_gfx950_u3d_splitkv_seg64_t1xb",
            256,
            seqlen_q=1,
            seqlen_k=4096,
        )
        self.assertEqual(len(specs), 5)

    def test_tuning_ids_are_unique_and_hashed(self):
        _candidate, _req, specs = _specs_for(
            "attention_gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm"
        )
        ids = [s.tuning_id for s in specs]
        self.assertTrue(ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all("@" in tid for tid in ids))

    def test_pinned_tuning_id_roundtrips(self):
        candidate, req, specs = _specs_for(
            "attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm"
        )
        pinned = specs[0].tuning_id
        again = candidate.select_spec(replace(req, tuning_id=pinned))
        self.assertEqual(again.tuning_id, pinned)
        results = dispatch_attention_all(
            replace(req, tuning_id=pinned),
            candidate_prefix=candidate.name,
        )
        matching = [r for r in results if r.spec.tuning_id == pinned]
        self.assertEqual(len(matching), 1)

    def test_runtime_cache_size_specializes_i64_but_keeps_the_id(self):
        """Addressing width is a runtime specialization: a different binary
        (identity), the same configuration (tuning_id), so an id recorded after
        binding still replays from the request."""
        from rocke.dispatch.core import spec_identity

        _candidate, _req, specs = _specs_for(
            "attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm"
        )
        base = specs[0]
        at_limit = base.with_num_kv_blocks(65536)
        above_limit = base.with_num_kv_blocks(65537)
        self.assertFalse(at_limit.kernel_spec.use_i64_kv_addr)
        self.assertEqual(at_limit.tuning_id, base.tuning_id)
        self.assertTrue(above_limit.kernel_spec.use_i64_kv_addr)
        self.assertEqual(above_limit.tuning_id, base.tuning_id)
        self.assertNotEqual(spec_identity(above_limit), spec_identity(base))
        self.assertEqual(above_limit.num_kv_blocks, 65537)

        _candidate, _req, split_specs = _specs_for(
            "attention_gfx950_u3d_splitkv_seg64_t1xb",
            seqlen_q=1,
            seqlen_k=4096,
        )
        split = split_specs[0]
        split_i64 = split.with_num_kv_blocks(65537)
        self.assertTrue(split_i64.kernel_spec.use_i64_kv_addr)
        self.assertEqual(split_i64.reduce_spec, split.reduce_spec)
        self.assertEqual(split_i64.tuning_id, split.tuning_id)
        self.assertNotEqual(spec_identity(split_i64), spec_identity(split))

        _candidate, _req, gfx942_split_specs = _specs_for(
            "attention_gfx942_u3d_splitkv_seg64_t1xb",
            arch="gfx942",
            seqlen_q=1,
            seqlen_k=4096,
        )
        with self.assertRaisesRegex(NotImplementedError, "does not support"):
            gfx942_split_specs[0].with_num_kv_blocks(65537)

        _candidate, _req, fp8_specs = _specs_for(
            "attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm",
            use_fp8=True,
            fp8_fnuz=False,
        )
        self.assertFalse(
            fp8_specs[0].with_num_kv_blocks(131072).kernel_spec.use_i64_kv_addr
        )
        self.assertTrue(
            fp8_specs[0].with_num_kv_blocks(131073).kernel_spec.use_i64_kv_addr
        )

    def test_tuning_wrapper_preserves_fp8_encoding(self):
        gfx950_candidate, _req, gfx950_specs = _specs_for(
            "attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm",
            use_fp8=True,
            fp8_fnuz=False,
        )
        _gfx942_candidate, _req, gfx942_specs = _specs_for(
            "attention_gfx942_u3d_splitkv_seg64_t1xb",
            arch="gfx942",
            seqlen_q=1,
            seqlen_k=4096,
            use_fp8=True,
            fp8_fnuz=True,
        )
        self.assertTrue(gfx950_specs)
        self.assertTrue(gfx942_specs)
        self.assertFalse(gfx950_specs[0].fp8_fnuz)
        self.assertTrue(gfx942_specs[0].fp8_fnuz)
        self.assertNotIn("fnuz", gfx950_specs[0].kernel_name())
        self.assertIn("fnuz", gfx942_specs[0].kernel_name())
        with self.assertRaisesRegex(ValueError, "requires OCP"):
            gfx950_candidate.built(replace(gfx950_specs[0], fp8_fnuz=True), "gfx950")

    def test_narrow_gfx942_never_offers_k_hbm_direct(self):
        _c, _req, specs = _specs_for(
            "attention_gfx942_u2d_narrow_nw2_mw16_t4xb_llvm", arch="gfx942"
        )
        self.assertTrue(specs)
        self.assertFalse(any(s.kernel_spec.use_k_hbm_direct for s in specs))

    def test_k_hbm_direct_is_rejected_on_narrow_gfx942_specs(self):
        with self.assertRaisesRegex(ValueError, "transposed-x8"):
            Gfx942Spec(
                head_size=128,
                block_size=16,
                num_query_heads=32,
                num_kv_heads=8,
                dtype="fp16",
                use_sinks=False,
                sliding_window=0,
                has_softcap=False,
                num_warps=2,
                block_m_per_warp=16,
                tile_size=64,
                use_k_hbm_direct=True,
            )

    def test_no_khbm_plus_sliced_ring(self):
        with self.assertRaisesRegex(ValueError, "k_sliced_ring"):
            Gfx942Spec(
                head_size=128,
                block_size=16,
                num_query_heads=32,
                num_kv_heads=8,
                dtype="fp16",
                use_sinks=False,
                sliding_window=0,
                has_softcap=False,
                num_warps=2,
                block_m_per_warp=32,
                tile_size=64,
                use_mfma_32x32x8=True,
                use_transposed_qk_32x32=True,
                use_conflict_free_v_store=True,
                use_k_sliced_ring=True,
                ring_depth=2,
                k_slice_hd=32,
                use_k_hbm_direct=True,
            )

    def test_transposed_x8_omits_k_hbm_direct(self):
        _candidate, _req, specs = _specs_for(
            "attention_gfx942_u2d_transposed_x8_nw2_mw32_t4xb_llvm",
            arch="gfx942",
        )
        self.assertTrue(specs)
        self.assertFalse(any(s.kernel_spec.use_k_hbm_direct for s in specs))

    def test_sampling_reaches_independent_knobs(self):
        specs = _sampled("attention_gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm", 256)
        flags = {
            "use_q_reread": False,
            "use_q_direct_reg": False,
            "use_v_double_buffer": False,
            "use_k_single_buffer": False,
            "use_grouped_kv2_softmax": False,
        }
        for spec in specs:
            for name in flags:
                if getattr(spec.kernel_spec, name):
                    flags[name] = True
        self.assertTrue(all(flags.values()), flags)

    def test_execution_candidates_include_every_tuning_geometry(self):
        route = [c for c in attention_candidates() if c.algorithm == "unified_tuning"]
        execution = [
            c
            for c in attention_execution_candidates()
            if c.algorithm == "unified_tuning"
        ]
        from dispatch.attention.gfx942_unified import GFX942_TUNING_VARIANTS
        from dispatch.attention.gfx950_unified import GFX950_TUNING_VARIANTS

        expected = len(GFX942_TUNING_VARIANTS) + len(GFX950_TUNING_VARIANTS)
        self.assertEqual(len(route), expected)
        self.assertEqual(len(execution), expected)
        self.assertTrue(route)
        self.assertTrue(all(c.opt_in for c in route))
        self.assertTrue(all(c.opt_in for c in execution))

    def test_full_sample_includes_dead_end_knobs(self):
        """Dead ends stay out of KNOWN_WRONG_KNOBS and in the sampled sweep."""
        cases = (
            (
                "attention_gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm",
                "gfx950",
                "use_q_reread",
            ),
            (
                "attention_gfx942_u2d_transposed_x8_nw2_mw32_t4xb_llvm",
                "gfx942",
                "use_conflict_free_v",
            ),
        )
        for prefix, arch, knob in cases:
            with self.subTest(knob=knob):
                self.assertNotIn(knob, KNOWN_WRONG_KNOBS[arch])
                specs = _sampled(prefix, 256, seed=0, arch=arch)
                self.assertTrue(any(getattr(s.kernel_spec, knob) for s in specs))

    def test_production_sweep_excludes_dead_end_knobs(self):
        from dispatch.attention.axes import DEAD_END_KNOBS

        cases = (
            (
                "attention_gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm",
                "gfx950",
                "use_q_reread",
            ),
            (
                "attention_gfx942_u2d_transposed_x8_nw2_mw32_t4xb_llvm",
                "gfx942",
                "use_conflict_free_v",
            ),
        )
        for prefix, arch, knob in cases:
            with self.subTest(candidate=prefix):
                _c, _req, specs = _specs_for(prefix, n=10000, arch=arch)
                self.assertGreater(len(specs), 1)
                self.assertLess(len(specs), 500)
                for spec in specs:
                    for dead in DEAD_END_KNOBS[arch]:
                        self.assertFalse(getattr(spec.kernel_spec, dead, False), knob)

    def test_an_over_budget_geometry_still_resolves(self):
        """The wide32 8x baseline passes the kernel validators but its LDS pool
        (165888 B) does not fit; auto takes the single-K spec that does."""
        spec = attention_tuning_spec(_request(), "gfx950_u2d_wide32_nw4_mw32_t8xb_llvm")
        self.assertTrue(spec.kernel_spec.use_k_single_buffer)


def _gfx950_variant(spec_id):
    return next(v for v in GFX950_TUNING_VARIANTS if v.spec_id == spec_id)


_KQ_PAD = {"use_k_single_buffer": True, "use_kq_lds_pad": True, "kq_lds_pad_halves": 16}


class TestGfx950UnifiedLegality(unittest.TestCase):
    """What the tiled 2D validator does not model: the LDS footprint, and the
    padded-K layout the kernel disables or cannot share with Q."""

    def test_lds_model_is_the_lowered_pool(self):
        import re

        import kernels.common.attention_unified as au
        from dispatch.attention.tuning_specs import build_explicit_attention_2d
        from rocke import lower_kernel_to_llvm

        decode = dict(
            nhead_q=16, nhead_k=2, seqlen_q=1, seqlen_k=8192, kv_block_size=64
        )
        cases = (
            ("gfx950_u2d_narrow_nw1_mw16_t4xb_llvm", decode, {}),
            (
                "gfx950_u2d_narrow_nw1_mw16_t4xb_llvm",
                decode,
                {"use_k_single_buffer": True},
            ),
            ("gfx950_u2d_wide32_nw4_mw32_t8xb_llvm", {}, {}),
            ("gfx950_u2d_transposed32_nw4_mw32_t1xb_llvm", decode, {}),
            (
                "gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm",
                dict(mask_type=1, kv_block_size=64),
                {**_KQ_PAD, "use_q_reread": True},
            ),
        )
        au._RESOLVED_ATTENTION_ARCH = "gfx950"  # conftest restores it
        for spec_id, shape, knobs in cases:
            with self.subTest(spec_id=spec_id, knobs=knobs):
                problem = _problem(_request(**shape))
                space = unified_space(_gfx950_variant(spec_id))
                spec = space.build(problem, {**space.fixed(problem), **knobs})
                ir = lower_kernel_to_llvm(
                    build_explicit_attention_2d(spec.kernel_spec, arch="gfx950"),
                    arch="gfx950",
                )
                pool = sum(
                    int(n)
                    for n in re.findall(r"addrspace\(3\) global \[(\d+) x i8\]", ir)
                )
                self.assertEqual(_gfx950_2d_lds_bytes(spec.kernel_spec), pool)

    def test_over_budget_baseline_is_refused_and_auto_fits(self):
        req = _request(
            nhead_q=16, nhead_k=2, seqlen_q=1, seqlen_k=8192, kv_block_size=64
        )
        variant = _gfx950_variant("gfx950_u2d_narrow_nw1_mw16_t4xb_llvm")
        spec, why = canonicalize_tuning_spec(_problem(req), variant, {})
        self.assertIsNone(spec)
        self.assertIn("LDS budget", why)
        auto = attention_tuning_spec(req, variant.spec_id)
        self.assertEqual(dict(auto.knobs), {"use_k_single_buffer": True})
        cap = ArchTarget.from_gfx("gfx950").lds_capacity_bytes
        double_k = replace(auto.kernel_spec, use_k_single_buffer=False)
        # llc: "local memory (209152) exceeds limit (163840)" for the baseline.
        self.assertEqual(_gfx950_2d_lds_bytes(double_k), 209152)
        self.assertLessEqual(_gfx950_2d_lds_bytes(auto.kernel_spec), cap)

    def test_padded_k_with_aliased_q_is_refused(self):
        req = _request(mask_type=1, kv_block_size=64)
        spec_id = "gfx950_u2d_transposed32_nw2_mw32_t2xb_llvm"
        with self.assertRaisesRegex(ValueError, "aliased Q"):
            tuning_spec_with_knobs(req, spec_id, _KQ_PAD)
        reread = tuning_spec_with_knobs(req, spec_id, {**_KQ_PAD, "use_q_reread": True})
        self.assertTrue(reread.kernel_spec.use_kq_lds_pad)

    def test_a_pad_the_kernel_lays_out_as_none_names_the_unpadded_spec(self):
        req = _request(mask_type=1, kv_block_size=64)
        variant = _gfx950_variant("gfx950_u2d_transposed32_nw2_mw32_t2xb_hipcc")
        unpadded, _why = canonicalize_tuning_spec(_problem(req), variant, {})
        for halves in (8, 16, 24, 32):
            with self.subTest(halves=halves):
                padded, why = canonicalize_tuning_spec(
                    _problem(req),
                    variant,
                    {"use_kq_lds_pad": True, "kq_lds_pad_halves": halves},
                )
                self.assertIsNotNone(padded, why)
                self.assertEqual(padded.tuning_id, unpadded.tuning_id)
                self.assertEqual(padded.knobs, ())

    def test_offered_pads_are_laid_out_and_never_alias_q(self):
        shape = dict(mask_type=1, kv_block_size=64)
        for prefix in (
            "attention_gfx950_u2d_transposed32_nw2_mw32_t2xb_llvm",
            "attention_gfx950_u2d_wide32_nw2_mw32_t2xb_llvm",
        ):
            specs = _sampled(prefix, 64, **shape) + _specs_for(prefix, 300, **shape)[2]
            self.assertTrue(specs, prefix)
            for spec in specs:
                ks = spec.kernel_spec
                if ks.use_kq_lds_pad:
                    with self.subTest(spec=spec.tuning_id):
                        self.assertTrue(ks.use_k_single_buffer)
                        self.assertTrue(ks.use_q_reread or ks.use_q_direct_reg)


def _dense_request(**kw):
    base = dict(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=2048,
        seqlen_k=2048,
        hdim_q=128,
        hdim_v=128,
        arch="gfx950",
        dtype="bf16",
        mask_type=1,
    )
    base.update(kw)
    return AttentionRequest(**base)


def _dense_combos(req, level="production", limit=0, **kw):
    """Registered dense ``(candidate, spec)`` pairs for ``req.arch``, through the
    same entry point the combo-sweep bench uses."""
    combos = (
        (candidate, spec)
        for candidate, spec in iter_registered_attention_combos(
            req,
            candidate_prefix=f"attention_{req.arch}_dense",
            sweep_level=level,
            **kw,
        )
        if candidate.algorithm == "attention_dense"
    )
    return tuple(islice(combos, limit) if limit else combos)


def _by_candidate(combos):
    grouped = {}
    for candidate, spec in combos:
        grouped.setdefault(candidate.name, (candidate, []))[1].append(spec)
    return grouped


def _changed_fields(spec, base):
    spec, base = spec.kernel_spec, base.kernel_spec
    return {
        f.name
        for f in dataclasses.fields(spec)
        if getattr(spec, f.name) != getattr(base, f.name)
    }


def _dense_swept_fields(arch="gfx950"):
    return {
        name
        for axis in tuning_axes(arch, "dense")
        for choice in axis.choices
        for name, _value in choice
    }


def _assert_dense_fields_classified(case, arch, spec_type):
    fields = {f.name for f in dataclasses.fields(spec_type)}
    swept = _dense_swept_fields(arch)
    fixed = (
        DENSE_PROBLEM_FIELDS
        | DENSE_VARIANT_FIELDS[arch]
        | DENSE_LOOP_FIELDS
        | DENSE_UNTUNABLE_KNOBS[arch]
    )
    case.assertFalse(fields - fixed - swept, sorted(fields - fixed - swept))
    case.assertFalse((swept | fixed) - fields, sorted((swept | fixed) - fields))
    case.assertFalse(swept & fixed, sorted(swept & fixed))


def _assert_axes_never_restate_defaults(case, arch, spec_type):
    defaults = {f.name: f.default for f in dataclasses.fields(spec_type)}
    for axis in tuning_axes(arch, "dense"):
        case.assertEqual(axis.choices[0], (), axis.name)
        for choice in axis.choices[1:]:
            for name, value in choice:
                if name in DENSE_BASE_RELATIVE_KNOBS:
                    continue  # relative to the base spec, pruned when equal
                with case.subTest(axis=axis.name, knob=name, value=value):
                    case.assertNotEqual(value, defaults[name])


def _production_changed_fields(requests):
    changed = set()
    for req in requests:
        for _candidate, specs in _by_candidate(_dense_combos(req)).values():
            for spec in specs[1:]:
                changed |= _changed_fields(spec, specs[0])
    return changed


class TestGfx950DenseTuningSpace(unittest.TestCase):
    """The gfx950 dense kernel's knobs on the shared ``KnobAxis`` machinery."""

    def test_every_dense_spec_field_is_classified(self):
        _assert_dense_fields_classified(self, "gfx950", Gfx950AttentionDenseSpec)

    def test_untunable_knobs_accept_only_their_default(self):
        defaults = {
            f.name: f.default for f in dataclasses.fields(Gfx950AttentionDenseSpec)
        }
        shape = dict(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=8,
            num_kv_heads=1,
            head_size=128,
        )
        for name in DENSE_UNTUNABLE_KNOBS["gfx950"]:
            with self.subTest(knob=name):
                with self.assertRaises(ValueError):
                    Gfx950AttentionDenseSpec(**shape, **{name: defaults[name] + 1})

    def test_axis_choices_never_restate_a_field_default(self):
        _assert_axes_never_restate_defaults(self, "gfx950", Gfx950AttentionDenseSpec)

    def test_production_varies_every_declared_knob(self):
        changed = _production_changed_fields(
            (_dense_request(), _dense_request(hdim_q=64, hdim_v=64))
        )
        missing = _dense_swept_fields() - changed
        self.assertFalse(missing, sorted(missing))

    def test_production_specs_are_legal_distinct_and_start_at_the_shipped_spec(self):
        req = _dense_request()
        grouped = _by_candidate(_dense_combos(req))
        self.assertEqual(len(grouped), 6, sorted(grouped))
        for name, (candidate, specs) in grouped.items():
            with self.subTest(candidate=name):
                probe = replace(
                    req, algorithm=candidate.algorithm, spec_id=candidate.spec_id
                )
                self.assertEqual(specs[0], candidate.select_spec(probe))
                names = [spec.kernel_name() for spec in specs]
                self.assertEqual(len(names), len(set(names)))
                self.assertEqual(len({s.tuning_id for s in specs}), len(specs))
                for spec in specs:
                    self.assertEqual(spec.path, "dense")
                    ok, why = supports_attention_dense(spec.kernel_spec, arch="gfx950")
                    self.assertTrue(ok, (spec.kernel_name(), why))

    def test_causal_only_knobs_are_not_offered_without_causal(self):
        changed = set()
        req = _dense_request(mask_type=0)
        for _candidate, specs in _by_candidate(_dense_combos(req)).values():
            for spec in specs[1:]:
                changed |= _changed_fields(spec, specs[0])
        self.assertTrue(changed)
        self.assertNotIn("causal_diag_split", changed)
        self.assertNotIn("interleave", changed)

    def test_num_persistent_policies_reach_every_persistent_decode(self):
        grouped = _by_candidate(_dense_combos(_dense_request()))
        _candidate, specs = grouped["attention_gfx950_dense_persist_default"]
        kernels = [spec.kernel_spec for spec in specs]
        self.assertEqual(
            {k.resolved_persist_decode for k in kernels},
            {"qb_major", "hkv_major", "gqa_pair", "gqa_pair_2phase"},
        )
        swept_counts = {k.num_persistent for k in kernels}
        for policy in ("gqa_pair", "gqa_pair_2phase", "0.75x", "1.25x", "2x"):
            self.assertIn(
                resolve_dense_num_persistent(kernels[0], policy), swept_counts
            )

    def test_redundant_settings_are_pruned(self):
        """A setting that compiles to the default is dropped by
        canonicalization, so it can never mint a second id."""
        req, spec_id = _dense_request(), "gfx950_dense_grid_default"
        default = attention_tuning_spec(req, spec_id)
        base = default.kernel_spec

        def canonical(**knobs):
            return dict(tuning_spec_with_knobs(req, spec_id, knobs).knobs)

        restated = (
            dict(pv_loop_order=base.resolved_pv_loop_order()),
            dict(exp_per_pv_step=base.resolved_exp_per_pv_step()),
            dict(iglp_mode=base.resolved_iglp_mode()),
            dict(interleave=True),  # the grid body never reads it
            dict(persist_decode="hkv_major"),
        )
        for knobs in restated:
            with self.subTest(**knobs):
                self.assertEqual(canonical(**knobs), {})
        inert_partner = (
            (
                dict(lazy_rescale=False, lazy_rescale_threshold=4.0),
                "lazy_rescale_threshold",
            ),
            (
                dict(pv_sched_fence=False, pv_sched_fence_mask=0x008),
                "pv_sched_fence_mask",
            ),
            (
                dict(pv_sched_group_template=False, pv_sched_group_ds_read=4),
                "pv_sched_group_ds_read",
            ),
        )
        for knobs, inert in inert_partner:
            with self.subTest(**knobs):
                self.assertNotIn(inert, canonical(**knobs))
        self.assertEqual(
            canonical(pv_loop_order="k_major"), {"pv_loop_order": "k_major"}
        )
        self.assertEqual(canonical(lazy_rescale=False), {"lazy_rescale": False})

    def test_full_stream_and_samples_are_distinct_and_reproducible(self):
        req = _dense_request()
        streamed = _dense_combos(req, level="full", limit=300)
        names = [spec.kernel_name() for _candidate, spec in streamed]
        self.assertEqual(len(names), len(set(names)))

        def sampled(seed):
            return [
                (c.name, s.kernel_name())
                for c, s in _dense_combos(
                    req, level="full", tuning_sample=16, seed=seed
                )
            ]

        first = sampled(3)
        self.assertEqual(first, sampled(3))
        self.assertNotEqual(first, sampled(4))
        self.assertEqual(len(first), len(set(first)))
        for name, (_c, specs) in _by_candidate(
            _dense_combos(req, level="full", tuning_sample=16, seed=3)
        ).items():
            with self.subTest(candidate=name):
                self.assertEqual(len(specs), 16)

    def test_tuning_ids_replay_through_dispatch(self):
        """A dense config is served by pinning ``spec_id`` + tuning id, like a
        unified tuning geometry; ``auto`` is the variant's default spec."""
        req = _dense_request()
        spec_id = "gfx950_dense_persist_widedma_bm128"
        grouped = _by_candidate(_dense_combos(req))
        _c, specs = grouped["attention_gfx950_dense_persist_widedma_bm128"]
        self.assertEqual(attention_tuning_spec(req, spec_id), specs[0])
        for spec in specs[:: max(1, len(specs) // 8)]:
            with self.subTest(tuning_id=spec.tuning_id):
                self.assertEqual(
                    attention_tuning_spec(req, spec_id, spec.tuning_id), spec
                )
        wpe = {s.kernel_spec.waves_per_eu for s in specs}
        self.assertGreater(len(wpe), 1)  # WPE is a knob, not a request field

    def test_unpinned_requests_never_select_a_dense_variant(self):
        for algorithm in ("auto", "attention_dense"):
            with self.subTest(algorithm=algorithm):
                req = _dense_request(algorithm=algorithm)
                if algorithm == "auto":
                    self.assertEqual(
                        dispatch_attention(req).candidate.algorithm, "unified_2d"
                    )
                else:
                    with self.assertRaises(ValueError):
                        dispatch_attention(req)


def _gfx942_request(**kw):
    base = dict(arch="gfx942", dtype="fp16")
    base.update(kw)
    return _dense_request(**base)


def _gfx942_specs(req, level="production", limit=0, **kw):
    combos = _dense_combos(req, level=level, limit=limit, **kw)
    return [spec for _candidate, spec in combos]


def gfx942_dense_spec(req):
    return attention_tuning_spec(req, "gfx942_dense").kernel_spec


class TestGfx942DenseTuningSpace(unittest.TestCase):
    """The gfx942 dense kernel's knobs on the same machinery. gfx942 registers
    one dense candidate, so its geometry is swept rather than fixed by a variant."""

    def test_every_dense_spec_field_is_classified(self):
        _assert_dense_fields_classified(self, "gfx942", Gfx942AttentionDenseSpec)

    def test_axis_choices_never_restate_a_field_default(self):
        _assert_axes_never_restate_defaults(self, "gfx942", Gfx942AttentionDenseSpec)

    def test_untunable_knobs_have_nothing_to_sweep(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from kernels.gfx942.attention_dense import build_attention_dense

        base = gfx942_dense_spec(_gfx942_request())
        ok, why = supports_gfx942(replace(base, lds_num_buffers=2), arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("lds_num_buffers", why)

        def ir(spec):
            kernel = build_attention_dense(spec, arch="gfx942")
            return lower_kernel_to_llvm(kernel, arch="gfx942").replace(
                kernel.name, "@k"
            )

        flipped = replace(base, lazy_rescale=not base.lazy_rescale)
        self.assertEqual(ir(flipped), ir(base), "lazy_rescale reached gfx942 IR")

    def test_production_varies_every_declared_knob(self):
        changed = _production_changed_fields(
            (
                _gfx942_request(),
                _gfx942_request(seqlen_q=8192, seqlen_k=8192),
                _gfx942_request(
                    dtype="bf16", hdim_q=64, hdim_v=64, nhead_q=8, nhead_k=1
                ),
            )
        )
        missing = _dense_swept_fields("gfx942") - changed
        self.assertFalse(missing, sorted(missing))

    def test_production_specs_are_legal_distinct_and_start_at_the_shipped_spec(self):
        for req in (_gfx942_request(), _gfx942_request(seqlen_q=8192, seqlen_k=8192)):
            specs = _gfx942_specs(req)
            with self.subTest(seqlen=req.seqlen_q):
                self.assertEqual(specs[0].kernel_spec, gfx942_dense_spec(req))
                names = [spec.kernel_name() for spec in specs]
                self.assertEqual(len(names), len(set(names)))
                kernels = [s.kernel_spec for s in specs]
                for spec in kernels:
                    ok, why = supports_gfx942(spec, arch="gfx942")
                    self.assertTrue(ok, (spec.kernel_name(), why))
                self.assertEqual({k.persistent for k in kernels}, {True, False})
                self.assertGreater(len({k.block_m for k in kernels}), 1)

    def test_tuning_ids_replay_through_dispatch(self):
        """Geometry, persistence and decode are knobs of the one candidate,
        reached through its tuning ids rather than request fields."""
        req = _gfx942_request(seqlen_q=8192, seqlen_k=8192)
        specs = _gfx942_specs(req)
        self.assertEqual(attention_tuning_spec(req, "gfx942_dense"), specs[0])
        for spec in specs[:: max(1, len(specs) // 8)]:
            with self.subTest(tuning_id=spec.tuning_id):
                self.assertEqual(
                    attention_tuning_spec(req, "gfx942_dense", spec.tuning_id), spec
                )
        decodes = {s.kernel_spec.persist_decode for s in specs}
        self.assertIn("hkv_major", decodes)

    def test_redundant_settings_are_pruned(self):
        base = gfx942_dense_spec(_gfx942_request())
        d64 = gfx942_dense_spec(
            _gfx942_request(dtype="bf16", hdim_q=64, hdim_v=64, nhead_q=8, nhead_k=1)
        )

        req = _gfx942_request()
        d64_req = _gfx942_request(
            dtype="bf16", hdim_q=64, hdim_v=64, nhead_q=8, nhead_k=1
        )

        def canonical(knobs, r=req):
            return tuning_spec_with_knobs(r, "gfx942_dense", knobs)

        redundant = (
            dict(use_cfvst=True),  # the policy already turns it on here
            dict(use_exp2_fast=base.resolved_use_exp2_fast()),
            dict(persistent=base.persistent),  # base-relative no-op
            dict(num_persistent=base.num_persistent * 2),  # grid body ignores it
            dict(persist_decode="hkv_major"),
            dict(interleave=True),
        )
        default = attention_tuning_spec(req, "gfx942_dense")
        for knobs in redundant:
            with self.subTest(**knobs):
                self.assertEqual(canonical(knobs), default)
        # No cfvst path at D64: the V knobs are out of scope, so dropped, not refused.
        d64_default = attention_tuning_spec(d64_req, "gfx942_dense")
        self.assertEqual(d64_default.kernel_spec, d64)
        for knobs in (dict(v_row_pad=16), dict(use_v_swizzle=False)):
            with self.subTest(d64=knobs):
                self.assertEqual(canonical(knobs, d64_req), d64_default)
        self.assertNotEqual(canonical(dict(use_cfvst=False)), default)
        self.assertNotEqual(canonical(dict(pv_sched_fence_mask=0)), default)
        self.assertNotEqual(canonical(dict(persistent=not base.persistent)), default)

    def test_full_stream_and_samples_are_distinct_and_reproducible(self):
        req = _gfx942_request()
        streamed = _gfx942_specs(req, level="full", limit=300)
        self.assertEqual(
            len({s.kernel_name() for s in streamed}), len(streamed), "duplicates"
        )

        def sampled(seed):
            return [
                s.kernel_name()
                for s in _gfx942_specs(req, level="full", tuning_sample=16, seed=seed)
            ]

        first = sampled(3)
        self.assertEqual(len(first), 16)
        self.assertEqual(first, sampled(3))
        self.assertNotEqual(first, sampled(4))
        self.assertEqual(len(first), len(set(first)))


class TestAutoDispatchUnchanged(unittest.TestCase):
    def test_auto_still_selects_unified_3d_for_decode(self):
        import kernels.common.attention_unified as au

        old = au._RESOLVED_ATTENTION_ARCH
        try:
            au._RESOLVED_ATTENTION_ARCH = "gfx950"
            result = dispatch_attention(_request(seqlen_q=1, seqlen_k=4096))
        finally:
            au._RESOLVED_ATTENTION_ARCH = old
        self.assertEqual(result.candidate.name, "attention_unified_3d")


if __name__ == "__main__":
    unittest.main()
