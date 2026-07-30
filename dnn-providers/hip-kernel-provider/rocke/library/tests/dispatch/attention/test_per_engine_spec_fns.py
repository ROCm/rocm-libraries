# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Byte-identity + non-interference tests for the per-engine spec builders.

Each per-engine ``spec_fn`` (GEMM-style) is extracted verbatim from a branch of
``_tiled_spec_from_problem``. These CPU-only tests prove every extraction is a
PURE MOVE and that new extractions don't disturb the others.

EXTENDING THIS FILE: to cover a newly-migrated cohort, add ONE ``_Cohort`` entry
to ``_COHORTS`` (spec_fn + eligibility gate + a problem factory + an independent
reference reconstruction of the pre-refactor branch body). No new test file, no
new test methods -- the table drives them all.

Each entry supplies:
  - ``name``      : label for subTest
  - ``arch``      : arch to pin (``_RESOLVED_ATTENTION_ARCH``)
  - ``spec_fn``   : the extracted builder under test
  - ``gate``      : the cohort eligibility predicate
  - ``problems``  : list of factories producing eligible problems (cover
                    sub-branches, e.g. hd64 vs hd128)
  - ``reference`` : independent reconstruction of the pre-refactor branch body
                    (a genuine second copy -- this is what makes the equality
                    check meaningful rather than circular)
  - ``foreign``   : a factory producing a problem the gate REJECTS (for the
                    non-interference check) + the field/value that proves it took
                    a different branch
"""

from __future__ import annotations

import unittest
from dataclasses import asdict

import kernels.common.attention_unified as au
from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    _enable_gfx942_bf16_flash,
    _enable_gfx942_flash_k_sliced_ldsseq,
    _enable_gfx942_flash_k_sliced_ring,
    _enable_gfx942_flash_mask_limit,
    _enable_gfx942_flash_q_direct,
    _enable_gfx942_fp16_flash,
    _enable_i64_kv_addr,
    _gfx942_bf16_wide_geometry,
    _gfx942_bf16_wide_tile_size,
    _gfx942_bf16_wide_use_cfvst,
    _gfx942_flash_kv_cache_policy,
    _gfx942_flash_use_cfvst,
    _gfx942_flash_use_single_buffer,
    _gfx942_flash_wide_setting,
    _kv_storage_dtype,
    _select_2d_block_m_per_warp,
    _select_2d_tile_size,
    _select_2d_waves_per_eu,
    _select_gfx942_flash_num_warps,
    _select_gfx942_flash_ring_depth,
    _select_gfx942_flash_k_slice_hd,
    _tiled_2d_impl,
)
import builders.common.attention_spec_builder as bld


class _PinnedArch:
    def __init__(self, arch: str):
        self._arch = arch

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = self._arch
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


def _problem(**kw) -> UnifiedAttentionProblem:
    base = dict(
        total_q=2048,
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=2048,
        max_seqlen_k=2048,
        dtype="fp16",
    )
    base.update(kw)
    if "total_q" not in kw:
        base["total_q"] = base["num_seqs"] * base["max_seqlen_q"]
    return UnifiedAttentionProblem(**base)


# --------------------------------------------------------------------------
# gfx942 fp16 flash cohort
# --------------------------------------------------------------------------
def _reference_gfx942_fp16_flash(problem):
    """Independent reconstruction of the pre-refactor fp16-flash branch body."""
    UnifiedAttention2DTiledSpec, _, _ = _tiled_2d_impl("gfx942")
    num_warps = _select_gfx942_flash_num_warps(problem)
    use_cfvst = _gfx942_flash_use_cfvst(problem)
    use_single = _gfx942_flash_use_single_buffer(problem)
    use_mask_limit = _enable_gfx942_flash_mask_limit(problem)
    return UnifiedAttention2DTiledSpec(
        head_size=problem.head_size,
        block_size=problem.block_size,
        num_query_heads=problem.num_query_heads,
        num_kv_heads=problem.num_kv_heads,
        dtype=problem.dtype,
        use_sinks=problem.use_sinks,
        sliding_window=problem.sliding_window,
        has_softcap=problem.softcap > 0,
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        num_seqs=problem.num_seqs,
        num_warps=num_warps,
        waves_per_eu=_select_2d_waves_per_eu(problem),
        kv_storage_dtype=_kv_storage_dtype(problem),
        tile_size=_select_2d_tile_size(problem),
        block_m_per_warp=_select_2d_block_m_per_warp(problem),
        use_mfma_32x32x8=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=use_mask_limit,
        use_transposed_invariant_hoist=use_mask_limit,
        use_transposed_mask_once=use_mask_limit,
        use_transposed_mask_limit=use_mask_limit,
        use_conflict_free_v_store=use_cfvst,
        use_k_single_buffer=use_single,
        use_k_sliced_ring=_enable_gfx942_flash_k_sliced_ring(problem),
        ring_depth=_select_gfx942_flash_ring_depth(problem),
        k_slice_hd=_select_gfx942_flash_k_slice_hd(problem),
        use_k_sliced_ldsseq=_enable_gfx942_flash_k_sliced_ldsseq(problem),
        use_q_direct_global=_enable_gfx942_flash_q_direct(problem),
        kv_cache_policy=_gfx942_flash_kv_cache_policy(problem),
        use_i64_kv_addr=_enable_i64_kv_addr(problem),
    )


# --------------------------------------------------------------------------
# gfx942 bf16 flash cohort
# --------------------------------------------------------------------------
def _reference_gfx942_bf16_flash(problem):
    """Independent reconstruction of the pre-refactor bf16-flash branch body."""
    UnifiedAttention2DTiledSpec, _, _ = _tiled_2d_impl("gfx942")
    use_ring = _enable_gfx942_flash_k_sliced_ring(problem)
    if use_ring:
        nw = _gfx942_flash_wide_setting()
        single_k = False
        use_cfvst = True
    else:
        nw, single_k = _gfx942_bf16_wide_geometry(problem)
        use_cfvst = _gfx942_bf16_wide_use_cfvst(problem)
    use_mask_limit = _enable_gfx942_flash_mask_limit(problem)
    return UnifiedAttention2DTiledSpec(
        head_size=problem.head_size,
        block_size=problem.block_size,
        num_query_heads=problem.num_query_heads,
        num_kv_heads=problem.num_kv_heads,
        dtype=problem.dtype,
        use_sinks=problem.use_sinks,
        sliding_window=problem.sliding_window,
        has_softcap=problem.softcap > 0,
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        num_seqs=problem.num_seqs,
        num_warps=nw,
        waves_per_eu=_select_2d_waves_per_eu(problem),
        kv_storage_dtype=_kv_storage_dtype(problem),
        tile_size=64 if use_ring else _gfx942_bf16_wide_tile_size(problem),
        block_m_per_warp=32,
        use_mfma_32x32x8=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=use_mask_limit,
        use_transposed_invariant_hoist=use_mask_limit,
        use_transposed_mask_once=use_mask_limit,
        use_transposed_mask_limit=use_mask_limit,
        use_conflict_free_v_store=use_cfvst,
        use_k_single_buffer=single_k,
        use_k_sliced_ring=use_ring,
        ring_depth=_select_gfx942_flash_ring_depth(problem),
        k_slice_hd=_select_gfx942_flash_k_slice_hd(problem),
        use_k_sliced_ldsseq=_enable_gfx942_flash_k_sliced_ldsseq(problem),
        use_q_direct_global=_enable_gfx942_flash_q_direct(problem),
        kv_cache_policy=_gfx942_flash_kv_cache_policy(problem),
        use_i64_kv_addr=_enable_i64_kv_addr(problem),
    )


class _Cohort:
    def __init__(
        self,
        name,
        arch,
        spec_fn,
        gate,
        problems,
        reference,
        foreign,
        foreign_field,
        foreign_value,
    ):
        self.name = name
        self.arch = arch
        self.spec_fn = spec_fn
        self.gate = gate
        self.problems = problems
        self.reference = reference
        self.foreign = foreign
        self.foreign_field = foreign_field
        self.foreign_value = foreign_value


_COHORTS = [
    _Cohort(
        name="gfx942_fp16_flash",
        arch="gfx942",
        spec_fn=lambda p: bld._spec_gfx942_fp16_flash(p),
        gate=_enable_gfx942_fp16_flash,
        # MHA fp16 (the dense_pipe cohort): long- and short-context.
        problems=[
            lambda: _problem(num_query_heads=16, num_kv_heads=16),
            lambda: _problem(
                num_query_heads=16, num_kv_heads=16, max_seqlen_q=512, total_q=512
            ),
        ],
        reference=_reference_gfx942_fp16_flash,
        # bf16 disqualifies the fp16-flash gate -> different branch.
        foreign=lambda: _problem(dtype="bf16"),
        foreign_field="dtype",
        foreign_value="bf16",
    ),
    _Cohort(
        name="gfx942_bf16_flash",
        arch="gfx942",
        spec_fn=lambda p: bld._spec_gfx942_bf16_flash(p),
        gate=_enable_gfx942_bf16_flash,
        # GQA bf16: hd128 (no-ring) and hd64 (ring) exercise both sub-branches.
        problems=[
            lambda: _problem(dtype="bf16"),
            lambda: _problem(dtype="bf16", head_size=64),
        ],
        reference=_reference_gfx942_bf16_flash,
        # fp16 disqualifies the bf16-flash gate -> different branch.
        foreign=lambda: _problem(dtype="fp16"),
        foreign_field="dtype",
        foreign_value="fp16",
    ),
]


class TestPerEngineSpecFns(unittest.TestCase):
    def test_cohorts_eligible(self):
        for c in _COHORTS:
            with self.subTest(cohort=c.name), _PinnedArch(c.arch):
                for mk in c.problems:
                    self.assertTrue(c.gate(mk()), f"{c.name}: problem not eligible")

    def test_spec_fn_matches_reference(self):
        # Pure move: the extracted spec_fn equals an independent reconstruction.
        for c in _COHORTS:
            with self.subTest(cohort=c.name), _PinnedArch(c.arch):
                for mk in c.problems:
                    p = mk()
                    self.assertEqual(asdict(c.spec_fn(p)), asdict(c.reference(p)))

    def test_pipeline_delegates_to_spec_fn(self):
        # The full builder returns exactly the spec_fn's spec for the cohort.
        for c in _COHORTS:
            with self.subTest(cohort=c.name), _PinnedArch(c.arch):
                for mk in c.problems:
                    p = mk()
                    self.assertEqual(
                        asdict(bld._tiled_spec_from_problem(p)),
                        asdict(c.spec_fn(p)),
                    )

    def test_non_interference(self):
        # A problem the gate rejects must NOT receive this cohort's spec -- it
        # flows to a different branch (proven via a distinguishing field).
        for c in _COHORTS:
            with self.subTest(cohort=c.name), _PinnedArch(c.arch):
                p = c.foreign()
                self.assertFalse(c.gate(p))
                spec = bld._tiled_spec_from_problem(p)
                self.assertEqual(getattr(spec, c.foreign_field), c.foreign_value)


if __name__ == "__main__":
    unittest.main()
