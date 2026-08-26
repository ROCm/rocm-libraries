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
import kernels.common.attention_unified as _kau
from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    _enable_combo_2d,
    _enable_early_v_schedule,
    _enable_fp8_mfma_qk,
    _enable_gfx942_3d_invariant_hoist,
    _enable_gfx942_3d_wide_kv_load,
    _enable_gfx942_bf16_flash,
    _enable_gfx942_flash_k_sliced_ldsseq,
    _enable_gfx942_flash_k_sliced_ring,
    _enable_gfx942_flash_mask_limit,
    _enable_gfx942_flash_q_direct,
    _enable_gfx942_fp16_flash,
    _enable_i64_kv_addr,
    _enable_k_single_buffer,
    _enable_mfma_32x32,
    _enable_register_pv,
    _enable_sched_barrier,
    _enable_softmax_mfma_interleave,
    _enable_transposed_half_local_pv,
    _enable_transposed_qk_32x32,
    _enable_transposed_subflags,
    _enable_v_double_buffer,
    _gfx942_bf16_wide_geometry,
    _gfx942_bf16_wide_tile_size,
    _gfx942_bf16_wide_use_cfvst,
    _gfx942_flash_kv_cache_policy,
    _gfx942_flash_use_cfvst,
    _gfx942_flash_use_single_buffer,
    _gfx942_flash_wide_setting,
    _gfx942_3d_tile_size_override,
    _kv_storage_dtype,
    _num_segments,
    _select_2d_block_m_per_warp,
    _select_2d_num_warps,
    _select_2d_tile_size,
    _select_2d_waves_per_eu,
    _select_3d_waves_per_eu,
    _select_gfx942_flash_num_warps,
    _select_gfx942_flash_ring_depth,
    _select_gfx942_flash_k_slice_hd,
    _tiled_2d_impl,
    _tiled_3d_impl,
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


# --------------------------------------------------------------------------
# gfx942 generic (non-flash fallthrough) cohort
# --------------------------------------------------------------------------
def _reference_gfx942_generic(problem):
    """Independent reconstruction of the pre-refactor gfx942 fallthrough.

    The gfx942 residual: the shared fallthrough with the gfx950-only schedule
    fields omitted (the gfx942 spec class does not declare them). The combo /
    transposed helpers hard-gate to gfx950, so on gfx942 they are off.
    """
    UnifiedAttention2DTiledSpec, _, _ = _tiled_2d_impl("gfx942")
    combo = _enable_combo_2d(problem)
    combo_no_sw = combo and problem.sliding_window == 0
    subflags = _enable_transposed_subflags(problem)
    scalar_state = combo or subflags
    skip_legacy_qreg = combo or subflags
    _bias_active = problem.softcap > 0 or problem.use_alibi or problem.use_qq_bias
    mask_opts = (combo_no_sw and not _bias_active) or subflags
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
        num_warps=_select_2d_num_warps(problem),
        waves_per_eu=_select_2d_waves_per_eu(problem),
        kv_storage_dtype=_kv_storage_dtype(problem),
        tile_size=_select_2d_tile_size(problem),
        block_m_per_warp=_select_2d_block_m_per_warp(problem),
        use_mfma_32x32=_enable_mfma_32x32(problem),
        use_transposed_qk_32x32=_enable_transposed_qk_32x32(problem),
        use_transposed_half_local_pv=_enable_transposed_half_local_pv(problem),
        use_transposed_scalar_state=scalar_state,
        use_transposed_mask_once=mask_opts,
        use_transposed_mask_limit=mask_opts,
        use_mfma32_skip_legacy_qreg=skip_legacy_qreg,
        use_early_v_schedule=_enable_early_v_schedule(problem),
        use_fast_paged_kv_desc=(
            combo_no_sw
            and not problem.use_fp8
            and problem.num_query_heads == 64
            and problem.num_kv_heads == 8
            and _select_2d_tile_size(problem) == 64
        ),
        use_register_pv=_enable_register_pv(problem),
        use_fp8_mfma_qk=_enable_fp8_mfma_qk(problem),
        use_i64_kv_addr=_enable_i64_kv_addr(problem),
    )


# --------------------------------------------------------------------------
# gfx950 generic (combo / schedule + D256 override) cohort
# --------------------------------------------------------------------------
def _reference_gfx950_generic(problem):
    """Independent reconstruction of the pre-refactor gfx950 fallthrough.

    The full machinery: combo / subflags, the schedule fields set directly (the
    gfx950 spec class always declares them), and the D256 gfx950 fast-route tail
    override. Mirrors the guarded fallthrough for gfx950 byte-for-byte.
    """
    from dataclasses import replace

    UnifiedAttention2DTiledSpec, _, _ = _tiled_2d_impl("gfx950")
    combo = _enable_combo_2d(problem)
    combo_no_sw = combo and problem.sliding_window == 0
    subflags = _enable_transposed_subflags(problem)
    scalar_state = combo or subflags
    skip_legacy_qreg = combo or subflags
    _bias_active = problem.softcap > 0 or problem.use_alibi or problem.use_qq_bias
    mask_opts = (combo_no_sw and not _bias_active) or subflags
    sched = {
        "use_v_double_buffer": _enable_v_double_buffer(problem),
        "use_sched_barrier": _enable_sched_barrier(problem),
    }
    if _enable_softmax_mfma_interleave(problem):
        sched["use_softmax_mfma_interleave"] = True
        sched["softmax_interleave_mode"] = 1
    if _enable_k_single_buffer(problem):
        sched["use_k_single_buffer"] = True
    spec = UnifiedAttention2DTiledSpec(
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
        num_warps=_select_2d_num_warps(problem),
        waves_per_eu=_select_2d_waves_per_eu(problem),
        kv_storage_dtype=_kv_storage_dtype(problem),
        tile_size=_select_2d_tile_size(problem),
        block_m_per_warp=_select_2d_block_m_per_warp(problem),
        use_mfma_32x32=_enable_mfma_32x32(problem),
        use_transposed_qk_32x32=_enable_transposed_qk_32x32(problem),
        use_transposed_half_local_pv=_enable_transposed_half_local_pv(problem),
        use_transposed_scalar_state=scalar_state,
        use_transposed_mask_once=mask_opts,
        use_transposed_mask_limit=mask_opts,
        use_mfma32_skip_legacy_qreg=skip_legacy_qreg,
        use_early_v_schedule=_enable_early_v_schedule(problem),
        use_fast_paged_kv_desc=(
            combo_no_sw
            and not problem.use_fp8
            and problem.num_query_heads == 64
            and problem.num_kv_heads == 8
            and _select_2d_tile_size(problem) == 64
        ),
        use_register_pv=_enable_register_pv(problem),
        use_fp8_mfma_qk=_enable_fp8_mfma_qk(problem),
        use_i64_kv_addr=_enable_i64_kv_addr(problem),
        **sched,
    )
    if _kau._d256_gfx950_fast(problem):
        spec = replace(spec, **_kau._d256_gfx950_spec_overrides())
    return spec


class _Cohort:
    def __init__(
        self,
        name,
        arch,
        spec_fn,
        gate,
        problems,
        reference,
        foreign=None,
        foreign_field=None,
        foreign_value=None,
    ):
        self.name = name
        self.arch = arch
        self.spec_fn = spec_fn
        self.gate = gate
        self.problems = problems
        self.reference = reference
        # ``foreign`` is optional: a same-arch problem the gate REJECTS, proving
        # non-interference. Some cohorts (e.g. gfx950 generic) have no same-arch
        # escape -- every problem on that arch routes to them -- so they set
        # ``foreign=None`` and the non-interference subtest is skipped for them
        # (cross-arch separation is covered by the other cohorts' foreigns).
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
        # A narrow bf16 problem fails the fp16-flash gate and falls through to
        # the generic cascade, which -- unlike either flash builder -- leaves the
        # 32x32 MFMA flag OFF. dtype is a useless discriminator (every branch
        # copies problem.dtype); use_mfma_32x32x8 actually differs, so a
        # mis-route back into this cohort would flip it True and fail the assert.
        foreign=lambda: _problem(
            dtype="bf16",
            num_query_heads=4,
            num_kv_heads=4,
            max_seqlen_q=16,
            total_q=16,
        ),
        foreign_field="use_mfma_32x32x8",
        foreign_value=False,
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
        # A narrow bf16 problem (the small_q_narrow exclusion) fails the
        # bf16-flash gate and falls through to the generic cascade, which leaves
        # the 32x32 MFMA flag OFF. Both flash builders hardcode it True, so this
        # field -- unlike dtype -- catches a mis-route back into the cohort.
        foreign=lambda: _problem(
            dtype="bf16",
            num_query_heads=4,
            num_kv_heads=4,
            max_seqlen_q=16,
            total_q=16,
        ),
        foreign_field="use_mfma_32x32x8",
        foreign_value=False,
    ),
    _Cohort(
        name="gfx942_generic",
        arch="gfx942",
        spec_fn=lambda p: bld._spec_generic_2d_non_gfx950(p),
        # Generic = the fallthrough: reached when neither gfx942 flash gate fires.
        gate=lambda p: not _enable_gfx942_bf16_flash(p)
        and not _enable_gfx942_fp16_flash(p),
        # GQA short-context (small_q_narrow carve-out) falls through on both dtypes.
        problems=[
            lambda: _problem(
                num_query_heads=32, num_kv_heads=8, max_seqlen_q=256, total_q=256
            ),
            lambda: _problem(
                num_query_heads=32,
                num_kv_heads=8,
                max_seqlen_q=256,
                total_q=256,
                dtype="bf16",
            ),
        ],
        reference=_reference_gfx942_generic,
        # A flash-eligible shape (MHA long fp16) escapes to the fp16 flash branch,
        # distinguished by use_mfma_32x32x8 (True on flash, False on generic).
        foreign=lambda: _problem(
            num_query_heads=16, num_kv_heads=16, max_seqlen_q=2048, total_q=2048
        ),
        foreign_field="use_mfma_32x32x8",
        foreign_value=True,
    ),
    _Cohort(
        name="gfx950_generic",
        arch="gfx950",
        spec_fn=lambda p: bld._spec_gfx950_generic(p),
        # Every gfx950 2D problem is generic (no gfx950 flash branches; combo /
        # d256 all live inside this spec_fn), so the gate is unconditionally true.
        gate=lambda p: True,
        # Cover the sub-paths: plain narrow, combo (64x8 d128), D256, and SW.
        problems=[
            lambda: _problem(num_query_heads=16, num_kv_heads=16, dtype="fp16"),
            lambda: _problem(num_query_heads=64, num_kv_heads=8, dtype="bf16"),
            lambda: _problem(
                num_query_heads=64,
                num_kv_heads=8,
                head_size=256,
                dtype="bf16",
                max_seqlen_q=1024,
                total_q=1024,
            ),
            lambda: _problem(
                num_query_heads=64, num_kv_heads=8, dtype="bf16", sliding_window=128
            ),
        ],
        reference=_reference_gfx950_generic,
        # No same-arch foreign: all gfx950 problems route here (see _Cohort doc).
        foreign=None,
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
            if c.foreign is None:
                continue
            with self.subTest(cohort=c.name), _PinnedArch(c.arch):
                p = c.foreign()
                self.assertFalse(c.gate(p))
                spec = bld._tiled_spec_from_problem(p)
                self.assertEqual(getattr(spec, c.foreign_field), c.foreign_value)


def _reference_generic_3d(problem):
    """Independent reconstruction of the pre-refactor generic 3D fallthrough.

    gfx942 and gfx950 share one 3D path (the ``_gfx942_3d_*`` helpers self-gate),
    so a single reference covers both arches.
    """
    UnifiedAttention3DTiledSpec, *_ = _tiled_3d_impl(au._RESOLVED_ATTENTION_ARCH)
    return UnifiedAttention3DTiledSpec(
        head_size=problem.head_size,
        block_size=problem.block_size,
        num_query_heads=problem.num_query_heads,
        num_kv_heads=problem.num_kv_heads,
        dtype=problem.dtype,
        use_sinks=problem.use_sinks,
        sliding_window=problem.sliding_window,
        has_softcap=problem.softcap > 0,
        num_segments=_num_segments(problem),
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        num_seqs=problem.num_seqs,
        waves_per_eu=_select_3d_waves_per_eu(problem),
        kv_storage_dtype=_kv_storage_dtype(problem),
        tile_size_override=_gfx942_3d_tile_size_override(problem),
        use_invariant_hoist=_enable_gfx942_3d_invariant_hoist(problem),
        use_wide_kv_load=_enable_gfx942_3d_wide_kv_load(problem),
        use_i64_kv_addr=_enable_i64_kv_addr(problem),
    )


class TestGeneric3dSpecFn(unittest.TestCase):
    """Byte-identity + delegation for the extracted generic 3D spec builder.

    Separate from the 2D ``_COHORTS`` table because the 3D cascade uses a
    different builder (``_tiled_3d_spec_from_problem``) and dataclass.
    """

    @staticmethod
    def _decode_problem(**kw) -> UnifiedAttentionProblem:
        base = dict(
            total_q=1,
            num_seqs=1,
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            max_seqlen_q=1,
            max_seqlen_k=8192,
            dtype="bf16",
        )
        base.update(kw)
        return UnifiedAttentionProblem(**base)

    _PROBLEMS = [
        {},
        {"head_size": 256},
        {"num_query_heads": 32, "num_kv_heads": 8, "dtype": "fp16"},
    ]

    def _run(self, arch):
        with _PinnedArch(arch):
            for kw in self._PROBLEMS:
                p = self._decode_problem(**kw)
                self.assertEqual(
                    asdict(bld._spec_generic_3d(p)),
                    asdict(_reference_generic_3d(p)),
                    f"{arch} {kw}: spec_fn != reference",
                )
                self.assertEqual(
                    asdict(bld._tiled_3d_spec_from_problem(p)),
                    asdict(bld._spec_generic_3d(p)),
                    f"{arch} {kw}: pipeline does not delegate to spec_fn",
                )

    def test_gfx942(self):
        self._run("gfx942")

    def test_gfx950(self):
        self._run("gfx950")


if __name__ == "__main__":
    unittest.main()
