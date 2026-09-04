# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only tests for the gfx942 GQA head-fold in the 4-warp SWA attention kernel.

The head-fold packs the ``num_queries_per_kv`` query heads that share a KV head
into one workgroup's 128-row M-tile so paged K/V is read once per KV head. It
changes device-side index math (grid.x, the 32-token block, and the
``row m -> (token m//4, head m%4)`` map) but is not covered by the representative
golden (that only holds the D256 4wgqa kernel) and the on-GPU fp32 oracle passes
both the folded and unfolded kernels, so nothing else guards it.

These tests run off-GPU (build/emit + pure index math) and are written to FAIL
against the pre-fold kernel:
  * the launch grid must fold over KV heads (grid.x == num_kv_heads, not
    num_query_heads) with 32-token blocks -- pre-fold it is num_query_heads;
  * the builder must emit the distinct ``_4wgqa_fold`` kernel name -- pre-fold
    it is always ``_4wgqa``;
  * the fold predicate must select exactly the D128 / 4:1-GQA / SWA / bf16 /
    block<=32 cohort and reject every other axis;
  * the row->(token,head) map must be a bijection with the head index contained
    in [0, num_query_heads) and an exact inverse (the O-store map);
  * non-divisible head counts must raise (the invariant the fold relies on).
"""

from __future__ import annotations

import re
import unittest

import kernels.common.attention_unified as au
from kernels.common.attention_unified import (
    _get_2d_launch_meta,
    _tiled_cache_key,
    _tiled_spec_from_problem,
)

# `gfx942_gqa_fold_eligible` is imported lazily inside the predicate tests so this
# file also imports against pre-fold `develop` -- there the grid/emit tests RUN and
# their assertions FAIL (the tripwire), rather than the whole module erroring.
from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa
from rocke.core.ir_print import print_ir


class _PinArch:
    """Pin ``_RESOLVED_ATTENTION_ARCH`` so spec/route resolution runs off-GPU."""

    def __init__(self, arch: str):
        self.arch = arch

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = self.arch
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


def _problem(
    dtype="bf16",
    block_size=16,
    head_size=128,
    sliding_window=4096,
    num_query_heads=32,
    num_kv_heads=8,
    sq=8192,
):
    return au.UnifiedAttentionProblem(
        total_q=sq,
        num_seqs=1,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        max_seqlen_q=sq,
        max_seqlen_k=sq,
        dtype=dtype,
        sliding_window=sliding_window,
    )


class TestGfx942GqaHeadFoldPredicate(unittest.TestCase):
    def test_fold_cohort_selected(self):
        from kernels.common.attention_unified import gfx942_gqa_fold_eligible

        # D128, 4:1 GQA (nqpk==4), SWA, bf16, block in {16,32} -> fold.
        for bs in (16, 32):
            self.assertTrue(
                gfx942_gqa_fold_eligible(128, 4, 4096, "bf16", bs),
                msg=f"block_size={bs} should be fold-eligible",
            )

    def test_each_excluded_axis_rejected(self):
        from kernels.common.attention_unified import gfx942_gqa_fold_eligible

        # Every non-qualifying axis must fall back (take the unfolded path).
        cases = {
            "fp16": (128, 4, 4096, "fp16", 16),
            "non-4:1 GQA": (128, 8, 4096, "bf16", 16),
            "causal (no window)": (128, 4, 0, "bf16", 16),
            "D256": (256, 4, 4096, "bf16", 16),
            "block>32": (128, 4, 4096, "bf16", 64),
        }
        for name, args in cases.items():
            self.assertFalse(
                gfx942_gqa_fold_eligible(*args),
                msg=f"{name} must NOT be fold-eligible",
            )


class TestGfx942GqaHeadFoldLaunchGrid(unittest.TestCase):
    """The single-source predicate drives the grid; fold must grid over KV heads."""

    def _grid(self, problem):
        au._2D_LAUNCH_META.clear()
        with _PinArch("gfx942"):
            return _get_2d_launch_meta(problem, _tiled_cache_key(problem)).grid

    def test_fold_grids_over_kv_heads(self):
        p = _problem(dtype="bf16", block_size=16)  # fold-eligible
        gx, gy, gz = self._grid(p)
        # FAILS before the change: pre-fold grid.x == num_query_heads (32).
        self.assertEqual(gx, p.num_kv_heads, "fold grid.x must be num_kv_heads")
        self.assertEqual(
            gy, p.total_q // 32 + p.num_seqs, "fold grid.y = 32-token blocks"
        )
        self.assertNotEqual(gx, p.num_query_heads)

    def test_nonfold_grids_over_query_heads(self):
        p = _problem(dtype="fp16", block_size=16)  # 4-warp route, but fold-excluded
        gx, _, _ = self._grid(p)
        self.assertEqual(
            gx, p.num_query_heads, "non-fold grid.x must be num_query_heads"
        )


class TestGfx942GqaHeadFoldEmit(unittest.TestCase):
    def _ir(self, dtype, block_size=16):
        with _PinArch("gfx942"):
            spec = _tiled_spec_from_problem(
                _problem(dtype=dtype, block_size=block_size)
            )
            return print_ir(build_gfx942_4warp_gqa(spec, arch="gfx942"))

    def test_fold_emits_distinct_kernel_name(self):
        ir = self._ir("bf16")
        # FAILS before the change: pre-fold kernel name has no `_fold` suffix.
        self.assertTrue(
            re.search(r"_4wgqa_fold\b", ir), "fold kernel must emit `_4wgqa_fold`"
        )

    def test_nonfold_keeps_baseline_name(self):
        ir = self._ir("fp16")
        self.assertFalse(re.search(r"_4wgqa_fold\b", ir), "fp16 must not fold")
        self.assertTrue(
            re.search(r"_4wgqa\b", ir), "fp16 keeps the baseline `_4wgqa` name"
        )


class TestGfx942GqaHeadFoldMap(unittest.TestCase):
    """The row->(token,head) invariants the folded kernel relies on (§H)."""

    def test_mmap_bijection_head_range_and_inverse(self):
        TOKBLK, GQAG, TILE_M = 32, 4, 128
        H, HKV = 32, 8  # H == HKV * GQAG
        seen = set()
        for m in range(TILE_M):
            tok, hloc = m // GQAG, m % GQAG  # forward map (Q-load / mask / O-store)
            self.assertTrue(0 <= tok < TOKBLK and 0 <= hloc < GQAG)
            self.assertEqual(tok * GQAG + hloc, m)  # exact inverse (O-store recovers m)
            seen.add((tok, hloc))
            for kvh in range(HKV):  # folded global head must stay in range
                self.assertTrue(0 <= kvh * GQAG + hloc < H)
        # bijection: 128 rows cover exactly 32 tokens x 4 heads, each once.
        self.assertEqual(len(seen), TILE_M)
        self.assertEqual(seen, {(t, h) for t in range(TOKBLK) for h in range(GQAG)})


class TestGfx942GqaHeadFoldInvariant(unittest.TestCase):
    def test_nondivisible_head_counts_raise(self):
        # The fold requires num_query_heads == num_kv_heads * nqpk; the spec must
        # reject a non-divisible pairing rather than silently overflow the head index.
        p = _problem(num_query_heads=30, num_kv_heads=8)
        with self.assertRaises(ValueError):
            _ = p.num_queries_per_kv


if __name__ == "__main__":
    unittest.main()
