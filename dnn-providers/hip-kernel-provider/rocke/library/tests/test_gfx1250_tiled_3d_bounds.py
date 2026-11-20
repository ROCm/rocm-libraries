# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every paged read of the gfx1250 split-KV 3D kernel is bounded to its sequence.

The kernel walks 32-token KV tiles (``_T``) and admits ``block_size`` 16 and 32.
At 16 a tile spans two paged blocks, so the tail tile of a sequence whose length
is not a tile multiple asks the block table for a logical block past the last one
the host filled in: decode with ``seq_len == 40`` reads entry 3 when only pages
0-2 are valid, and ``bindings.py`` validates only ``row[:cdiv(seq_len,
block_size)]``. Those keys are masked to -inf, but the mask is applied to the
score, after the K/V load built from the page id has already been issued -- an
arbitrary page id is an illegal access, the AICK-2955 failure on the gfx942 D128
body. ``block_size == 32`` is bounded by construction (one page per tile) and is
covered here so a future tile-geometry change cannot quietly unbound it.

All block-table reads funnel through the builder's ``_phys_block`` closure, so
these assert the clamp on the variants that reach it by structurally distinct
paths: single-wave, cooperative multi-wave, DTLA async-V staging, and the
software-pipeline stack.

No gfx1250 GPU is needed (or available in CI): the checks walk the builder's IR.
The arch has no C++ mirror under ``platform/cpp/instances/``, so there is no twin
to assert against.

    python3 -m pytest rocke/library/tests/test_gfx1250_tiled_3d_bounds.py
"""

from __future__ import annotations

import unittest

from kernels.gfx1250.attention_tiled_3d import (
    UnifiedAttention3DTiledSpec,
    build_unified_attention_3d_tiled,
)

from ._bounds_check_ir import clamped_to_last_of, global_loads, walk_ops

# (tag, block_size, spec overrides, stages V through the async DMA) -- one per
# structurally distinct KV path. The DTLA and software-pipeline paths stage V
# with ``tile.global_load_async_to_lds`` instead of a ``memref.global_load``,
# which is exactly the difference that made an earlier version of the V check
# silently inspect nothing on those two.
_VARIANTS = (
    ("single-wave bs16", 16, {}, False),
    ("multi-wave bs16", 16, {"num_waves": 4, "use_wide_lds_reads": False}, False),
    (
        "dtla bs16",
        16,
        {"use_dtla_prefetch": True, "use_wide_lds_reads": False},
        True,
    ),
    (
        "sw-pipeline bs16",
        16,
        {"use_sw_pipeline": True, "use_wide_lds_reads": False},
        True,
    ),
    ("single-wave bs32", 32, {}, False),
)

_ASYNC_V_OP = "tile.global_load_async_to_lds"


def _kernel(block_size: int, **overrides):
    spec = UnifiedAttention3DTiledSpec(
        head_size=64,
        block_size=block_size,
        num_query_heads=32,
        num_kv_heads=4,
        dtype="bf16",
        use_sinks=True,
        sliding_window=0,
        has_softcap=False,
        num_segments=16,
        num_seqs=256,
        kv_storage_dtype="bf16",
        **overrides,
    )
    return build_unified_attention_3d_tiled(spec, arch="gfx1250")


class TestGfx1250Tiled3dBoundsReads(unittest.TestCase):
    def _kernels(self):
        for tag, bs, overrides, async_v in _VARIANTS:
            yield tag, _kernel(bs, **overrides), bs, async_v

    def test_each_variant_builds_a_distinct_kernel(self):
        # Guards the variant table itself. A lever that is renamed, retired or
        # silently ignored collapses two rows onto the same kernel, which would
        # leave the bounds assertions below passing while covering one path
        # fewer. Comparing the count of tags against the table cannot catch
        # that: it is true by construction.
        names = [k.name for _, k, _, _ in self._kernels()]
        self.assertEqual(len(set(names)), len(_VARIANTS), names)

    def test_async_staging_levers_are_live(self):
        # The DTLA and software-pipeline rows exist to cover the async-DMA V
        # path. If either lever stops emitting the async op, those rows no
        # longer test anything the plain rows do not, so pin the op per row.
        for tag, k, _, async_v in self._kernels():
            with self.subTest(variant=tag):
                n = sum(1 for op in walk_ops(k.body) if op.name == _ASYNC_V_OP)
                if async_v:
                    self.assertTrue(n, f"{tag}: expected async V staging, found none")
                else:
                    self.assertFalse(n, f"{tag}: unexpected async V staging")

    def test_block_table_reads_clamped_to_last_page(self):
        for tag, k, bs, _ in self._kernels():
            with self.subTest(variant=tag):
                loads = global_loads(k, "block_tables_ptr")
                self.assertTrue(loads, f"{tag}: no block-table read found")
                for op in loads:
                    self.assertTrue(
                        clamped_to_last_of(op.operands[1], "seq_lens_ptr", bs),
                        f"{tag}: block-table read not bounded by the last page",
                    )

    def test_kv_reads_go_through_the_clamped_block_table(self):
        for tag, k, bs, _ in self._kernels():
            for param in ("key_cache_ptr", "value_cache_ptr"):
                with self.subTest(variant=tag, param=param):
                    loads = global_loads(k, param)
                    # An empty list would assert nothing: the loop below is
                    # only meaningful if the reads were actually found.
                    self.assertTrue(loads, f"{tag}: no {param} read found")
                    for op in loads:
                        self.assertTrue(
                            clamped_to_last_of(op.operands[1], "seq_lens_ptr", bs),
                            f"{tag}: {param} read not bounded by the last page",
                        )


if __name__ == "__main__":
    unittest.main()
