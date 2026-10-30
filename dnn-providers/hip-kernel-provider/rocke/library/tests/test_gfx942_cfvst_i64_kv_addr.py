# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 conflict-free V store (cfvst) V loads address paged caches past 2 GiB.

The shipped gfx942 bf16 D64 causal prefill spec selects cfvst. Under
``use_i64_kv_addr`` every load call's token rows lie in one page, so V loads
index that page's pointer (a 64-bit page base) with an i32 offset, which keeps
the register cost of the i32 path. The builder refuses a geometry where a call
spans pages; no cfvst geometry the gfx942 tuning space offers does. Below the
i32 limit the spec stays i32 and indexes the cache base with i32 offsets. The
two diagnostic load forms (HIPDNN_GFX942_CFV_STORE_SEPOFF / _SCALAR_LOAD) follow
the same addressing.

No GPU: the specs are built and lowered to LLVM text.
"""

from __future__ import annotations

import os
import re
import unittest
from unittest import mock

import builders.common.attention_spec_builder as _asb
import kernels.common.attention_unified as _au
from dataclasses import replace

from dispatch.attention import AttentionRequest, attention_candidates
from kernels import UnifiedAttentionProblem
from kernels.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from rocke.core.lower_llvm import lower_kernel_to_llvm

# 32 KiB blocks (bf16 D64, 8 kv heads, 32 tokens): 65534 blocks is exactly the
# i32 buffer range, 69632 blocks is 2.125 GiB.
_AT_I32_LIMIT = 65534
_PAST_2GIB = 69632
# A page pointer: value_cache_ptr plus a 64-bit byte offset.
_PAGE_PTR = re.compile(
    r"(%gptr\d+) = getelementptr inbounds i8, ptr addrspace\(1\) %value_cache_ptr, i64 "
)
# A typed V element load address: (base, index width).
_V_GEP = re.compile(
    r"getelementptr inbounds (?:bfloat|half), ptr addrspace\(1\) (%[\w.]+), (i\d+) "
)


def _spec(num_kv_blocks: int):
    p = UnifiedAttentionProblem(
        total_q=8192,
        num_seqs=2,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=64,
        block_size=32,
        max_seqlen_q=4096,
        max_seqlen_k=4096,
        dtype="bf16",
        num_kv_blocks=num_kv_blocks,
    )
    with mock.patch.object(_au, "_resolve_attention_arch", return_value="gfx942"):
        return _asb._tiled_spec_from_problem(p, "gfx942")


def _spanning_spec():
    """i64 cfvst on 8-token pages: a 16-token load call spans two pages."""
    return UnifiedAttention2DTiledSpec(
        head_size=64,
        block_size=8,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32x8=True,
        use_transposed_qk_32x32=True,
        use_conflict_free_v_store=True,
        use_i64_kv_addr=True,
    )


def _v_loads(spec, env=None) -> tuple[list[tuple[str, str]], set[str]]:
    """The V element loads as (base kind, index width), and the page pointers.

    Base kind is ``"cache"`` for value_cache_ptr and ``"page"`` for a page
    pointer derived from it. K loads index key_cache_ptr and are excluded.
    """
    with mock.patch.dict(os.environ, env or {}):
        kernel = build_unified_attention_2d_tiled(spec, arch="gfx942")
    llvm = lower_kernel_to_llvm(kernel, arch="gfx942")
    pages = set(_PAGE_PTR.findall(llvm))
    loads = []
    for base, width in _V_GEP.findall(llvm):
        if base == "%value_cache_ptr":
            loads.append(("cache", width))
        elif base in pages:
            loads.append(("page", width))
    return loads, pages


class TestGfx942CfvstI64KvAddr(unittest.TestCase):
    def test_shipped_spec_past_2gib_is_i64_cfvst(self):
        spec = _spec(_PAST_2GIB)
        self.assertTrue(spec.use_conflict_free_v_store)
        self.assertTrue(spec.use_i64_kv_addr)

    def test_i64_spec_loads_v_from_page_pointers(self):
        spec = _spec(_PAST_2GIB)
        loads, pages = _v_loads(spec)
        self.assertTrue(loads)
        self.assertEqual(set(loads), {("page", "i32")})
        # One page pointer per page of the tile, not one per load.
        self.assertEqual(len(pages), spec.tile_size_eff // spec.block_size)

    def test_spanning_geometry_is_refused(self):
        with self.assertRaisesRegex(ValueError, "within one page"):
            build_unified_attention_2d_tiled(_spanning_spec(), arch="gfx942")

    def test_every_tuning_cfvst_geometry_builds_i64(self):
        # A tuning axis that let a load call span pages would make the i64 spec
        # refuse to build, so pin that every offered cfvst geometry builds.
        geometries = {}
        for candidate in attention_candidates():
            if "gfx942" not in candidate.name:
                continue
            for hd in (64, 128):
                req = AttentionRequest(
                    batch=2,
                    nhead_q=32,
                    nhead_k=8,
                    seqlen_q=4096,
                    seqlen_k=4096,
                    hdim_q=hd,
                    hdim_v=hd,
                    arch="gfx942",
                    dtype="bf16",
                    algorithm=candidate.algorithm,
                    spec_id=candidate.spec_id,
                )
                for tuning in candidate.sweep_space(req):
                    spec = tuning.kernel_spec
                    if getattr(spec, "use_conflict_free_v_store", False):
                        key = (spec.head_size, spec.block_size, spec.num_warps)
                        geometries.setdefault(key + (spec.tile_size_eff,), spec)
        self.assertTrue(geometries)
        for key, spec in sorted(geometries.items()):
            with self.subTest(geometry=key):
                build_unified_attention_2d_tiled(
                    replace(spec, use_i64_kv_addr=True), arch="gfx942"
                )

    def test_i32_spec_v_loads_keep_i32_offsets(self):
        spec = _spec(_AT_I32_LIMIT)
        self.assertTrue(spec.use_conflict_free_v_store)
        self.assertFalse(spec.use_i64_kv_addr)
        loads, pages = _v_loads(spec)
        self.assertTrue(loads)
        self.assertEqual(set(loads), {("cache", "i32")})
        self.assertFalse(pages)

    def test_diagnostic_load_forms_follow_the_addressing(self):
        cases = ((_spec(_PAST_2GIB), ("page", "i32")), (_spec(0), ("cache", "i32")))
        for spec, expected in cases:
            vector_loads = len(_v_loads(spec)[0])
            for flag in (
                "HIPDNN_GFX942_CFV_STORE_SEPOFF",
                "HIPDNN_GFX942_CFV_STORE_SCALAR_LOAD",
            ):
                with self.subTest(i64=spec.use_i64_kv_addr, flag=flag):
                    loads, _ = _v_loads(spec, {flag: "1"})
                    # Two scalar loads per dim pair, so the flag really took effect.
                    self.assertEqual(len(loads), 2 * vector_loads)
                    self.assertEqual(set(loads), {expected})


if __name__ == "__main__":
    unittest.main()
