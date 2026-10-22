# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 conflict-free V store (cfvst) V loads address paged caches past 2 GiB.

The shipped gfx942 bf16 D64 causal prefill spec selects cfvst. Under
``use_i64_kv_addr`` every V load must index ``value_cache`` with an i64 element
offset, or a cache past 2^31 bytes wraps. Below the i32 limit the spec stays i32
and its V loads keep i32 offsets. The two diagnostic load forms
(HIPDNN_GFX942_CFV_STORE_SEPOFF / _SCALAR_LOAD) take the same i64 offset.

No GPU: the specs come from the real selector and are lowered to LLVM text.
"""

from __future__ import annotations

import os
import re
import unittest
from unittest import mock

import builders.common.attention_spec_builder as _asb
import kernels.common.attention_unified as _au
from kernels import UnifiedAttentionProblem
from kernels.gfx942.attention_tiled_2d import build_unified_attention_2d_tiled
from rocke.core.lower_llvm import lower_kernel_to_llvm

# 32 KiB blocks (bf16 D64, 8 kv heads, 32 tokens): 65534 blocks is exactly the
# i32 buffer range, 69632 blocks is 2.125 GiB.
_AT_I32_LIMIT = 65534
_PAST_2GIB = 69632
_V_GEP = re.compile(
    r"getelementptr inbounds \w+, ptr addrspace\(1\) %value_cache_ptr, (i\d+) "
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


def _v_gep_widths(spec, env=None) -> list[str]:
    with mock.patch.dict(os.environ, env or {}):
        kernel = build_unified_attention_2d_tiled(spec, arch="gfx942")
    return _V_GEP.findall(lower_kernel_to_llvm(kernel, arch="gfx942"))


class TestGfx942CfvstI64KvAddr(unittest.TestCase):
    def test_shipped_spec_past_2gib_is_i64_cfvst(self):
        spec = _spec(_PAST_2GIB)
        self.assertTrue(spec.use_conflict_free_v_store)
        self.assertTrue(spec.use_i64_kv_addr)

    def test_i64_spec_v_loads_use_i64_offsets(self):
        widths = _v_gep_widths(_spec(_PAST_2GIB))
        self.assertTrue(widths)
        self.assertEqual(set(widths), {"i64"})

    def test_i32_spec_v_loads_keep_i32_offsets(self):
        spec = _spec(_AT_I32_LIMIT)
        self.assertTrue(spec.use_conflict_free_v_store)
        self.assertFalse(spec.use_i64_kv_addr)
        widths = _v_gep_widths(spec)
        self.assertTrue(widths)
        self.assertEqual(set(widths), {"i32"})

    def test_diagnostic_load_forms_use_i64_offsets(self):
        spec = _spec(_PAST_2GIB)
        vector_loads = len(_v_gep_widths(spec))
        for flag in (
            "HIPDNN_GFX942_CFV_STORE_SEPOFF",
            "HIPDNN_GFX942_CFV_STORE_SCALAR_LOAD",
        ):
            with self.subTest(flag=flag):
                widths = _v_gep_widths(spec, {flag: "1"})
                # Two scalar loads per dim pair, so the flag really took effect.
                self.assertEqual(len(widths), 2 * vector_loads)
                self.assertEqual(set(widths), {"i64"})


if __name__ == "__main__":
    unittest.main()
