# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""I32 element-offset guard of the gfx1151 attention binding (host metadata only)."""

import unittest
from types import SimpleNamespace

from dispatch.attention import AttentionRequest
from dispatch.attention.bindings import (
    _check_max_element_offset_i32,
    _gfx1151_validate_and_collect,
)
from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec


class _FakeTensor:
    def __init__(self, shape, stride=None, dtype="torch.float16"):
        self.shape = tuple(shape)
        if stride is None:
            stride, running = [], 1
            for extent in reversed(self.shape):
                stride.append(running)
                running *= extent
            stride = tuple(reversed(stride))
        self._stride = tuple(stride)
        self.dtype = dtype
        self.device = SimpleNamespace(type="cuda")

    def stride(self, axis=None):
        return self._stride if axis is None else self._stride[axis]

    def data_ptr(self):
        return 0


def _paged_case(pages):
    heads, dim, block = 4, 64, 16
    request = AttentionRequest(
        batch=1,
        nhead_q=heads,
        nhead_k=heads,
        seqlen_q=16,
        seqlen_k=64,
        hdim_q=dim,
        hdim_v=dim,
        arch="gfx1151",
        dtype="fp16",
        layout="paged",
        kv_block_size=block,
    )
    spec = WmmaFmhaFwdSpec(
        head_size=dim,
        num_query_heads=heads,
        layout="paged",
        page_block_size=block,
        query_tail=True,
        kv_tail=True,
    )
    cache = _FakeTensor((pages, block, heads, dim))
    tensors = {
        "q": _FakeTensor((16, heads, dim)),
        "out": _FakeTensor((16, heads, dim)),
        "k": cache,
        "v": cache,
        "cu_seqlens_q": _FakeTensor((2,), dtype="torch.int32"),
        "seqused_k": _FakeTensor((1,), dtype="torch.int32"),
        "block_table": _FakeTensor((1, 4), dtype="torch.int32"),
    }
    return request, spec, tensors


class TestGfx1151BindingI32Guard(unittest.TestCase):
    def test_helper_accepts_up_to_the_i32_limit(self):
        _check_max_element_offset_i32(_FakeTensor((1, 0x8000_0000)), "t")

    def test_helper_rejects_one_past_the_i32_limit(self):
        with self.assertRaisesRegex(ValueError, "does not fit"):
            _check_max_element_offset_i32(_FakeTensor((1, 0x8000_0001)), "t")

    def test_helper_follows_strides_not_element_count(self):
        # Few elements, huge span: only the stride-aware bound catches this.
        with self.assertRaises(ValueError):
            _check_max_element_offset_i32(_FakeTensor((2, 4), stride=(1 << 31, 1)), "t")

    def test_helper_ignores_empty_tensors(self):
        _check_max_element_offset_i32(_FakeTensor((0, 1 << 40)), "t")

    def test_small_paged_cache_is_accepted(self):
        request, spec, tensors = _paged_case(pages=8)
        values = _gfx1151_validate_and_collect(request, spec, tensors)
        self.assertIs(values["K"], tensors["k"])

    def test_oversized_kv_cache_is_rejected_before_launch(self):
        # 2**20 pages * 16 tokens * 4 heads * 64 dims = 2**32 elements.
        request, spec, tensors = _paged_case(pages=1 << 20)
        with self.assertRaisesRegex(ValueError, "k spans element offset"):
            _gfx1151_validate_and_collect(request, spec, tensors)


if __name__ == "__main__":
    unittest.main()
