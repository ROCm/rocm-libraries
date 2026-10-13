# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""I32 byte-extent and FP8 vector-alignment guards of the gfx1151 attention
binding (host metadata only)."""

import unittest
from types import SimpleNamespace

from dispatch.attention import AttentionRequest
from dispatch.attention.bindings import (
    _check_byte_extent_i32,
    _gfx1151_validate_and_collect,
)
from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec


class _FakeTensor:
    def __init__(self, shape, stride=None, dtype="torch.float16", ptr=0):
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
        self._ptr = ptr

    def stride(self, axis=None):
        return self._stride if axis is None else self._stride[axis]

    def data_ptr(self):
        return self._ptr


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
    def test_helper_accepts_up_to_the_i32_byte_limit(self):
        _check_byte_extent_i32(_FakeTensor((1, 0x3FFF_FFFF)), "t", 2)
        _check_byte_extent_i32(_FakeTensor((1, 0x7FFF_FFFF)), "t", 1)

    def test_helper_rejects_one_element_past_the_i32_byte_limit(self):
        with self.assertRaisesRegex(ValueError, "does not fit"):
            _check_byte_extent_i32(_FakeTensor((1, 0x4000_0000)), "t", 2)

    def test_fp16_tensor_with_i32_element_offsets_is_rejected(self):
        # 16 x 32K x 32 x 128 fp16: the highest element offset (2**31 - 1)
        # fits I32 but the byte offset (~4 GiB) does not; the transposed-QK V
        # gather would read past the 2 GiB descriptor and get zeros.
        with self.assertRaisesRegex(ValueError, "v spans"):
            _check_byte_extent_i32(_FakeTensor((16, 32768, 32, 128)), "v", 2)

    def test_helper_follows_strides_not_element_count(self):
        # Few elements, huge span: only the stride-aware bound catches this.
        with self.assertRaises(ValueError):
            _check_byte_extent_i32(_FakeTensor((2, 4), stride=(1 << 30, 1)), "t", 2)

    def test_helper_ignores_empty_tensors(self):
        _check_byte_extent_i32(_FakeTensor((0, 1 << 40)), "t", 2)

    def test_small_paged_cache_is_accepted(self):
        request, spec, tensors = _paged_case(pages=8)
        values = _gfx1151_validate_and_collect(request, spec, tensors)
        self.assertIs(values["K"], tensors["k"])

    def test_oversized_kv_cache_is_rejected_before_launch(self):
        # 2**20 pages * 16 tokens * 4 heads * 64 dims = 2**32 elements.
        request, spec, tensors = _paged_case(pages=1 << 20)
        with self.assertRaisesRegex(ValueError, "k spans .* bytes"):
            _gfx1151_validate_and_collect(request, spec, tensors)


def _dense_fp8_case(kv_token_stride=None, kv_ptr=0, v_lds_stage=False):
    batch, seq, heads, dim = 1, 32, 2, 64
    request = AttentionRequest(
        batch=batch,
        nhead_q=heads,
        nhead_k=heads,
        seqlen_q=seq,
        seqlen_k=seq,
        hdim_q=dim,
        hdim_v=dim,
        arch="gfx1151",
        dtype="fp16",
        use_fp8=True,
    )
    spec = WmmaFmhaFwdSpec(head_size=dim, kv_dtype="fp8e4m3", v_lds_stage=v_lds_stage)
    row = heads * dim if kv_token_stride is None else kv_token_stride
    kv_stride = (seq * row, row, dim, 1)
    kv = dict(stride=kv_stride, dtype="torch.float8_e4m3fn", ptr=kv_ptr)
    tensors = {
        "q": _FakeTensor((batch, seq, heads, dim)),
        "out": _FakeTensor((batch, seq, heads, dim)),
        "k": _FakeTensor((batch, seq, heads, dim), **kv),
        "v": _FakeTensor((batch, seq, heads, dim), **kv),
    }
    return request, spec, tensors


class TestGfx1151Fp8VectorAlignment(unittest.TestCase):
    """FP8 K rows are read as 16-byte vectors and LDS-staged FP8 V rows as
    8-byte vectors, emitted with that alignment; misaligned layouts must be
    rejected rather than read incorrectly."""

    def test_aligned_layout_is_accepted(self):
        request, spec, tensors = _dense_fp8_case(kv_token_stride=144)
        values = _gfx1151_validate_and_collect(request, spec, tensors)
        self.assertIs(values["K"], tensors["k"])

    def test_token_stride_off_the_16_byte_grid_is_rejected(self):
        # Non-overlapping (136 >= 2 * 64) but row 1 of K starts at byte 136.
        request, spec, tensors = _dense_fp8_case(kv_token_stride=136)
        with self.assertRaisesRegex(ValueError, r"k stride\(1\)=136"):
            _gfx1151_validate_and_collect(request, spec, tensors)

    def test_misaligned_data_pointer_is_rejected(self):
        request, spec, tensors = _dense_fp8_case(kv_ptr=8)
        with self.assertRaisesRegex(ValueError, "k data pointer"):
            _gfx1151_validate_and_collect(request, spec, tensors)

    def test_staged_v_needs_only_8_byte_rows(self):
        # Direct V gathers are scalar; LDS-staged V loads 8-byte vectors.
        request, spec, tensors = _dense_fp8_case(v_lds_stage=True)
        tensors["v"] = _FakeTensor(
            (1, 32, 2, 64),
            stride=(32 * 136, 136, 64, 1),
            dtype="torch.float8_e4m3fn",
        )
        _gfx1151_validate_and_collect(request, spec, tensors)
        tensors["v"]._stride = (32 * 132, 132, 64, 1)
        with self.assertRaisesRegex(ValueError, r"v stride\(1\)=132"):
            _gfx1151_validate_and_collect(request, spec, tensors)


if __name__ == "__main__":
    unittest.main()
