# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""I32 byte-extent and row-alignment guards of the RDNA attention binding
(host metadata only), for both the prefill (``WmmaFmhaFwdV2Spec``) and the
split-KV decode (``WmmaFmhaDecodeSpec``) objects."""

import dataclasses
import unittest
from types import SimpleNamespace

from dispatch.attention import AttentionRequest
from dispatch.attention.bindings import (
    _check_byte_extent_i32,
    _rdna_scalars,
    _rdna_validate_and_collect,
)
from kernels.gfx1151.wmma_fmha_decode import (
    WmmaFmhaDecodeSpec,
    resolve_wmma_fmha_decode_spec,
)
from kernels.gfx1151.wmma_fmha_fwd_v2 import WmmaFmhaFwdV2Spec

_KINDS = ("prefill", "decode")


def _spec(kind, **kw):
    """One spec of ``kind`` with neutral feature names: ``head`` (bucket),
    ``kv_dtype``, ``layout``, ``page_block_size``."""
    head = kw.pop("head", 64)
    if kind == "prefill":
        return WmmaFmhaFwdV2Spec(dtype="fp16", head_dim=head, **kw)
    return resolve_wmma_fmha_decode_spec(
        WmmaFmhaDecodeSpec(head_size=head, dtype="fp16", **kw), arch="gfx1151"
    )


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


def _paged_case(kind, pages):
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
    spec = _spec(kind, layout="paged", page_block_size=block)
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


class TestRdnaBindingI32Guard(unittest.TestCase):
    def test_helper_accepts_up_to_the_i32_byte_limit(self):
        _check_byte_extent_i32(_FakeTensor((1, 0x3FFF_FFFF)), "t", 2)
        _check_byte_extent_i32(_FakeTensor((1, 0x7FFF_FFFF)), "t", 1)

    def test_helper_rejects_one_element_past_the_i32_byte_limit(self):
        with self.assertRaisesRegex(ValueError, "does not fit"):
            _check_byte_extent_i32(_FakeTensor((1, 0x4000_0000)), "t", 2)

    def test_fp16_tensor_with_i32_element_offsets_is_rejected(self):
        # 16 x 32K x 32 x 128 fp16: the highest element offset (2**31 - 1)
        # fits I32 but the byte offset (~4 GiB) does not; a buffer load past
        # the 2 GiB descriptor would read zeros.
        with self.assertRaisesRegex(ValueError, "v spans"):
            _check_byte_extent_i32(_FakeTensor((16, 32768, 32, 128)), "v", 2)

    def test_helper_follows_strides_not_element_count(self):
        # Few elements, huge span: only the stride-aware bound catches this.
        with self.assertRaises(ValueError):
            _check_byte_extent_i32(_FakeTensor((2, 4), stride=(1 << 30, 1)), "t", 2)

    def test_helper_ignores_empty_tensors(self):
        _check_byte_extent_i32(_FakeTensor((0, 1 << 40)), "t", 2)

    def test_small_paged_cache_is_accepted(self):
        for kind in _KINDS:
            with self.subTest(kind=kind):
                request, spec, tensors = _paged_case(kind, pages=8)
                values = _rdna_validate_and_collect(request, spec, tensors)
                self.assertIs(values["K"], tensors["k"])

    def test_oversized_kv_cache_is_rejected_before_launch(self):
        # 2**20 pages * 16 tokens * 4 heads * 64 dims = 2**32 elements.
        for kind in _KINDS:
            with self.subTest(kind=kind):
                request, spec, tensors = _paged_case(kind, pages=1 << 20)
                with self.assertRaisesRegex(ValueError, "k spans .* bytes"):
                    _rdna_validate_and_collect(request, spec, tensors)


def _dense_case(kind, **spec_kw):
    batch, seq, heads, dim = 1, 16, 4, 64
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
        emit_lse=True,
    )
    spec = _spec(kind, **spec_kw)
    full = (batch, seq, heads, dim)
    tensors = {
        "q": _FakeTensor(full),
        "k": _FakeTensor(full),
        "v": _FakeTensor(full),
        "out": _FakeTensor(full),
    }
    return request, spec, tensors


# Strides that keep every operand's element count tiny but whose byte extent
# exceeds 2 GiB: one head (or row) sits 2**28 floats = 1 GiB apart.
_FAR = 1 << 28


class TestRdnaOperandByteExtent(unittest.TestCase):
    """Operands with their own call site in the binding (everything but the shared
    q / k / v / out loop) are rejected when they span more than 2 GiB."""

    def _check(self, tensors, request, spec, name, fits):
        """``fits`` operands are accepted as built; the oversized one is rejected."""
        _rdna_validate_and_collect(request, spec, fits)
        with self.assertRaisesRegex(ValueError, f"{name} spans .* bytes"):
            _rdna_validate_and_collect(request, spec, tensors)

    def test_dense_lse(self):
        for kind in _KINDS:
            with self.subTest(kind=kind):
                request, spec, tensors = _dense_case(kind)
                tensors["lse"] = _FakeTensor((1, 4, 16), dtype="torch.float32")
                fits = dict(tensors)
                tensors["lse"] = _FakeTensor(
                    (1, 4, 16), stride=(1, _FAR, 1), dtype="torch.float32"
                )
                self._check(tensors, request, spec, "lse", fits)

    def test_packed_lse(self):
        for kind in _KINDS:
            with self.subTest(kind=kind):
                request, spec, tensors = _paged_case(kind, pages=8)
                request = dataclasses.replace(request, emit_lse=True)
                tensors["lse"] = _FakeTensor((4, 16), dtype="torch.float32")
                fits = dict(tensors)
                tensors["lse"] = _FakeTensor(
                    (4, 16), stride=(_FAR, 1), dtype="torch.float32"
                )
                self._check(tensors, request, spec, "lse", fits)

    def test_qq_bias(self):
        for kind in _KINDS:
            with self.subTest(kind=kind):
                feature = "use_qq_bias" if kind == "prefill" else "qq_bias"
                request, spec, tensors = _dense_case(kind, **{feature: True})
                tensors["lse"] = _FakeTensor((1, 4, 16), dtype="torch.float32")
                tensors["qq_bias"] = _FakeTensor((2, 16), dtype="torch.float32")
                fits = dict(tensors)
                tensors["qq_bias"] = _FakeTensor(
                    (2, 16), stride=(1 << 30, 1), dtype="torch.float32"
                )
                self._check(tensors, request, spec, "qq_bias", fits)

    def test_attn_bias(self):
        for kind, bias, dtype in (
            ("prefill", "f32", "torch.float32"),
            ("prefill", "q", "torch.float16"),
            ("decode", "f32", "torch.float32"),
            ("decode", "q", "torch.float16"),
        ):
            with self.subTest(kind=kind, bias=bias):
                request, spec, tensors = _dense_case(kind, bias=bias)
                tensors["lse"] = _FakeTensor((1, 4, 16), dtype="torch.float32")
                tensors["attn_bias"] = _FakeTensor((1, 4, 16, 16), dtype=dtype)
                fits = dict(tensors)
                # One head 2**29 elements apart: over 2 GiB for 2- and 4-byte elements.
                tensors["attn_bias"] = _FakeTensor(
                    (1, 4, 16, 16), stride=(0, 1 << 29, 16, 1), dtype=dtype
                )
                self._check(tensors, request, spec, "attn_bias", fits)


def _dense_fp8_case(kind, kv_token_stride=None, kv_ptr=0):
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
    spec = _spec(kind, kv_dtype="fp8e4m3")
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


class TestRdnaFp8RowAlignment(unittest.TestCase):
    """FP8 K/V rows load as 8-byte vectors (16-bit rows as 16-byte ones);
    misaligned layouts must be rejected rather than read incorrectly."""

    def test_aligned_layouts_are_accepted(self):
        # 136 bytes is off the 16-byte grid but on the 8-byte FP8 row grid.
        for kind in _KINDS:
            for stride in (128, 136, 144):
                with self.subTest(kind=kind, stride=stride):
                    request, spec, tensors = _dense_fp8_case(
                        kind, kv_token_stride=stride
                    )
                    values = _rdna_validate_and_collect(request, spec, tensors)
                    self.assertIs(values["K"], tensors["k"])
                    self.assertEqual(values["stride_k_token"], stride)

    def test_token_stride_off_the_8_byte_grid_is_rejected(self):
        # Non-overlapping (132 >= 2 * 64) but row 1 of K starts at byte 132.
        for kind in _KINDS:
            with self.subTest(kind=kind):
                request, spec, tensors = _dense_fp8_case(kind, kv_token_stride=132)
                with self.assertRaisesRegex(ValueError, r"k stride\(1\)=132"):
                    _rdna_validate_and_collect(request, spec, tensors)

    def test_v_rows_follow_the_same_rule(self):
        for kind in _KINDS:
            with self.subTest(kind=kind):
                request, spec, tensors = _dense_fp8_case(kind)
                tensors["v"] = _FakeTensor(
                    (1, 32, 2, 64),
                    stride=(32 * 136, 136, 64, 1),
                    dtype="torch.float8_e4m3fn",
                )
                _rdna_validate_and_collect(request, spec, tensors)
                tensors["v"]._stride = (32 * 132, 132, 64, 1)
                with self.assertRaisesRegex(ValueError, r"v stride\(1\)=132"):
                    _rdna_validate_and_collect(request, spec, tensors)

    def test_misaligned_data_pointer_is_rejected(self):
        for kind in _KINDS:
            with self.subTest(kind=kind):
                request, spec, tensors = _dense_fp8_case(kind, kv_ptr=4)
                with self.assertRaisesRegex(ValueError, "k data pointer"):
                    _rdna_validate_and_collect(request, spec, tensors)
                # 8 bytes is enough for FP8 rows.
                request, spec, tensors = _dense_fp8_case(kind, kv_ptr=8)
                _rdna_validate_and_collect(request, spec, tensors)


class TestRdnaSoftmaxScale(unittest.TestCase):
    """The kernels need ``softmax_scale * log2(e) > 0`` (else NaN/zero rows)."""

    _VIEW = SimpleNamespace(softcap=False, kv_fp8=False)
    _BAD = (0.0, -0.125, float("nan"), float("inf"), float("-inf"))

    def test_positive_scale_is_folded_into_scale_log2(self):
        resolve = _rdna_scalars(self._VIEW, 64, {"softmax_scale": 0.125})
        values = {}
        resolve(values, {})
        self.assertAlmostEqual(values["scale_log2"], 0.125 * 1.4426950408889634)
        resolve(values, {"softmax_scale": 0.5})
        self.assertAlmostEqual(values["scale_log2"], 0.5 * 1.4426950408889634)

    def test_bad_scale_is_rejected_at_bind_time(self):
        for scale in self._BAD:
            with self.subTest(scale=scale), self.assertRaisesRegex(
                ValueError, "softmax_scale must be finite and positive"
            ):
                _rdna_scalars(self._VIEW, 64, {"softmax_scale": scale})

    def test_bad_scale_is_rejected_per_launch(self):
        resolve = _rdna_scalars(self._VIEW, 64, {})
        for scale in self._BAD:
            with self.subTest(scale=scale), self.assertRaisesRegex(
                ValueError, "softmax_scale must be finite and positive"
            ):
                resolve({}, {"softmax_scale": scale})


if __name__ == "__main__":
    unittest.main()
