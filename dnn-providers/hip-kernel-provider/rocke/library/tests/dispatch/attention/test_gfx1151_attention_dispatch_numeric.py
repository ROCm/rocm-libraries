# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Actual-surface regressions for the gfx1151 WMMA runtime binder.

Unlike ``library/tests/test_wmma_fmha_fwd_numeric.py`` (which launches the
built kernel directly), the GPU tests here go through the *public dispatch
surface* -- ``dispatch_attention(req).bind_torch(tensors, **kwargs).launch()``
-- so an ABI-order mismatch, a launcher-cache bug, a dropped stream, or a
tensor-layout translation error in ``bind_gfx1151_attention_torch`` fails
here even when the kernel body itself is correct.

Tensors are lightweight duck-typed views (``shape``/``dtype``/``stride``/
``data_ptr``/``device``) over real HIP buffers allocated by the frozen
benchmark harness's ``DeviceBuffers`` -- deliberately not ``torch``, since
``bindings.py`` never imports it and this is the proof that holds.

The trailing ``TestMetadataSafety`` class needs no GPU: it calls
``bind_gfx1151_attention_torch`` directly with plain ``SimpleNamespace``
tensor doubles to pin the structural (shape/dtype/stride) launch gate
without ever reading ``cu_seqlens``/``seqused_k``/``block_table`` contents.
"""

from __future__ import annotations

import ctypes
import dataclasses
import unittest
from typing import Optional, Tuple
from types import SimpleNamespace

import numpy as np
import pytest

from dispatch.attention import AttentionMaskType, AttentionRequest, dispatch_attention
from rocke.runtime import hip_module
from rocke.runtime.hip_module import Runtime, get_device_arch

_NEEDS_GPU = pytest.mark.skipif(get_device_arch() != "gfx1151", reason="needs a gfx1151 GPU")


# ---------------------------------------------------------------------
# Duck-typed device tensor (never torch) backed by a real HIP pointer.
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Device:
    type: str = "cuda"
    index: int = 0


@dataclasses.dataclass(frozen=True)
class _DeviceTensor:
    """Satisfies ``bind_gfx1151_attention_torch``'s tensor protocol
    (shape/dtype/stride/data_ptr/device/is_contiguous) over a numpy-array-
    shaped view of a ``DeviceBuffers``-owned device pointer."""

    _ptr: int
    _shape: Tuple[int, ...]
    _strides: Tuple[int, ...]  # element strides
    _dtype: str

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def device(self) -> _Device:
        return _Device()

    def stride(self, dim: Optional[int] = None):
        return self._strides if dim is None else self._strides[dim]

    def data_ptr(self) -> int:
        return self._ptr

    def is_contiguous(self) -> bool:
        return self._strides[-1] == 1

    def element_size(self) -> int:
        return np.dtype(self._dtype).itemsize

    def numel(self) -> int:
        n = 1
        for s in self._shape:
            n *= s
        return n


def _view(buffers, name: str) -> _DeviceTensor:
    array = buffers.arrays[name]
    return _DeviceTensor(
        _ptr=buffers.ptrs[name],
        _shape=tuple(array.shape),
        _strides=tuple(s // array.itemsize for s in array.strides),
        _dtype=str(array.dtype),
    )


def _tensors_for(buffers, case) -> dict:
    tensors = {"q": _view(buffers, "q"), "k": _view(buffers, "k"), "v": _view(buffers, "v")}
    tensors["out"] = _view(buffers, "rocke_out")
    for name in ("cu_seqlens_q", "cu_seqlens_k", "seqused_k", "block_table", "sinks", "alibi_slopes", "qq_bias"):
        if name in buffers.ptrs:
            tensors[name] = _view(buffers, name)
    return tensors


# ---------------------------------------------------------------------
# Non-default HIP stream (mirrors HipGraphs' minimal ctypes pattern in
# benchmarks/gfx1151/attention/benchmark_sdpa.py -- not a benchmark edit,
# just the same small stream-lifecycle idiom used there).
# ---------------------------------------------------------------------


def _create_stream() -> int:
    lib = hip_module._resolve_hip()
    fn = lib.hipStreamCreateWithFlags
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    fn.restype = ctypes.c_int
    stream = ctypes.c_void_p()
    status = fn(ctypes.byref(stream), 1)  # hipStreamNonBlocking
    if status:
        raise RuntimeError(f"hipStreamCreateWithFlags: HIP error {status}")
    return int(stream.value)


def _destroy_stream(stream: int) -> None:
    lib = hip_module._resolve_hip()
    fn = lib.hipStreamDestroy
    fn.argtypes = [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    fn(ctypes.c_void_p(stream))


def _mask_ordinal(mask: str) -> AttentionMaskType:
    return {
        "none": AttentionMaskType.NO_MASK,
        "causal_topleft": AttentionMaskType.TOP_LEFT_CAUSAL,
        "causal_bottomright": AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
    }[mask]


def _request_for(case) -> AttentionRequest:
    seqlen_q = max(case.q_lengths) if case.q_lengths else case.seqlen_q
    seqlen_k = max(case.k_lengths) if case.k_lengths else case.seqlen_k
    return AttentionRequest(
        batch=case.batch,
        nhead_q=case.heads_q,
        nhead_k=case.heads_kv,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        hdim_q=case.head_dim,
        hdim_v=case.head_dim,
        arch="gfx1151",
        dtype=case.dtype,
        mask_type=_mask_ordinal(case.mask),
        use_sinks=case.sinks,
        sliding_window=case.window,
        kv_block_size=case.block_size if case.layout == "paged" else 16,
        use_softcap=bool(case.softcap),
        use_alibi=case.alibi,
        use_qq_bias=case.qq_bias,
        use_fp8=bool(case.kv_dtype),
        layout=case.layout,
    )


def _launch_and_check(case, *, stream: int = 0, fp8_scales=None):
    from benchmarks.gfx1151.attention.benchmark_sdpa import DeviceBuffers
    from benchmarks.gfx1151.attention.cases import make_inputs, reference
    from rocke.runtime.launcher import release_retained_for_stream
    from rocke.runtime.torch_interop import resolve_stream

    inputs = make_inputs(case)
    if fp8_scales is not None:
        inputs.k_scale = np.asarray(fp8_scales[0], np.float32)
        inputs.v_scale = np.asarray(fp8_scales[1], np.float32)
    stream = resolve_stream(stream)
    rt = Runtime()
    buffers = DeviceBuffers(rt, inputs)
    try:
        req = _request_for(case)
        result = dispatch_attention(req)
        kwargs = {"softmax_scale": case.scale, "stream": stream, "fence": False}
        if case.softcap:
            kwargs["softcap"] = case.softcap
        if case.kv_dtype:
            kwargs["k_scale"] = float(inputs.k_scale)
            kwargs["v_scale"] = float(inputs.v_scale)
        binding = result.bind_torch(_tensors_for(buffers, case), **kwargs)
        binding.launch()
        rt.stream_sync(stream)
        release_retained_for_stream(stream)
        actual = buffers.read_output("rocke_out")
        expected = reference(case, inputs)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=case.atol, equal_nan=False)
        return actual
    finally:
        rt.sync()
        release_retained_for_stream(stream)
        buffers.close()


def _case(**kw):
    from benchmarks.gfx1151.attention.cases import SdpaCase

    return SdpaCase(**kw)


@pytest.mark.gpu
@_NEEDS_GPU
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_dispatch_dense_partial_tail_nondefault_stream(dtype):
    """Dense request with a non-multiple-of-16 Q/K tail, launched on an
    explicit non-default HIP stream through the public dispatch surface."""
    case = _case(
        name="dispatch_dense_tail", group="dispatch", dtype=dtype, batch=2,
        seqlen_q=17, seqlen_k=19, heads_q=4, heads_kv=2, head_dim=64,
        mask="causal_bottomright", seed=101,
    )
    stream = _create_stream()
    try:
        _launch_and_check(case, stream=stream)
    finally:
        _destroy_stream(stream)


@pytest.mark.gpu
@_NEEDS_GPU
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_dispatch_ragged_empty_and_mixed_lengths(dtype):
    """Ragged batch with an empty sequence (no queries, but real keys) and
    unequal per-sequence Q/K lengths."""
    case = _case(
        name="dispatch_ragged_mixed", group="dispatch", dtype=dtype, batch=5,
        seqlen_q=0, seqlen_k=0, heads_q=4, heads_kv=2, head_dim=64,
        layout="ragged", mask="causal_bottomright",
        q_lengths=(0, 17, 1, 3, 0), k_lengths=(9, 17, 5, 2, 0), seed=27,
    )
    _launch_and_check(case)


@pytest.mark.gpu
@_NEEDS_GPU
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_dispatch_paged_shuffled_partial_page(dtype):
    """Paged KV with a shuffled (non-identity) block table and a partial
    last page, through the public dispatch surface."""
    case = _case(
        name="dispatch_paged_shuffled", group="dispatch", dtype=dtype, batch=5,
        seqlen_q=0, seqlen_k=0, heads_q=4, heads_kv=2, head_dim=64,
        layout="paged", block_size=16, mask="causal_bottomright",
        q_lengths=(0, 17, 1, 3, 0), k_lengths=(9, 19, 1, 0, 0), seed=27,
    )
    _launch_and_check(case)


@pytest.mark.gpu
@_NEEDS_GPU
def test_dispatch_dense_fp8_kv():
    """OCP fp8e4m3 KV storage with explicit non-power-of-two dequant scales,
    through the public dispatch surface (``use_fp8=True`` -> ``kv_dtype``)."""
    pytest.importorskip("ml_dtypes")
    case = _case(
        name="dispatch_dense_fp8_kv", group="dispatch", dtype="fp16", batch=1,
        seqlen_q=32, seqlen_k=32, heads_q=2, heads_kv=2, head_dim=64,
        mask="causal_topleft", kv_dtype="fp8e4m3", seed=7,
    )
    _launch_and_check(case, fp8_scales=(0.3, 0.7))


# ---------------------------------------------------------------------
# Structural launch-safety gate: no GPU, no torch, no ``cu_seqlens``/
# ``seqused_k``/``block_table`` content reads -- only shape/dtype/stride.
# ---------------------------------------------------------------------


def _fake_tensor(shape, dtype="float16", strides=None, device="cuda:0"):
    if strides is None:
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        strides = tuple(strides)

    def stride(dim=None):
        return strides if dim is None else strides[dim]

    return SimpleNamespace(
        shape=tuple(shape), dtype=dtype, device=SimpleNamespace(type=device.split(":")[0]),
        stride=stride, is_contiguous=lambda: strides[-1] == 1, data_ptr=lambda: 0x1000,
    )


def _dense_spec(**kw):
    from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec

    base = dict(head_size=64, num_query_heads=4, num_kv_heads=2, dtype="fp16")
    base.update(kw)
    return WmmaFmhaFwdSpec(**base)


def _dense_request(**kw):
    base = dict(
        batch=2, nhead_q=4, nhead_k=2, seqlen_q=32, seqlen_k=32,
        hdim_q=64, hdim_v=64, arch="gfx1151", dtype="fp16",
    )
    base.update(kw)
    return AttentionRequest(**base)


class TestMetadataSafety(unittest.TestCase):
    """Direct ``bind_gfx1151_attention_torch`` calls -- no GPU, no torch, no
    compile: every case here raises before the binder ever reaches
    ``_gfx1151_launcher`` (build/compile/cache)."""

    def _bind(self, tensors, **kwargs):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        return bind_gfx1151_attention_torch(
            _dense_request(), _dense_spec(), tensors, **kwargs
        )

    def test_rejects_dense_batch_stride_gap(self):
        """A [B, S, H, D] tensor whose batch stride skips rows (e.g. a
        slice of a larger allocation) must be rejected: the dense kernel
        ABI folds the batch offset into ``batch * seqlen * stride_token``
        with no separate batch-stride argument."""
        q = _fake_tensor((2, 32, 4, 64))
        gapped_k = _fake_tensor((2, 32, 2, 64), strides=(32 * 2 * 64 + 64, 2 * 64, 64, 1))
        with self.assertRaisesRegex(ValueError, "batch stride"):
            self._bind({"q": q, "k": gapped_k, "v": gapped_k, "out": q})

    def test_rejects_wrong_kv_dtype_for_fp8_spec(self):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        q = _fake_tensor((2, 32, 4, 64), dtype="float16")
        k = _fake_tensor((2, 32, 2, 64), dtype="float16")  # should be fp8e4m3
        spec = _dense_spec(kv_dtype="fp8e4m3")
        with self.assertRaisesRegex(ValueError, "dtype must be fp8"):
            bind_gfx1151_attention_torch(
                _dense_request(use_fp8=True), spec, {"q": q, "k": k, "v": k, "out": q},
                k_scale=0.5, v_scale=0.25,
            )

    def test_rejects_fnuz_fp8_kv(self):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        q = _fake_tensor((2, 32, 4, 64), dtype="float16")
        k = _fake_tensor((2, 32, 2, 64), dtype="float8_e4m3fnuz")
        spec = _dense_spec(kv_dtype="fp8e4m3")
        with self.assertRaisesRegex(ValueError, "not FNUZ"):
            bind_gfx1151_attention_torch(
                _dense_request(use_fp8=True), spec, {"q": q, "k": k, "v": k, "out": q},
                k_scale=0.5, v_scale=0.25,
            )

    def test_requires_k_scale_v_scale_for_fp8_kv(self):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        q = _fake_tensor((2, 32, 4, 64), dtype="float16")
        k = _fake_tensor((2, 32, 2, 64), dtype="float8_e4m3fn")
        spec = _dense_spec(kv_dtype="fp8e4m3")
        with self.assertRaisesRegex(ValueError, "k_scale.*v_scale"):
            bind_gfx1151_attention_torch(
                _dense_request(use_fp8=True), spec, {"q": q, "k": k, "v": k, "out": q},
            )

    def test_ragged_requires_cu_seqlens_k(self):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        q = _fake_tensor((10, 4, 64))
        k = _fake_tensor((12, 2, 64))
        cu_q = _fake_tensor((3,), dtype="int32")
        spec = _dense_spec(layout="ragged", query_tail=True, kv_tail=True)
        with self.assertRaisesRegex(ValueError, "cu_seqlens_k"):
            bind_gfx1151_attention_torch(
                _dense_request(batch=2, layout="ragged"), spec,
                {"q": q, "k": k, "v": k, "out": q, "cu_seqlens_q": cu_q},
            )

    def test_paged_rejects_kv_cache_page_size_mismatch(self):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        q = _fake_tensor((10, 4, 64))
        k = _fake_tensor((3, 8, 2, 64))  # page size 8, spec expects 16
        cu_q = _fake_tensor((3,), dtype="int32")
        used = _fake_tensor((2,), dtype="int32")
        table = _fake_tensor((2, 4), dtype="int32")
        spec = _dense_spec(layout="paged", page_block_size=16, query_tail=True, kv_tail=True)
        with self.assertRaisesRegex(ValueError, "page"):
            bind_gfx1151_attention_torch(
                _dense_request(batch=2, layout="paged"), spec,
                {
                    "q": q, "k": k, "v": k, "out": q, "cu_seqlens_q": cu_q,
                    "seqused_k": used, "block_table": table,
                },
            )

    def test_rejects_undersized_packed_output_and_values(self):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        q = _fake_tensor((10, 4, 64))
        k = _fake_tensor((12, 2, 64))
        cu = _fake_tensor((3,), dtype="int32")
        tensors = {"q": q, "k": k, "v": k, "out": q, "cu_seqlens_q": cu, "cu_seqlens_k": cu}
        for name, value in (("out", _fake_tensor((9, 4, 64))), ("v", _fake_tensor((11, 2, 64)))):
            with self.subTest(tensor=name), self.assertRaises(ValueError):
                bind_gfx1151_attention_torch(
                    _dense_request(layout="ragged"), _dense_spec(layout="ragged"),
                    dict(tensors, **{name: value}),
                )

    def test_rejects_strided_sequence_metadata(self):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        q = _fake_tensor((10, 4, 64))
        k = _fake_tensor((12, 2, 64))
        cu = _fake_tensor((3,), dtype="int32")
        tensors = {"q": q, "k": k, "v": k, "out": q, "cu_seqlens_q": cu, "cu_seqlens_k": cu}
        for name in ("cu_seqlens_q", "cu_seqlens_k"):
            with self.subTest(tensor=name), self.assertRaises(ValueError):
                bind_gfx1151_attention_torch(
                    _dense_request(layout="ragged"), _dense_spec(layout="ragged"),
                    dict(tensors, **{name: _fake_tensor((3,), dtype="int32", strides=(2,))}),
                )

    def test_rejects_rows_that_break_vector_alignment(self):
        q = _fake_tensor((2, 32, 4, 64), strides=(32 * 4 * 65, 4 * 65, 65, 1))
        k = _fake_tensor((2, 32, 2, 64))
        with self.assertRaises(ValueError):
            self._bind({"q": q, "k": k, "v": k, "out": _fake_tensor((2, 32, 4, 64))})

    def test_transposed_qk_rejects_partial_kv_tiles_before_launch(self):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        q = _fake_tensor((2, 32, 4, 64))
        k = _fake_tensor((2, 48, 2, 64))
        with self.assertRaises(ValueError):
            bind_gfx1151_attention_torch(
                _dense_request(seqlen_k=48), _dense_spec(transposed_qk=True, block_n=64),
                {"q": q, "k": k, "v": k, "out": q},
            )


@pytest.mark.gpu
@_NEEDS_GPU
def test_dispatch_aligned_gqa_on_nondefault_stream():
    case = _case(
        name="dispatch_aligned_gqa", group="dispatch", dtype="fp16", batch=2,
        seqlen_q=32, seqlen_k=96, heads_q=6, heads_kv=2, head_dim=128,
        mask="causal_topleft", seed=103,
    )
    stream = _create_stream()
    try:
        _launch_and_check(case, stream=stream)
    finally:
        _destroy_stream(stream)


@pytest.mark.gpu
@_NEEDS_GPU
@pytest.mark.parametrize(
    "mask,q_lengths,k_lengths",
    [
        ("causal_topleft", (32, 128), (64, 32)),
        ("causal_bottomright", (64, 96), (128, 32)),
    ],
)
def test_ragged_full_window_preserves_all_causal_keys(mask, q_lengths, k_lengths):
    case = _case(
        name="dispatch_ragged_full_window", group="dispatch", dtype="fp16", batch=2,
        seqlen_q=0, seqlen_k=0, heads_q=8, heads_kv=2, head_dim=64,
        layout="ragged", mask=mask, q_lengths=q_lengths, k_lengths=k_lengths, seed=104,
    )
    actual = _launch_and_check(case)
    if mask == "causal_bottomright":
        begin = q_lengths[0]
        end = begin + q_lengths[1] - k_lengths[1]
        np.testing.assert_array_equal(actual[begin:end], np.zeros_like(actual[begin:end]))
