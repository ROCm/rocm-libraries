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

_NEEDS_GPU = pytest.mark.skipif(
    get_device_arch() != "gfx1151", reason="needs a gfx1151 GPU"
)


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

    def __getitem__(self, key: slice) -> "_DeviceTensor":
        """Leading-dim slice view, as torch tensors provide for per-batch launches."""
        start, stop, step = key.indices(self._shape[0])
        assert step == 1 and stop >= start
        return _DeviceTensor(
            _ptr=self._ptr + start * self._strides[0] * self.element_size(),
            _shape=(stop - start,) + self._shape[1:],
            _strides=self._strides,
            _dtype=self._dtype,
        )


def _view(buffers, name: str) -> _DeviceTensor:
    array = buffers.arrays[name]
    return _DeviceTensor(
        _ptr=buffers.ptrs[name],
        _shape=tuple(array.shape),
        _strides=tuple(s // array.itemsize for s in array.strides),
        _dtype=str(array.dtype),
    )


def _tensors_for(buffers, case) -> dict:
    tensors = {
        "q": _view(buffers, "q"),
        "k": _view(buffers, "k"),
        "v": _view(buffers, "v"),
    }
    tensors["out"] = _view(buffers, "rocke_out")
    for name in (
        "cu_seqlens_q",
        "cu_seqlens_k",
        "seqused_k",
        "block_table",
        "sinks",
        "alibi_slopes",
        "qq_bias",
    ):
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


def _launch_and_check(
    case, *, stream: int = 0, fp8_scales=None, inputs=None, spec_overrides=None
):
    from benchmarks.gfx1151.attention.benchmark_sdpa import DeviceBuffers
    from benchmarks.gfx1151.attention.cases import make_inputs, reference
    from rocke.runtime.launcher import release_retained_for_stream
    from rocke.runtime.torch_interop import resolve_stream

    if inputs is None:
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
        if spec_overrides is None:
            binding = result.bind_torch(_tensors_for(buffers, case), **kwargs)
        else:
            from dispatch.attention.bindings import bind_gfx1151_attention_torch

            spec = dataclasses.replace(result.spec, **spec_overrides)
            binding = bind_gfx1151_attention_torch(
                req, spec, _tensors_for(buffers, case), **kwargs
            )
        binding.launch()
        rt.stream_sync(stream)
        release_retained_for_stream(stream)
        actual = buffers.read_output("rocke_out")
        expected = reference(case, inputs)
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=case.atol, equal_nan=False
        )
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
        name="dispatch_dense_tail",
        group="dispatch",
        dtype=dtype,
        batch=2,
        seqlen_q=17,
        seqlen_k=19,
        heads_q=4,
        heads_kv=2,
        head_dim=64,
        mask="causal_bottomright",
        seed=101,
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
        name="dispatch_ragged_mixed",
        group="dispatch",
        dtype=dtype,
        batch=5,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=4,
        heads_kv=2,
        head_dim=64,
        layout="ragged",
        mask="causal_bottomright",
        q_lengths=(0, 17, 1, 3, 0),
        k_lengths=(9, 17, 5, 2, 0),
        seed=27,
    )
    _launch_and_check(case)


@pytest.mark.gpu
@_NEEDS_GPU
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_dispatch_paged_shuffled_partial_page(dtype):
    """Paged KV with a shuffled (non-identity) block table and a partial
    last page, through the public dispatch surface."""
    case = _case(
        name="dispatch_paged_shuffled",
        group="dispatch",
        dtype=dtype,
        batch=5,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=4,
        heads_kv=2,
        head_dim=64,
        layout="paged",
        block_size=16,
        mask="causal_bottomright",
        q_lengths=(0, 17, 1, 3, 0),
        k_lengths=(9, 19, 1, 0, 0),
        seed=27,
    )
    _launch_and_check(case)


@pytest.mark.gpu
@_NEEDS_GPU
def test_dispatch_dense_fp8_kv():
    """OCP fp8e4m3 KV storage with explicit non-power-of-two dequant scales,
    through the public dispatch surface (``use_fp8=True`` -> ``kv_dtype``)."""
    pytest.importorskip("ml_dtypes")
    case = _case(
        name="dispatch_dense_fp8_kv",
        group="dispatch",
        dtype="fp16",
        batch=1,
        seqlen_q=32,
        seqlen_k=32,
        heads_q=2,
        heads_kv=2,
        head_dim=64,
        mask="causal_topleft",
        kv_dtype="fp8e4m3",
        seed=7,
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
        shape=tuple(shape),
        dtype=dtype,
        device=SimpleNamespace(type=device.split(":")[0]),
        stride=stride,
        is_contiguous=lambda: strides[-1] == 1,
        data_ptr=lambda: 0x1000,
    )


def _dense_spec(**kw):
    from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec

    base = dict(head_size=64, num_query_heads=4, num_kv_heads=2, dtype="fp16")
    base.update(kw)
    return WmmaFmhaFwdSpec(**base)


def _dense_request(**kw):
    base = dict(
        batch=2,
        nhead_q=4,
        nhead_k=2,
        seqlen_q=32,
        seqlen_k=32,
        hdim_q=64,
        hdim_v=64,
        arch="gfx1151",
        dtype="fp16",
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

    def test_misaligned_dense_batch_stride_is_rejected(self):
        """A batch stride that breaks 16-byte alignment of the per-batch
        views is rejected (the dense kernel folds ``batch * seqlen *
        stride_token`` itself; non-folded batches launch per batch)."""
        q = _fake_tensor((2, 32, 4, 64))
        gapped_k = _fake_tensor(
            (2, 32, 2, 64), strides=(32 * 2 * 64 + 1, 2 * 64, 64, 1)
        )
        with self.assertRaisesRegex(ValueError, "batch stride|strides must preserve"):
            self._bind({"q": q, "k": gapped_k, "v": gapped_k, "out": q})

    def test_gapped_dense_batch_stride_is_served_per_batch(self):
        from dispatch.attention.bindings import (
            _gfx1151_dense_needs_per_batch,
            _gfx1151_validate_and_collect,
        )

        q = _fake_tensor((2, 32, 4, 64))
        folded_k = _fake_tensor((2, 32, 2, 64))
        gapped_k = _fake_tensor(
            (2, 32, 2, 64), strides=(32 * 2 * 64 + 64, 2 * 64, 64, 1)
        )
        # [B, H, S, D] storage viewed as [B, S, H, D].
        permuted_k = _fake_tensor(
            (2, 32, 2, 64), strides=(2 * 32 * 64, 64, 32 * 64, 1)
        )
        request, spec = _dense_request(), _dense_spec()
        for k, expected in ((folded_k, False), (gapped_k, True), (permuted_k, True)):
            tensors = {"q": q, "k": k, "v": k, "out": q}
            _gfx1151_validate_and_collect(request, spec, tensors)
            self.assertEqual(_gfx1151_dense_needs_per_batch(request, tensors), expected)

    def test_lse_tensor_is_required_and_validated(self):
        from dispatch.attention.bindings import _gfx1151_validate_and_collect

        request = _dense_request(return_lse=True)
        spec = _dense_spec(store_lse=True)
        q = _fake_tensor((2, 32, 4, 64))
        k = _fake_tensor((2, 32, 2, 64))
        base = {"q": q, "k": k, "v": k, "out": q}
        with self.assertRaisesRegex(ValueError, r"requires tensors\['lse'\]"):
            _gfx1151_validate_and_collect(request, spec, base)
        good = _fake_tensor((2, 4, 32), dtype="float32")
        values = _gfx1151_validate_and_collect(request, spec, {**base, "lse": good})
        self.assertIs(values["lse"], good)
        self.assertNotIn("stride_lse_head", values)
        bad = (
            _fake_tensor((2, 4, 32), dtype="float16"),
            _fake_tensor((2, 32, 4), dtype="float32"),
            _fake_tensor((2, 4, 32), dtype="float32", strides=(4 * 32 + 8, 32, 1)),
        )
        for lse in bad:
            with (
                self.subTest(shape=lse.shape, dtype=lse.dtype),
                self.assertRaises(ValueError),
            ):
                _gfx1151_validate_and_collect(request, spec, {**base, "lse": lse})

    def test_overlapping_dense_out_batch_stride_is_rejected(self):
        q = _fake_tensor((2, 32, 4, 64))
        k = _fake_tensor((2, 32, 2, 64))
        overlapping = _fake_tensor(
            (2, 32, 4, 64), strides=(16 * 4 * 64, 4 * 64, 64, 1)
        )
        with self.assertRaisesRegex(ValueError, "out batch stride"):
            self._bind({"q": q, "k": k, "v": k, "out": overlapping})

    def test_unequal_head_dims_follow_v_dim(self):
        from dispatch.attention.bindings import _gfx1151_validate_and_collect

        request = _dense_request(hdim_q=64, hdim_v=128)
        spec = _dense_spec(v_head_size=128)
        q = _fake_tensor((2, 32, 4, 64))
        k = _fake_tensor((2, 32, 2, 64))
        v = _fake_tensor((2, 32, 2, 128))
        out = _fake_tensor((2, 32, 4, 128))
        values = _gfx1151_validate_and_collect(
            request, spec, {"q": q, "k": k, "v": v, "out": out}
        )
        self.assertEqual(values["stride_o_token"], 4 * 128)
        self.assertEqual(values["stride_v_token"], 2 * 128)
        with self.assertRaisesRegex(ValueError, "out trailing dims"):
            _gfx1151_validate_and_collect(
                request, spec, {"q": q, "k": k, "v": v, "out": q}
            )
        with self.assertRaisesRegex(ValueError, "v trailing dims"):
            _gfx1151_validate_and_collect(
                request, spec, {"q": q, "k": k, "v": k, "out": out}
            )

    def test_rejects_wrong_kv_dtype_for_fp8_spec(self):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        q = _fake_tensor((2, 32, 4, 64), dtype="float16")
        k = _fake_tensor((2, 32, 2, 64), dtype="float16")  # should be fp8e4m3
        spec = _dense_spec(kv_dtype="fp8e4m3")
        with self.assertRaisesRegex(ValueError, "dtype must be fp8"):
            bind_gfx1151_attention_torch(
                _dense_request(use_fp8=True),
                spec,
                {"q": q, "k": k, "v": k, "out": q},
                k_scale=0.5,
                v_scale=0.25,
            )

    def test_rejects_fnuz_fp8_kv(self):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        q = _fake_tensor((2, 32, 4, 64), dtype="float16")
        k = _fake_tensor((2, 32, 2, 64), dtype="float8_e4m3fnuz")
        spec = _dense_spec(kv_dtype="fp8e4m3")
        with self.assertRaisesRegex(ValueError, "not FNUZ"):
            bind_gfx1151_attention_torch(
                _dense_request(use_fp8=True),
                spec,
                {"q": q, "k": k, "v": k, "out": q},
                k_scale=0.5,
                v_scale=0.25,
            )

    def test_requires_k_scale_v_scale_for_fp8_kv(self):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        q = _fake_tensor((2, 32, 4, 64), dtype="float16")
        k = _fake_tensor((2, 32, 2, 64), dtype="float8_e4m3fn")
        spec = _dense_spec(kv_dtype="fp8e4m3")
        with self.assertRaisesRegex(ValueError, "k_scale.*v_scale"):
            bind_gfx1151_attention_torch(
                _dense_request(use_fp8=True),
                spec,
                {"q": q, "k": k, "v": k, "out": q},
            )

    def test_ragged_requires_cu_seqlens_k(self):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        q = _fake_tensor((10, 4, 64))
        k = _fake_tensor((12, 2, 64))
        cu_q = _fake_tensor((3,), dtype="int32")
        spec = _dense_spec(layout="ragged", query_tail=True, kv_tail=True)
        with self.assertRaisesRegex(ValueError, "cu_seqlens_k"):
            bind_gfx1151_attention_torch(
                _dense_request(batch=2, layout="ragged"),
                spec,
                {"q": q, "k": k, "v": k, "out": q, "cu_seqlens_q": cu_q},
            )

    def test_paged_rejects_kv_cache_page_size_mismatch(self):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        q = _fake_tensor((10, 4, 64))
        k = _fake_tensor((3, 8, 2, 64))  # page size 8, spec expects 16
        cu_q = _fake_tensor((3,), dtype="int32")
        used = _fake_tensor((2,), dtype="int32")
        table = _fake_tensor((2, 4), dtype="int32")
        spec = _dense_spec(
            layout="paged", page_block_size=16, query_tail=True, kv_tail=True
        )
        with self.assertRaisesRegex(ValueError, "page"):
            bind_gfx1151_attention_torch(
                _dense_request(batch=2, layout="paged"),
                spec,
                {
                    "q": q,
                    "k": k,
                    "v": k,
                    "out": q,
                    "cu_seqlens_q": cu_q,
                    "seqused_k": used,
                    "block_table": table,
                },
            )

    def test_rejects_undersized_packed_output_and_values(self):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        q = _fake_tensor((10, 4, 64))
        k = _fake_tensor((12, 2, 64))
        cu = _fake_tensor((3,), dtype="int32")
        tensors = {
            "q": q,
            "k": k,
            "v": k,
            "out": q,
            "cu_seqlens_q": cu,
            "cu_seqlens_k": cu,
        }
        for name, value in (
            ("out", _fake_tensor((9, 4, 64))),
            ("v", _fake_tensor((11, 2, 64))),
        ):
            with self.subTest(tensor=name), self.assertRaises(ValueError):
                bind_gfx1151_attention_torch(
                    _dense_request(layout="ragged"),
                    _dense_spec(layout="ragged"),
                    dict(tensors, **{name: value}),
                )

    def test_rejects_strided_sequence_metadata(self):
        from dispatch.attention.bindings import bind_gfx1151_attention_torch

        q = _fake_tensor((10, 4, 64))
        k = _fake_tensor((12, 2, 64))
        cu = _fake_tensor((3,), dtype="int32")
        tensors = {
            "q": q,
            "k": k,
            "v": k,
            "out": q,
            "cu_seqlens_q": cu,
            "cu_seqlens_k": cu,
        }
        for name in ("cu_seqlens_q", "cu_seqlens_k"):
            with self.subTest(tensor=name), self.assertRaises(ValueError):
                bind_gfx1151_attention_torch(
                    _dense_request(layout="ragged"),
                    _dense_spec(layout="ragged"),
                    dict(
                        tensors,
                        **{name: _fake_tensor((3,), dtype="int32", strides=(2,))},
                    ),
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
                _dense_request(seqlen_k=48),
                _dense_spec(transposed_qk=True, block_n=64),
                {"q": q, "k": k, "v": k, "out": q},
            )


@pytest.mark.gpu
@_NEEDS_GPU
def test_dispatch_aligned_gqa_on_nondefault_stream():
    case = _case(
        name="dispatch_aligned_gqa",
        group="dispatch",
        dtype="fp16",
        batch=2,
        seqlen_q=32,
        seqlen_k=96,
        heads_q=6,
        heads_kv=2,
        head_dim=128,
        mask="causal_topleft",
        seed=103,
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
        name="dispatch_ragged_full_window",
        group="dispatch",
        dtype="fp16",
        batch=2,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=8,
        heads_kv=2,
        head_dim=64,
        layout="ragged",
        mask=mask,
        q_lengths=q_lengths,
        k_lengths=k_lengths,
        seed=104,
    )
    actual = _launch_and_check(case)
    if mask == "causal_bottomright":
        begin = q_lengths[0]
        end = begin + q_lengths[1] - k_lengths[1]
        np.testing.assert_array_equal(
            actual[begin:end], np.zeros_like(actual[begin:end])
        )


@pytest.mark.gpu
@_NEEDS_GPU
@pytest.mark.parametrize(
    "layout,dtype,storage,tile,vlds",
    [
        ("dense", "fp16", "", 128, False),
        ("dense", "bf16", "", 64, True),
        ("ragged", "fp16", "fp8e4m3", 32, True),
        ("paged", "bf16", "fp8e4m3", 16, False),
    ],
)
def test_output_tiles_keep_full_qk_dimension_and_all_output_columns(
    layout, dtype, storage, tile, vlds
):
    from benchmarks.gfx1151.attention.cases import make_inputs

    lengths = (
        dict(
            batch=3, seqlen_q=0, seqlen_k=0, q_lengths=(0, 3, 17), k_lengths=(0, 0, 31)
        )
        if layout == "ragged"
        else (
            dict(
                batch=2,
                seqlen_q=0,
                seqlen_k=0,
                q_lengths=(1, 3),
                k_lengths=(17, 0),
                block_size=16,
            )
            if layout == "paged"
            else dict(
                batch=2,
                seqlen_q=1 if dtype == "fp16" else 17,
                seqlen_k=32 if dtype == "fp16" else 19,
            )
        )
    )
    case = _case(
        name="dispatch_output_tiles",
        group="dispatch",
        dtype=dtype,
        heads_q=4,
        heads_kv=2,
        head_dim=256,
        layout=layout,
        kv_dtype=storage,
        mask="causal_bottomright",
        seed=105,
        **lengths,
    )
    inputs = make_inputs(case)
    # The last QK dimension affects even the first output tile. Truncating QK
    # to the output width would make these logits uniform and fail the oracle.
    inputs.q.fill(0)
    inputs.q[..., -1] = 1
    signs = np.where(np.indices(inputs.k.shape[:-1]).sum(axis=0) % 2, 1.0, -1.0)
    inputs.k.fill(0)
    inputs.k[..., -1] = signs * 8
    inputs.v[...] = signs[..., None] + np.arange(256, dtype=np.float32) / 512
    actual = _launch_and_check(
        case,
        inputs=inputs,
        spec_overrides=dict(value_tile_size=tile, v_lds_stage=vlds),
    )
    if layout == "ragged":
        np.testing.assert_array_equal(actual[:3], np.zeros_like(actual[:3]))
    elif layout == "paged":
        np.testing.assert_array_equal(actual[1:], np.zeros_like(actual[1:]))


@pytest.mark.gpu
@_NEEDS_GPU
@pytest.mark.parametrize(
    "mask,seqlen_q", [("causal_topleft", 97), ("causal_bottomright", 65)]
)
def test_dense_tail_profile_preserves_long_queries_and_empty_prefix(mask, seqlen_q):
    case = _case(
        name="dispatch_dense_tail_profile",
        group="dispatch",
        dtype="fp16",
        batch=2,
        seqlen_q=seqlen_q,
        seqlen_k=33,
        heads_q=4,
        heads_kv=2,
        head_dim=64,
        mask=mask,
        seed=106,
    )
    actual = _launch_and_check(case)
    if mask == "causal_bottomright":
        prefix = seqlen_q - case.seqlen_k
        np.testing.assert_array_equal(
            actual[:, :prefix], np.zeros_like(actual[:, :prefix])
        )


# ---------------------------------------------------------------------
# Self-contained numerics for features the frozen benchmark inputs cannot
# express: a right window, unequal Q/V head dims, and a gapped batch stride.
# ---------------------------------------------------------------------


def _run_dense_direct(request, q, k, v, *, scale, kv_store=None):
    """Launch one dense request through the public binder and return the fp32
    output of shape ``[B, Sq, Hq, Dv]``.

    ``kv_store`` optionally holds larger ``(k, v)`` allocations whose leading
    ``[:, :Sk]`` slice is the logical K/V, so the batch stride has a gap."""
    from benchmarks.gfx1151.attention.benchmark_sdpa import _host_bytes
    from rocke.runtime.launcher import release_retained_for_stream
    from rocke.runtime.torch_interop import resolve_stream

    stream = resolve_stream(0)
    rt = Runtime()
    out = np.full(q.shape[:3] + (v.shape[-1],), np.nan, dtype=q.dtype)
    k_mem, v_mem = kv_store if kv_store is not None else (k, v)
    pointers = []
    try:

        def view(logical, backing):
            ptr = rt.alloc(backing.nbytes)
            pointers.append(ptr)
            rt.memcpy_h2d(ptr, _host_bytes(backing), backing.nbytes)
            return _DeviceTensor(
                _ptr=ptr,
                _shape=tuple(logical.shape),
                _strides=tuple(s // backing.itemsize for s in backing.strides),
                _dtype=str(backing.dtype),
            )

        tensors = {
            "q": view(q, q),
            "k": view(k, k_mem),
            "v": view(v, v_mem),
            "out": view(out, out),
        }
        binding = dispatch_attention(request).bind_torch(
            tensors, softmax_scale=scale, stream=stream, fence=False
        )
        binding.launch()
        rt.stream_sync(stream)
        release_retained_for_stream(stream)
        rt.memcpy_d2h(_host_bytes(out), pointers[-1], out.nbytes)
        return out.astype(np.float32)
    finally:
        rt.sync()
        release_retained_for_stream(stream)
        for ptr in pointers:
            rt.free(ptr)


def _windowed_reference(q, k, v, *, scale, ctx, left, right):
    """FP64 attention keeping ``k <= q + ctx + right`` (when ``right >= 0``) and
    ``k > q + ctx - left`` (when ``left > 0``); ``ctx`` is the alignment offset."""
    _, sq, hq, _ = q.shape
    sk, hkv = k.shape[1], k.shape[2]
    q64, k64, v64 = (x.astype(np.float64) for x in (q, k, v))
    k64 = np.repeat(k64, hq // hkv, axis=2)
    v64 = np.repeat(v64, hq // hkv, axis=2)
    qi = np.arange(sq).reshape(sq, 1)
    ki = np.arange(sk).reshape(1, sk)
    keep = np.ones((sq, sk), dtype=bool)
    if right >= 0:
        keep &= ki <= qi + ctx + right
    if left > 0:
        keep &= ki > qi + ctx - left
    scores = np.einsum("bqhd,bkhd->bhqk", q64 * scale, k64)
    scores = np.where(keep.reshape(1, 1, sq, sk), scores, -np.inf)
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", p, v64).astype(np.float32)


def _random_qkv(seed, batch, sq, sk, hq, hkv, dq, dv):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((batch, sq, hq, dq)).astype(np.float16),
        rng.standard_normal((batch, sk, hkv, dq)).astype(np.float16),
        rng.standard_normal((batch, sk, hkv, dv)).astype(np.float16),
    )


def _dense_direct_request(batch, sq, sk, hq, hkv, dq, dv, **extra):
    return AttentionRequest(
        batch=batch,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=sq,
        seqlen_k=sk,
        hdim_q=dq,
        hdim_v=dv,
        arch="gfx1151",
        dtype="fp16",
        layout="dense",
        **extra,
    )


@pytest.mark.gpu
@_NEEDS_GPU
@pytest.mark.parametrize(
    "mask,sq,sk,left,right",
    [
        (AttentionMaskType.NO_MASK, 48, 64, 0, 16),
        (AttentionMaskType.SLIDING_WINDOW, 33, 41, 24, 8),
        (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 33, 49, 0, 16),
        (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 40, 40, 20, 0),
    ],
)
def test_dispatch_two_sided_local_window(mask, sq, sk, left, right):
    hq, hkv, dim = 4, 2, 64
    q, k, v = _random_qkv(201, 2, sq, sk, hq, hkv, dim, dim)
    scale = 1.0 / np.sqrt(dim)
    request = _dense_direct_request(
        2, sq, sk, hq, hkv, dim, dim,
        mask_type=mask, sliding_window=left, window_right=right,
    )
    ctx = sk - sq if mask == AttentionMaskType.BOTTOM_RIGHT_CAUSAL else 0
    actual = _run_dense_direct(request, q, k, v, scale=scale)
    expected = _windowed_reference(
        q, k, v, scale=scale, ctx=ctx, left=left, right=right
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-2, equal_nan=False)


@pytest.mark.gpu
@_NEEDS_GPU
@pytest.mark.parametrize("dq,dv", [(64, 128), (128, 64), (80, 48)])
def test_dispatch_unequal_head_dims(dq, dv):
    hq, hkv, sq, sk = 4, 2, 37, 53
    q, k, v = _random_qkv(202, 2, sq, sk, hq, hkv, dq, dv)
    scale = 1.0 / np.sqrt(dq)
    request = _dense_direct_request(2, sq, sk, hq, hkv, dq, dv)
    actual = _run_dense_direct(request, q, k, v, scale=scale)
    expected = _windowed_reference(q, k, v, scale=scale, ctx=0, left=0, right=-1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-2, equal_nan=False)


@pytest.mark.gpu
@_NEEDS_GPU
def test_dispatch_dense_gapped_batch_stride_is_served_per_batch():
    batch, sq, sk, hq, hkv, dim, pad = 3, 32, 48, 4, 2, 64, 5
    q, k, v = _random_qkv(203, batch, sq, sk, hq, hkv, dim, dim)
    k_store = np.zeros((batch, sk + pad, hkv, dim), dtype=k.dtype)
    v_store = np.zeros_like(k_store)
    k_store[:, :sk], v_store[:, :sk] = k, v
    scale = 1.0 / np.sqrt(dim)
    request = _dense_direct_request(batch, sq, sk, hq, hkv, dim, dim)
    actual = _run_dense_direct(
        request, q, k, v, scale=scale, kv_store=(k_store, v_store)
    )
    expected = _windowed_reference(q, k, v, scale=scale, ctx=0, left=0, right=-1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-2, equal_nan=False)


def _run_dense_views(request, q, k, v, *, scale, out_bhsd=False):
    """Launch one dense request over numpy *views* of larger backing arrays and
    return the fp32 output as ``[B, Sq, Hq, Dv]``.

    Each of ``q``/``k``/``v`` is a ``[B, S, H, D]``-shaped view; its element
    strides and byte offset into the root array are forwarded unchanged, so
    permuted (BHSD-backed) and packed-QKV views reach the binder as-is."""
    from benchmarks.gfx1151.attention.benchmark_sdpa import _host_bytes
    from rocke.runtime.launcher import release_retained_for_stream
    from rocke.runtime.torch_interop import resolve_stream

    stream = resolve_stream(0)
    rt = Runtime()
    b, sq, hq = q.shape[:3]
    dv = v.shape[-1]
    if out_bhsd:
        out_root = np.full((b, hq, sq, dv), np.nan, dtype=q.dtype)
        out = out_root.transpose(0, 2, 1, 3)
    else:
        out_root = out = np.full((b, sq, hq, dv), np.nan, dtype=q.dtype)
    uploaded = {}
    try:

        def root_of(a):
            return a.base if a.base is not None else a

        def device_view(a):
            root = root_of(a)
            if id(root) not in uploaded:
                ptr = rt.alloc(root.nbytes)
                rt.memcpy_h2d(ptr, _host_bytes(root), root.nbytes)
                uploaded[id(root)] = ptr
            offset = (
                a.__array_interface__["data"][0]
                - root.__array_interface__["data"][0]
            )
            return _DeviceTensor(
                _ptr=uploaded[id(root)] + offset,
                _shape=tuple(a.shape),
                _strides=tuple(st // a.itemsize for st in a.strides),
                _dtype=str(a.dtype),
            )

        tensors = {
            "q": device_view(q),
            "k": device_view(k),
            "v": device_view(v),
            "out": device_view(out),
        }
        binding = dispatch_attention(request).bind_torch(
            tensors, softmax_scale=scale, stream=stream, fence=False
        )
        binding.launch()
        rt.stream_sync(stream)
        release_retained_for_stream(stream)
        rt.memcpy_d2h(_host_bytes(out_root), uploaded[id(out_root)], out_root.nbytes)
        return np.ascontiguousarray(out).astype(np.float32)
    finally:
        rt.sync()
        release_retained_for_stream(stream)
        for ptr in uploaded.values():
            rt.free(ptr)


@pytest.mark.gpu
@_NEEDS_GPU
@pytest.mark.parametrize("out_bhsd", [False, True])
@pytest.mark.parametrize("batch", [1, 3])
def test_dispatch_dense_bhsd_backed_views(batch, out_bhsd):
    """BHSD-backed storage exposed as permuted ``[B, S, H, D]`` views."""
    sq, sk, hq, hkv, dim = 40, 56, 4, 2, 64
    q, k, v = _random_qkv(204, batch, sq, sk, hq, hkv, dim, dim)

    def bhsd_backed(x):
        root = np.ascontiguousarray(x.transpose(0, 2, 1, 3))
        return root.transpose(0, 2, 1, 3)

    scale = 1.0 / np.sqrt(dim)
    request = _dense_direct_request(batch, sq, sk, hq, hkv, dim, dim)
    actual = _run_dense_views(
        request,
        bhsd_backed(q),
        bhsd_backed(k),
        bhsd_backed(v),
        scale=scale,
        out_bhsd=out_bhsd,
    )
    expected = _windowed_reference(q, k, v, scale=scale, ctx=0, left=0, right=-1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-2, equal_nan=False)


@pytest.mark.gpu
@_NEEDS_GPU
@pytest.mark.parametrize("batch", [1, 2])
def test_dispatch_dense_packed_qkv_views(batch):
    """Q/K/V as strided slices of one packed ``[B, S, Hq + 2*Hkv, D]`` buffer."""
    s, hq, hkv, dim = 48, 4, 2, 64
    q, k, v = _random_qkv(205, batch, s, s, hq, hkv, dim, dim)
    packed = np.concatenate([q, k, v], axis=2)
    pq = packed[:, :, :hq]
    pk = packed[:, :, hq : hq + hkv]
    pv = packed[:, :, hq + hkv :]
    scale = 1.0 / np.sqrt(dim)
    request = _dense_direct_request(batch, s, s, hq, hkv, dim, dim)
    actual = _run_dense_views(request, pq, pk, pv, scale=scale)
    expected = _windowed_reference(q, k, v, scale=scale, ctx=0, left=0, right=-1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-2, equal_nan=False)


# ---------------------------------------------------------------------
# Softmax statistics (LSE): natural-log log-sum-exp per (batch, head, query).
# ---------------------------------------------------------------------


def _launch_np(request, arrays, outputs, *, scale):
    """Upload each named host array, launch through the public binder, and
    return the downloaded arrays named in ``outputs``."""
    from benchmarks.gfx1151.attention.benchmark_sdpa import _host_bytes
    from rocke.runtime.launcher import release_retained_for_stream
    from rocke.runtime.torch_interop import resolve_stream

    stream = resolve_stream(0)
    rt = Runtime()
    ptrs = {}
    try:
        tensors = {}
        for name, a in arrays.items():
            a = np.ascontiguousarray(a)
            arrays[name] = a
            ptrs[name] = rt.alloc(max(a.nbytes, 1))
            rt.memcpy_h2d(ptrs[name], _host_bytes(a), a.nbytes)
            tensors[name] = _DeviceTensor(
                _ptr=ptrs[name],
                _shape=tuple(a.shape),
                _strides=tuple(s // a.itemsize for s in a.strides),
                _dtype=str(a.dtype),
            )
        binding = dispatch_attention(request).bind_torch(
            tensors, softmax_scale=scale, stream=stream, fence=False
        )
        binding.launch()
        rt.stream_sync(stream)
        release_retained_for_stream(stream)
        for name in outputs:
            rt.memcpy_d2h(_host_bytes(arrays[name]), ptrs[name], arrays[name].nbytes)
        return {name: arrays[name].copy() for name in outputs}
    finally:
        rt.sync()
        release_retained_for_stream(stream)
        for ptr in ptrs.values():
            rt.free(ptr)


def _lse_reference(q, k, scale, *, ctx, left, right, sinks=None):
    """FP64 natural-log LSE ``[B, Hq, Sq]`` for ``[B, S, H, D]`` inputs; ``-inf``
    for rows with no visible key and no sink."""
    _, sq, hq, _ = q.shape
    sk, hkv = k.shape[1], k.shape[2]
    q64, k64 = q.astype(np.float64), k.astype(np.float64)
    k64 = np.repeat(k64, hq // hkv, axis=2)
    qi = np.arange(sq).reshape(sq, 1)
    ki = np.arange(sk).reshape(1, sk)
    keep = np.ones((sq, sk), dtype=bool)
    if right >= 0:
        keep &= ki <= qi + ctx + right
    if left > 0:
        keep &= ki > qi + ctx - left
    scores = np.einsum("bqhd,bkhd->bhqk", q64 * scale, k64)
    scores = np.where(keep.reshape(1, 1, sq, sk), scores, -np.inf)
    if sinks is not None:
        sink = np.broadcast_to(
            sinks.astype(np.float64).reshape(1, hq, 1, 1), scores.shape[:3] + (1,)
        )
        scores = np.concatenate([scores, sink], axis=-1)
    row_max = scores.max(axis=-1, keepdims=True)
    safe_max = np.where(np.isfinite(row_max), row_max, 0.0)
    denom = np.exp(scores - safe_max).sum(axis=-1)
    with np.errstate(divide="ignore"):
        lse = np.log(denom) + safe_max[..., 0]
    return lse.astype(np.float32)


def _assert_lse_close(actual, expected):
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    assert np.isfinite(actual[finite]).all()
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=0, atol=1e-3)


@pytest.mark.gpu
@_NEEDS_GPU
@pytest.mark.parametrize(
    "mask,sq,sk,left,right,use_sinks",
    [
        (AttentionMaskType.NO_MASK, 33, 47, 0, -1, False),
        (AttentionMaskType.TOP_LEFT_CAUSAL, 40, 40, 0, -1, True),
        (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 17, 49, 0, -1, False),
        (AttentionMaskType.SLIDING_WINDOW, 33, 41, 24, 8, True),
        # Bottom-right causal with sk < sq: the first sq - sk rows see no key.
        (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 48, 20, 0, -1, False),
        (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 48, 20, 0, -1, True),
    ],
)
def test_dispatch_dense_lse(mask, sq, sk, left, right, use_sinks):
    batch, hq, hkv, dim = 2, 4, 2, 64
    q, k, v = _random_qkv(301, batch, sq, sk, hq, hkv, dim, dim)
    scale = 1.0 / np.sqrt(dim)
    sinks = (
        np.random.default_rng(302).standard_normal(hq).astype(np.float16)
        if use_sinks
        else None
    )
    request = _dense_direct_request(
        batch,
        sq,
        sk,
        hq,
        hkv,
        dim,
        dim,
        mask_type=mask,
        sliding_window=left,
        window_right=right,
        use_sinks=use_sinks,
        return_lse=True,
    )
    arrays = {
        "q": q,
        "k": k,
        "v": v,
        "out": np.full((batch, sq, hq, dim), np.nan, dtype=q.dtype),
        "lse": np.full((batch, hq, sq), np.nan, dtype=np.float32),
    }
    if use_sinks:
        arrays["sinks"] = sinks
    got = _launch_np(request, arrays, ("lse",), scale=scale)["lse"]
    ctx = sk - sq if mask == AttentionMaskType.BOTTOM_RIGHT_CAUSAL else 0
    if mask in (
        AttentionMaskType.TOP_LEFT_CAUSAL,
        AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
    ):
        right = 0
    expected = _lse_reference(q, k, scale, ctx=ctx, left=left, right=right, sinks=sinks)
    _assert_lse_close(got, expected)
    if not use_sinks and mask == AttentionMaskType.BOTTOM_RIGHT_CAUSAL and sk < sq:
        assert np.isneginf(got[:, :, : sq - sk]).all()


@pytest.mark.gpu
@_NEEDS_GPU
def test_dispatch_dense_lse_does_not_change_output():
    batch, sq, sk, hq, hkv, dim = 2, 37, 53, 4, 2, 64
    q, k, v = _random_qkv(303, batch, sq, sk, hq, hkv, dim, dim)
    scale = 1.0 / np.sqrt(dim)
    base = _dense_direct_request(batch, sq, sk, hq, hkv, dim, dim)
    with_lse = _dense_direct_request(batch, sq, sk, hq, hkv, dim, dim, return_lse=True)

    def arrays(with_stats):
        a = {
            "q": q,
            "k": k,
            "v": v,
            "out": np.full((batch, sq, hq, dim), np.nan, dtype=q.dtype),
        }
        if with_stats:
            a["lse"] = np.full((batch, hq, sq), np.nan, dtype=np.float32)
        return a

    out_base = _launch_np(base, arrays(False), ("out",), scale=scale)["out"]
    out_lse = _launch_np(with_lse, arrays(True), ("out",), scale=scale)["out"]
    np.testing.assert_array_equal(out_base, out_lse)


@pytest.mark.gpu
@_NEEDS_GPU
def test_dispatch_dense_lse_gapped_batch_stride_is_served_per_batch():
    """K/V with a batch-stride gap force per-batch launches; ``lse`` must be
    sliced per batch too."""
    batch, sq, sk, hq, hkv, dim, pad = 3, 32, 48, 4, 2, 64, 5
    q, k, v = _random_qkv(304, batch, sq, sk, hq, hkv, dim, dim)
    k_store = np.zeros((batch, sk + pad, hkv, dim), dtype=k.dtype)
    v_store = np.zeros_like(k_store)
    k_store[:, :sk], v_store[:, :sk] = k, v
    scale = 1.0 / np.sqrt(dim)
    request = _dense_direct_request(batch, sq, sk, hq, hkv, dim, dim, return_lse=True)
    lse = np.full((batch, hq, sq), np.nan, dtype=np.float32)
    out = np.full((batch, sq, hq, dim), np.nan, dtype=q.dtype)
    got = _launch_views_lse(
        request, q, k_store[:, :sk], v_store[:, :sk], out, lse, scale=scale
    )
    expected = _lse_reference(q, k, scale, ctx=0, left=0, right=-1)
    _assert_lse_close(got, expected)


def _launch_views_lse(request, q, k, v, out, lse, *, scale):
    """Launch over numpy views of larger backing arrays and return ``lse``."""
    from benchmarks.gfx1151.attention.benchmark_sdpa import _host_bytes
    from rocke.runtime.launcher import release_retained_for_stream
    from rocke.runtime.torch_interop import resolve_stream

    stream = resolve_stream(0)
    rt = Runtime()
    uploaded = {}
    try:

        def device_view(a):
            root = a.base if a.base is not None else a
            if id(root) not in uploaded:
                ptr = rt.alloc(root.nbytes)
                rt.memcpy_h2d(ptr, _host_bytes(root), root.nbytes)
                uploaded[id(root)] = ptr
            offset = (
                a.__array_interface__["data"][0] - root.__array_interface__["data"][0]
            )
            return _DeviceTensor(
                _ptr=uploaded[id(root)] + offset,
                _shape=tuple(a.shape),
                _strides=tuple(st // a.itemsize for st in a.strides),
                _dtype=str(a.dtype),
            )

        tensors = {
            name: device_view(a)
            for name, a in (("q", q), ("k", k), ("v", v), ("out", out), ("lse", lse))
        }
        binding = dispatch_attention(request).bind_torch(
            tensors, softmax_scale=scale, stream=stream, fence=False
        )
        binding.launch()
        rt.stream_sync(stream)
        release_retained_for_stream(stream)
        rt.memcpy_d2h(_host_bytes(lse), uploaded[id(lse)], lse.nbytes)
        return lse.copy()
    finally:
        rt.sync()
        release_retained_for_stream(stream)
        for ptr in uploaded.values():
            rt.free(ptr)


@pytest.mark.gpu
@_NEEDS_GPU
@pytest.mark.parametrize("use_sinks", [False, True])
def test_dispatch_ragged_lse_gqa(use_sinks):
    """Packed ragged GQA: ``lse`` is ``[Hq, total_q]``. Empty-query sequences
    own no rows; with bottom-right alignment and ``Sk < Sq`` the leading rows
    of a sequence see no key and store ``-inf`` (without sinks)."""
    hq, hkv, dim = 4, 2, 64
    q_lengths = (0, 17, 1, 3, 25)
    k_lengths = (9, 17, 5, 2, 40)
    total_q, total_k = sum(q_lengths), sum(k_lengths)
    rng = np.random.default_rng(305)
    q = rng.standard_normal((total_q, hq, dim)).astype(np.float16)
    k = rng.standard_normal((total_k, hkv, dim)).astype(np.float16)
    v = rng.standard_normal((total_k, hkv, dim)).astype(np.float16)
    sinks = rng.standard_normal(hq).astype(np.float16) if use_sinks else None
    cu_q = np.concatenate([[0], np.cumsum(q_lengths)]).astype(np.int32)
    cu_k = np.concatenate([[0], np.cumsum(k_lengths)]).astype(np.int32)
    scale = 1.0 / np.sqrt(dim)
    request = AttentionRequest(
        batch=len(q_lengths),
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=max(q_lengths),
        seqlen_k=max(k_lengths),
        hdim_q=dim,
        hdim_v=dim,
        arch="gfx1151",
        dtype="fp16",
        layout="ragged",
        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        use_sinks=use_sinks,
        return_lse=True,
    )
    arrays = {
        "q": q,
        "k": k,
        "v": v,
        "out": np.full((total_q, hq, dim), np.nan, dtype=q.dtype),
        "lse": np.full((hq, total_q), np.nan, dtype=np.float32),
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
    }
    if use_sinks:
        arrays["sinks"] = sinks
    got = _launch_np(request, arrays, ("lse",), scale=scale)["lse"]
    expected = np.empty((hq, total_q), dtype=np.float32)
    for i in range(len(q_lengths)):
        qs, qe, ks, ke = cu_q[i], cu_q[i + 1], cu_k[i], cu_k[i + 1]
        if qe == qs:
            continue
        expected[:, qs:qe] = _lse_reference(
            q[None, qs:qe],
            k[None, ks:ke],
            scale,
            ctx=int(ke - ks) - int(qe - qs),
            left=0,
            right=0,
            sinks=sinks,
        )[0]
    _assert_lse_close(got, expected)
    if not use_sinks:
        assert np.isneginf(got).any()
