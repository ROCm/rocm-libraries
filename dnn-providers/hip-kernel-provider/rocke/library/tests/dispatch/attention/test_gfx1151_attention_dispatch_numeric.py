# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Actual-surface regressions for the RDNA (gfx11 / gfx12) WMMA attention
binders.

The GPU tests here go through the *public dispatch surface* --
``dispatch_attention(req).bind_torch(tensors, **kwargs).launch()`` -- so an
ABI-order mismatch, a launcher-cache bug, a dropped stream, a split-workspace
bug or a tensor-layout translation error in ``bind_rdna_prefill_torch`` /
``bind_rdna_decode_torch`` fails here even when the kernel body itself is
correct. Every dispatch test pins ``algorithm`` and runs both kernels: the
multi-wave prefill (``wmma_fmha_fwd_v2``) and the split-KV decode
(``wmma_fmha_decode``), which serve the same request contract. A few tests
bind an explicit (wider) head-width bucket directly through the two binders.

Tensors are lightweight duck-typed views (``shape``/``dtype``/``stride``/
``data_ptr``/``device``) over real HIP buffers allocated by the frozen
benchmark harness's ``DeviceBuffers`` -- deliberately not ``torch``, since
``bindings.py`` never imports it and this is the proof that holds.

The trailing ``TestMetadataSafety`` class needs no GPU: it calls
``_rdna_validate_and_collect`` and the two binders directly with plain
``SimpleNamespace`` tensor doubles to pin the structural (shape/dtype/stride/
alignment) launch gate without ever reading ``cu_seqlens``/``seqused_k``/
``block_table`` contents or compiling a kernel.
"""

from __future__ import annotations

import ctypes
import dataclasses
import math
from typing import Optional, Tuple
from types import SimpleNamespace

import numpy as np
import pytest

from dispatch.attention import AttentionMaskType, AttentionRequest, dispatch_attention
from rocke.core.arch import generic_arch_from_target_id
from rocke.runtime import hip_module
from rocke.runtime.hip_module import Runtime, get_device_arch

# Every gfx11 (gfx12) device dispatches one gfx11-generic (gfx12-generic) set.
_ARCH = generic_arch_from_target_id(get_device_arch() or "")
_NEEDS_GPU = pytest.mark.skipif(
    _ARCH not in ("gfx11-generic", "gfx12-generic"),
    reason="needs an RDNA3/3.5 (gfx11) or RDNA4 (gfx12) GPU",
)

_PREFILL = "wmma_fmha_fwd_v2"
_DECODE = "wmma_fmha_decode"
_BOTH_KERNELS = pytest.mark.parametrize("algorithm", [_PREFILL, _DECODE])


def _decode_serves(head_dim_q: int, head_dim_v: int) -> bool:
    from kernels.gfx1151.wmma_fmha_decode import wmma_fmha_decode_bucket

    bucket = wmma_fmha_decode_bucket(head_dim_q=head_dim_q, head_dim_v=head_dim_v)
    return bucket is not None


def _require_served(algorithm: str, head_dim_q: int, head_dim_v: int) -> None:
    """Skip the decode parametrization where no decode bucket serves the
    widths (a bucket serves Q/K widths only from its own minimum up)."""
    if algorithm == _DECODE and not _decode_serves(head_dim_q, head_dim_v):
        pytest.skip(f"no decode bucket serves head widths {head_dim_q}/{head_dim_v}")


def _decode_bucket_serves(bucket: int, head_dim_q: int) -> bool:
    """Whether the decode object of ``bucket`` serves ``head_dim_q`` (the
    binder's own rule: the bucket the Q width selects next to a full-bucket V
    width is this one)."""
    from kernels.gfx1151.wmma_fmha_decode import wmma_fmha_decode_bucket

    return (
        bucket in (64, 128, 256)
        and wmma_fmha_decode_bucket(head_dim_q=head_dim_q, head_dim_v=bucket) == bucket
    )


def _bucket_spec(algorithm, bucket, *, v_bucket=0, causal=False, bias=None):
    """An explicit fp16 object of ``algorithm`` with head-width ``bucket``
    (prefill: Q/K ``bucket`` and V/O ``v_bucket``; decode: one bucket)."""
    if algorithm == _PREFILL:
        from kernels.gfx1151.wmma_fmha_fwd_v2 import WmmaFmhaFwdV2Spec

        return WmmaFmhaFwdV2Spec(
            dtype="fp16",
            head_dim=bucket,
            head_dim_v=v_bucket,
            mask="causal" if causal else "none",
            bias=bias,
        )
    from kernels.gfx1151.wmma_fmha_decode import (
        WmmaFmhaDecodeSpec,
        resolve_wmma_fmha_decode_spec,
    )

    return resolve_wmma_fmha_decode_spec(
        WmmaFmhaDecodeSpec(
            head_size=max(bucket, v_bucket), dtype="fp16", bias=bias or ""
        ),
        arch=_ARCH,
    )


def _bind_explicit(request, spec, tensors, **kwargs):
    from dispatch.attention.bindings import (
        bind_rdna_decode_torch,
        bind_rdna_prefill_torch,
    )
    from kernels.gfx1151.wmma_fmha_decode import WmmaFmhaDecodeSpec

    if isinstance(spec, WmmaFmhaDecodeSpec):
        return bind_rdna_decode_torch(request, spec, tensors, **kwargs)
    return bind_rdna_prefill_torch(request, spec, tensors, **kwargs)


def _dispatch_pinned(request: AttentionRequest):
    """``dispatch_attention`` for a request that pins one RDNA kernel, with
    the pin checked (never a silent fallback to the other kernel)."""
    result = dispatch_attention(request)
    assert result.candidate.algorithm == request.algorithm, result.candidate.name
    return result


def _num_cus_for(request: AttentionRequest, path: str) -> int:
    """A request ``num_cus`` whose decode plan takes ``path``: one split
    (``single``), the fused in-kernel merge (``fused``, 2..4 splits) or the
    separate merge kernel (``merge``, more than 4 splits)."""
    from dispatch.attention.gfx1151 import request_decode_plan

    spec = _dispatch_pinned(request).spec
    for n in range(2, 8193, 2):
        plan = request_decode_plan(dataclasses.replace(request, num_cus=n), spec)
        if (
            (path == "single" and plan.num_splits == 1)
            or (path == "fused" and plan.fused)
            or (path == "merge" and plan.num_splits > 4)
        ):
            return n
    pytest.fail(f"no CU count gives a {path!r} decode plan for {request}")


def _assert_decode_path(binding, path: str) -> None:
    plan = binding.launch.plan
    if path == "single":
        assert plan.num_splits == 1 and not plan.fused
        assert plan.workspace["bytes"] == 0
    elif path == "fused":
        assert 2 <= plan.num_splits <= 4 and plan.fused
    else:
        assert plan.num_splits > 4 and not plan.fused


# ---------------------------------------------------------------------
# Duck-typed device tensor (never torch) backed by a real HIP pointer.
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Device:
    type: str = "cuda"
    index: int = 0


@dataclasses.dataclass(frozen=True)
class _DeviceTensor:
    """Satisfies the RDNA binders' tensor protocol (shape/dtype/stride/
    data_ptr/device/is_contiguous) over a numpy-array-shaped view of a
    ``DeviceBuffers``-owned device pointer."""

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


def _view_as(buffers, name: str, array: np.ndarray) -> _DeviceTensor:
    """Describe a logical NumPy view over a named contiguous device allocation."""
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


def _request_for(case, algorithm: str, *, num_cus: int = 0) -> AttentionRequest:
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
        arch=_ARCH,
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
        num_cus=num_cus,
        algorithm=algorithm,
    )


def _launch_and_check(
    case,
    algorithm: str,
    *,
    stream: int = 0,
    fp8_scales=None,
    inputs=None,
    num_cus: int = 0,
    launches: int = 1,
):
    """Launch ``case`` through the dispatch surface ``launches`` times (one
    binding; the output is poisoned between launches) and check every result
    against the CPU oracle and the first launch. Returns ``(output,
    binding)``."""
    from benchmarks.gfx1151.attention.benchmark_sdpa import DeviceBuffers
    from benchmarks.gfx1151.attention.cases import make_inputs, reference
    from rocke.runtime.launcher import release_retained_for_stream
    from rocke.runtime.torch_interop import resolve_stream

    if case.kv_dtype:
        pytest.importorskip("ml_dtypes")
    if inputs is None:
        inputs = make_inputs(case)
    if fp8_scales is not None:
        inputs.k_scale = np.asarray(fp8_scales[0], np.float32)
        inputs.v_scale = np.asarray(fp8_scales[1], np.float32)
    stream = resolve_stream(stream)
    rt = Runtime()
    buffers = DeviceBuffers(rt, inputs)
    try:
        result = _dispatch_pinned(_request_for(case, algorithm, num_cus=num_cus))
        kwargs = {"softmax_scale": case.scale, "stream": stream, "fence": False}
        if case.softcap:
            kwargs["softcap"] = case.softcap
        if case.kv_dtype:
            kwargs["k_scale"] = float(inputs.k_scale)
            kwargs["v_scale"] = float(inputs.v_scale)
        binding = result.bind_torch(_tensors_for(buffers, case), **kwargs)
        expected = reference(case, inputs)
        out_bytes = buffers.arrays["rocke_out"].nbytes
        first = None
        for _ in range(launches):
            if first is not None and out_bytes:
                rt.memset(buffers.ptrs["rocke_out"], 0xFF, out_bytes)  # NaN
                rt.sync()
            binding.launch()
            rt.stream_sync(stream)
            release_retained_for_stream(stream)
            actual = buffers.read_output("rocke_out")
            np.testing.assert_allclose(
                actual, expected, rtol=0, atol=case.atol, equal_nan=False
            )
            if first is None:
                first = actual
            else:
                np.testing.assert_array_equal(actual, first)
        return first, binding
    finally:
        rt.sync()
        release_retained_for_stream(stream)
        buffers.close()


def _case(**kw):
    from benchmarks.gfx1151.attention.cases import SdpaCase

    return SdpaCase(**kw)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_dispatch_dense_partial_tail_nondefault_stream(dtype, algorithm):
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
        _launch_and_check(case, algorithm, stream=stream)
    finally:
        _destroy_stream(stream)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_dispatch_ragged_empty_and_mixed_lengths(dtype, algorithm):
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
    _launch_and_check(case, algorithm)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize(
    "dtype,block_size", [("fp16", 16), ("bf16", 16), ("fp16", 8), ("bf16", 32)]
)
def test_dispatch_paged_shuffled_partial_page(dtype, block_size, algorithm):
    """Paged KV with a shuffled (non-identity) block table and a partial
    last page, including a page shorter than one 16-key tile."""
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
        block_size=block_size,
        mask="causal_bottomright",
        q_lengths=(0, 17, 1, 3, 0),
        k_lengths=(9, 19, 1, 0, 0),
        seed=27,
    )
    _launch_and_check(case, algorithm)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize(
    "layout,dtype",
    [("dense", "fp16"), ("ragged", "bf16"), ("paged", "fp16")],
)
def test_dispatch_fp8_kv(layout, dtype, algorithm):
    """OCP fp8e4m3 KV storage with explicit non-power-of-two dequant scales,
    through the public dispatch surface (``use_fp8=True`` -> ``kv_dtype``)."""
    lengths = (
        dict(batch=1, seqlen_q=32, seqlen_k=32)
        if layout == "dense"
        else dict(
            batch=3, seqlen_q=0, seqlen_k=0, q_lengths=(1, 17, 0), k_lengths=(40, 23, 7)
        )
    )
    case = _case(
        name="dispatch_fp8_kv",
        group="dispatch",
        dtype=dtype,
        heads_q=2 if layout == "dense" else 4,
        heads_kv=2,
        head_dim=64,
        layout=layout,
        block_size=16 if layout == "paged" else 0,
        mask="causal_topleft" if layout == "dense" else "causal_bottomright",
        kv_dtype="fp8e4m3",
        seed=7,
        **lengths,
    )
    _launch_and_check(case, algorithm, fp8_scales=(0.3, 0.7))


# Score features the frozen benchmark oracle expresses: softcap, sinks, ALiBi
# and QQ-bias across the three layouts (and a causal window).
_FEATURE_CASES = {
    "softcap_dense": dict(
        layout="dense", batch=2, seqlen_q=19, seqlen_k=37, softcap=30.0
    ),
    "sinks_window_dense": dict(
        layout="dense", batch=2, seqlen_q=33, seqlen_k=41, window=16, sinks=True
    ),
    "qq_bias_dense": dict(
        layout="dense", batch=1, seqlen_q=32, seqlen_k=48, mask="none", qq_bias=True
    ),
    "alibi_ragged": dict(
        layout="ragged",
        batch=3,
        q_lengths=(1, 5, 17),
        k_lengths=(40, 18, 33),
        alibi=True,
    ),
    "qq_bias_ragged": dict(
        layout="ragged",
        batch=3,
        q_lengths=(16, 33, 8),
        k_lengths=(16, 64, 40),
        qq_bias=True,
    ),
    "sinks_alibi_paged": dict(
        layout="paged",
        batch=2,
        block_size=32,
        q_lengths=(1, 3),
        k_lengths=(70, 9),
        sinks=True,
        alibi=True,
    ),
    "softcap_window_paged8": dict(
        layout="paged",
        batch=2,
        block_size=8,
        q_lengths=(5, 1),
        k_lengths=(29, 50),
        softcap=20.0,
        window=24,
    ),
}


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize("feature", sorted(_FEATURE_CASES))
def test_dispatch_score_features(feature, algorithm):
    fields = dict(_FEATURE_CASES[feature])
    fields.setdefault("mask", "causal_bottomright")
    if "q_lengths" in fields:
        fields.update(seqlen_q=0, seqlen_k=0)
    case = _case(
        name=f"dispatch_{feature}",
        group="dispatch",
        dtype="bf16" if "paged" in feature else "fp16",
        heads_q=8,
        heads_kv=2,
        head_dim=64,
        seed=111,
        **fields,
    )
    _launch_and_check(case, algorithm)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_dispatch_aligned_gqa_on_nondefault_stream(dtype, algorithm):
    case = _case(
        name="dispatch_aligned_gqa",
        group="dispatch",
        dtype=dtype,
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
        _launch_and_check(case, algorithm, stream=stream)
    finally:
        _destroy_stream(stream)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize(
    "mask,q_lengths,k_lengths",
    [
        ("causal_topleft", (32, 128), (64, 32)),
        ("causal_bottomright", (64, 96), (128, 32)),
    ],
)
def test_ragged_full_window_preserves_all_causal_keys(
    mask, q_lengths, k_lengths, algorithm
):
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
    actual, _ = _launch_and_check(case, algorithm)
    if mask == "causal_bottomright":
        begin = q_lengths[0]
        end = begin + q_lengths[1] - k_lengths[1]
        np.testing.assert_array_equal(
            actual[begin:end], np.zeros_like(actual[begin:end])
        )


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize(
    "layout,dtype,storage",
    [
        ("dense", "fp16", ""),
        ("dense", "bf16", ""),
        ("ragged", "fp16", "fp8e4m3"),
        ("paged", "bf16", "fp8e4m3"),
    ],
)
def test_full_qk_width_reaches_every_output_column(layout, dtype, storage, algorithm):
    """D256: the last QK column decides every score and every one of the 256
    output columns carries a distinct value, so truncating QK to an output
    slice or dropping an output column fails the oracle."""
    from benchmarks.gfx1151.attention.cases import make_inputs

    if storage:
        pytest.importorskip("ml_dtypes")
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
        name="dispatch_full_qk_width",
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
    inputs.q.fill(0)
    inputs.q[..., -1] = 1
    signs = np.where(np.indices(inputs.k.shape[:-1]).sum(axis=0) % 2, 1.0, -1.0)
    inputs.k.fill(0)
    inputs.k[..., -1] = signs * 8
    inputs.v[...] = signs[..., None] + np.arange(256, dtype=np.float32) / 512
    actual, _ = _launch_and_check(case, algorithm, inputs=inputs)
    if layout == "ragged":
        np.testing.assert_array_equal(actual[:3], np.zeros_like(actual[:3]))
    elif layout == "paged":
        np.testing.assert_array_equal(actual[1:], np.zeros_like(actual[1:]))


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize(
    "mask,seqlen_q", [("causal_topleft", 97), ("causal_bottomright", 65)]
)
def test_dense_long_queries_and_empty_bottom_right_prefix(mask, seqlen_q, algorithm):
    """More queries than keys: every query row is served, and bottom-right
    rows before the first key are zero."""
    case = _case(
        name="dispatch_dense_long_queries",
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
    actual, _ = _launch_and_check(case, algorithm)
    if mask == "causal_bottomright":
        prefix = seqlen_q - case.seqlen_k
        np.testing.assert_array_equal(
            actual[:, :prefix], np.zeros_like(actual[:, :prefix])
        )


# ---------------------------------------------------------------------
# Split-KV decode: one split, the fused in-kernel merge (2..4 splits) and the
# merge kernel (more than 4), forced through the request's CU count. Each
# binding launches twice: the fused merge's arrival counters must re-arm.
# ---------------------------------------------------------------------


@pytest.mark.gpu
@_NEEDS_GPU
@pytest.mark.parametrize("path", ["single", "fused", "merge"])
@pytest.mark.parametrize("layout", ["dense", "ragged", "paged"])
def test_decode_split_kv_paths_relaunch(layout, path):
    lengths = (
        dict(batch=1, seqlen_q=3, seqlen_k=9001)
        if layout == "dense"
        else dict(
            batch=2,
            seqlen_q=0,
            seqlen_k=0,
            q_lengths=(1, 3),
            k_lengths=(8191, 9100),
            block_size=16 if layout == "paged" else 0,
        )
    )
    case = _case(
        name="dispatch_decode_splits",
        group="dispatch",
        dtype="bf16" if layout == "ragged" else "fp16",
        heads_q=8,
        heads_kv=2,
        head_dim=64,
        layout=layout,
        mask="causal_bottomright",
        seed=121,
        **lengths,
    )
    num_cus = _num_cus_for(_request_for(case, _DECODE), path)
    _, binding = _launch_and_check(case, _DECODE, num_cus=num_cus, launches=2)
    _assert_decode_path(binding, path)


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
        binding = _dispatch_pinned(request).bind_torch(
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


def _dense_direct_request(batch, sq, sk, hq, hkv, dq, dv, *, algorithm, **extra):
    return AttentionRequest(
        batch=batch,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=sq,
        seqlen_k=sk,
        hdim_q=dq,
        hdim_v=dv,
        arch=_ARCH,
        dtype="fp16",
        layout="dense",
        algorithm=algorithm,
        **extra,
    )


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize(
    "mask,sq,sk,left,right",
    [
        (AttentionMaskType.NO_MASK, 48, 64, 0, 16),
        (AttentionMaskType.SLIDING_WINDOW, 33, 41, 24, 8),
        (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 33, 49, 0, 16),
        (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 40, 40, 20, 0),
    ],
)
def test_dispatch_two_sided_local_window(mask, sq, sk, left, right, algorithm):
    hq, hkv, dim = 4, 2, 64
    q, k, v = _random_qkv(201, 2, sq, sk, hq, hkv, dim, dim)
    scale = 1.0 / np.sqrt(dim)
    request = _dense_direct_request(
        2,
        sq,
        sk,
        hq,
        hkv,
        dim,
        dim,
        algorithm=algorithm,
        mask_type=mask,
        sliding_window=left,
        window_right=right,
    )
    ctx = sk - sq if mask == AttentionMaskType.BOTTOM_RIGHT_CAUSAL else 0
    actual = _run_dense_direct(request, q, k, v, scale=scale)
    expected = _windowed_reference(
        q, k, v, scale=scale, ctx=ctx, left=left, right=right
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-2, equal_nan=False)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize(
    "mask", [AttentionMaskType.TOP_LEFT_CAUSAL, AttentionMaskType.BOTTOM_RIGHT_CAUSAL]
)
@pytest.mark.parametrize("window_left", [9, -1])
def test_dispatch_causal_mask_with_window_left(mask, window_left, algorithm):
    """cuDNN-style causal sliding window: a causal mask plus an explicit left
    bound keeps keys in ``[q + ctx - window_left, q + ctx]``; ``window_left=-1``
    is unbounded on the left and keeps the causal diagonal."""
    batch, sq, sk, hq, hkv, dim = 2, 37, 53, 4, 2, 64
    q, k, v = _random_qkv(211, batch, sq, sk, hq, hkv, dim, dim)
    scale = 1.0 / np.sqrt(dim)
    request = _dense_direct_request(
        batch,
        sq,
        sk,
        hq,
        hkv,
        dim,
        dim,
        algorithm=algorithm,
        mask_type=mask,
        window_left=window_left,
    )
    ctx = sk - sq if mask == AttentionMaskType.BOTTOM_RIGHT_CAUSAL else 0
    actual = _run_dense_direct(request, q, k, v, scale=scale)
    expected = _windowed_reference(
        q, k, v, scale=scale, ctx=ctx, left=max(window_left + 1, 0), right=0
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-2, equal_nan=False)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize("dq,dv", [(64, 128), (128, 64), (80, 48), (192, 128)])
def test_dispatch_unequal_head_dims(dq, dv, algorithm):
    _require_served(algorithm, dq, dv)
    hq, hkv, sq, sk = 4, 2, 37, 53
    q, k, v = _random_qkv(202, 2, sq, sk, hq, hkv, dq, dv)
    scale = 1.0 / np.sqrt(dq)
    request = _dense_direct_request(2, sq, sk, hq, hkv, dq, dv, algorithm=algorithm)
    actual = _run_dense_direct(request, q, k, v, scale=scale)
    expected = _windowed_reference(q, k, v, scale=scale, ctx=0, left=0, right=-1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-2, equal_nan=False)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
def test_dispatch_dense_gapped_batch_stride_uses_runtime_stride(algorithm):
    batch, sq, sk, hq, hkv, dim, pad = 3, 32, 48, 4, 2, 64, 5
    q, k, v = _random_qkv(203, batch, sq, sk, hq, hkv, dim, dim)
    k_store = np.zeros((batch, sk + pad, hkv, dim), dtype=k.dtype)
    v_store = np.zeros_like(k_store)
    k_store[:, :sk], v_store[:, :sk] = k, v
    scale = 1.0 / np.sqrt(dim)
    request = _dense_direct_request(
        batch, sq, sk, hq, hkv, dim, dim, algorithm=algorithm
    )
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
                a.__array_interface__["data"][0] - root.__array_interface__["data"][0]
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
        binding = _dispatch_pinned(request).bind_torch(
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
@_BOTH_KERNELS
@pytest.mark.parametrize("out_bhsd", [False, True])
@pytest.mark.parametrize("batch", [1, 3])
def test_dispatch_dense_bhsd_backed_views(batch, out_bhsd, algorithm):
    """BHSD-backed storage exposed as permuted ``[B, S, H, D]`` views."""
    sq, sk, hq, hkv, dim = 40, 56, 4, 2, 64
    q, k, v = _random_qkv(204, batch, sq, sk, hq, hkv, dim, dim)

    def bhsd_backed(x):
        root = np.ascontiguousarray(x.transpose(0, 2, 1, 3))
        return root.transpose(0, 2, 1, 3)

    scale = 1.0 / np.sqrt(dim)
    request = _dense_direct_request(
        batch, sq, sk, hq, hkv, dim, dim, algorithm=algorithm
    )
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
@_BOTH_KERNELS
@pytest.mark.parametrize("batch", [1, 2])
def test_dispatch_dense_packed_qkv_views(batch, algorithm):
    """Q/K/V as strided slices of one packed ``[B, S, Hq + 2*Hkv, D]`` buffer."""
    s, hq, hkv, dim = 48, 4, 2, 64
    q, k, v = _random_qkv(205, batch, s, s, hq, hkv, dim, dim)
    packed = np.concatenate([q, k, v], axis=2)
    pq = packed[:, :, :hq]
    pk = packed[:, :, hq : hq + hkv]
    pv = packed[:, :, hq + hkv :]
    scale = 1.0 / np.sqrt(dim)
    request = _dense_direct_request(batch, s, s, hq, hkv, dim, dim, algorithm=algorithm)
    actual = _run_dense_views(request, pq, pk, pv, scale=scale)
    expected = _windowed_reference(q, k, v, scale=scale, ctx=0, left=0, right=-1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-2, equal_nan=False)


# ---------------------------------------------------------------------
# Softmax statistics (LSE): natural-log log-sum-exp per (batch, head, query).
# ---------------------------------------------------------------------


def _launch_np(
    request,
    arrays,
    outputs,
    *,
    scale,
    overrides=None,
    spec=None,
    return_binding=False,
):
    """Upload each named host array, launch through the public binder, and
    return the downloaded arrays named in ``outputs`` (with the binding when
    ``return_binding``). ``overrides`` maps a name to ``(shape,
    element_strides)`` describing a view over the uploaded contiguous buffer
    (expanded, permuted or padded layouts). An explicit ``spec`` binds through
    ``bind_rdna_prefill_torch`` / ``bind_rdna_decode_torch`` instead of the
    dispatch-selected object."""
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
            if a.nbytes:
                rt.memcpy_h2d(ptrs[name], _host_bytes(a), a.nbytes)
            shape = tuple(a.shape)
            strides = tuple(s // a.itemsize for s in a.strides)
            if overrides and name in overrides:
                shape, strides = overrides[name]
            tensors[name] = _DeviceTensor(
                _ptr=ptrs[name],
                _shape=tuple(shape),
                _strides=tuple(strides),
                _dtype=str(a.dtype),
            )
        kwargs = dict(softmax_scale=scale, stream=stream, fence=False)
        if spec is None:
            binding = _dispatch_pinned(request).bind_torch(tensors, **kwargs)
        else:
            binding = _bind_explicit(request, spec, tensors, **kwargs)
        binding.launch()
        rt.stream_sync(stream)
        release_retained_for_stream(stream)
        for name in outputs:
            if arrays[name].nbytes:
                rt.memcpy_d2h(
                    _host_bytes(arrays[name]), ptrs[name], arrays[name].nbytes
                )
        got = {name: arrays[name].copy() for name in outputs}
        return (got, binding) if return_binding else got
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
@_BOTH_KERNELS
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
def test_dispatch_dense_lse(mask, sq, sk, left, right, use_sinks, algorithm):
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
        algorithm=algorithm,
        mask_type=mask,
        sliding_window=left,
        window_right=right,
        use_sinks=use_sinks,
        emit_lse=True,
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
@_BOTH_KERNELS
def test_dispatch_dense_lse_does_not_change_output(algorithm):
    batch, sq, sk, hq, hkv, dim = 2, 37, 53, 4, 2, 64
    q, k, v = _random_qkv(303, batch, sq, sk, hq, hkv, dim, dim)
    scale = 1.0 / np.sqrt(dim)
    base = _dense_direct_request(batch, sq, sk, hq, hkv, dim, dim, algorithm=algorithm)
    with_lse = _dense_direct_request(
        batch, sq, sk, hq, hkv, dim, dim, algorithm=algorithm, emit_lse=True
    )

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
@_BOTH_KERNELS
def test_dispatch_dense_lse_gapped_batch_stride_uses_runtime_stride(algorithm):
    """K/V and LSE preserve independent runtime batch strides."""
    batch, sq, sk, hq, hkv, dim, pad = 3, 32, 48, 4, 2, 64, 5
    q, k, v = _random_qkv(304, batch, sq, sk, hq, hkv, dim, dim)
    k_store = np.zeros((batch, sk + pad, hkv, dim), dtype=k.dtype)
    v_store = np.zeros_like(k_store)
    k_store[:, :sk], v_store[:, :sk] = k, v
    scale = 1.0 / np.sqrt(dim)
    request = _dense_direct_request(
        batch, sq, sk, hq, hkv, dim, dim, algorithm=algorithm, emit_lse=True
    )
    lse_store = np.full((batch, hq, sq + 3), np.nan, dtype=np.float32)
    out = np.full((batch, sq, hq, dim), np.nan, dtype=q.dtype)
    got = _launch_views_lse(
        request,
        q,
        k_store[:, :sk],
        v_store[:, :sk],
        out,
        lse_store[:, :, :sq],
        scale=scale,
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
        binding = _dispatch_pinned(request).bind_torch(
            tensors, softmax_scale=scale, stream=stream, fence=False
        )
        binding.launch()
        rt.stream_sync(stream)
        release_retained_for_stream(stream)
        lse_root = lse.base if lse.base is not None else lse
        rt.memcpy_d2h(_host_bytes(lse_root), uploaded[id(lse_root)], lse_root.nbytes)
        return np.ascontiguousarray(lse)
    finally:
        rt.sync()
        release_retained_for_stream(stream)
        for ptr in uploaded.values():
            rt.free(ptr)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize("use_sinks", [False, True])
def test_dispatch_ragged_lse_gqa(use_sinks, algorithm):
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
        arch=_ARCH,
        dtype="fp16",
        layout="ragged",
        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        use_sinks=use_sinks,
        emit_lse=True,
        algorithm=algorithm,
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


def _bias_reference(q, k, v, bias, scale, *, ctx, sinks=None):
    """FP64 ``(out, lse)`` with ``bias`` (broadcastable to ``[B, Hq, Sq, Sk]``)
    added to the scaled scores; ``ctx`` is the bottom-right causal offset or
    ``None`` for no mask."""
    _, sq, hq, _ = q.shape
    sk, hkv = k.shape[1], k.shape[2]
    q64, k64, v64 = (x.astype(np.float64) for x in (q, k, v))
    k64 = np.repeat(k64, hq // hkv, axis=2)
    v64 = np.repeat(v64, hq // hkv, axis=2)
    scores = np.einsum("bqhd,bkhd->bhqk", q64 * scale, k64) + bias
    if ctx is not None:
        qi = np.arange(sq).reshape(sq, 1)
        ki = np.arange(sk).reshape(1, sk)
        scores = np.where((ki <= qi + ctx).reshape(1, 1, sq, sk), scores, -np.inf)
    if sinks is not None:
        sink = np.broadcast_to(
            sinks.astype(np.float64).reshape(1, hq, 1, 1), scores.shape[:3] + (1,)
        )
        scores = np.concatenate([scores, sink], axis=-1)
    row_max = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - row_max)
    denom = e.sum(axis=-1, keepdims=True)
    p = (e / denom)[..., :sk]
    out = np.einsum("bhqk,bkhd->bqhd", p, v64).astype(np.float32)
    lse = (np.log(denom) + row_max)[..., 0].astype(np.float32)
    return out, lse


@pytest.mark.gpu
@_NEEDS_GPU
@pytest.mark.parametrize("path", ["fused", "merge"])
def test_decode_split_kv_merges_lse_with_sinks(path):
    """The split partials' LSE merge (fused or merge kernel) reproduces the
    whole-row output and natural-log LSE, sinks included, twice per binding."""
    batch, sq, sk, hq, hkv, dim = 2, 3, 9001, 8, 2, 64
    q, k, v = _random_qkv(311, batch, sq, sk, hq, hkv, dim, dim)
    sinks = np.random.default_rng(312).standard_normal(hq).astype(np.float16)
    scale = 1.0 / np.sqrt(dim)
    request = _dense_direct_request(
        batch,
        sq,
        sk,
        hq,
        hkv,
        dim,
        dim,
        algorithm=_DECODE,
        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        use_sinks=True,
        emit_lse=True,
    )
    request = dataclasses.replace(request, num_cus=_num_cus_for(request, path))
    out, lse = _bias_reference(q, k, v, 0.0, scale, ctx=sk - sq, sinks=sinks)

    def arrays():
        return {
            "q": q,
            "k": k,
            "v": v,
            "sinks": sinks,
            "out": np.full((batch, sq, hq, dim), np.nan, dtype=q.dtype),
            "lse": np.full((batch, hq, sq), np.nan, dtype=np.float32),
        }

    got, binding = _launch_np(
        request, arrays(), ("out", "lse"), scale=scale, return_binding=True
    )
    _assert_decode_path(binding, path)
    np.testing.assert_allclose(got["out"], out, rtol=0, atol=2e-2)
    _assert_lse_close(got["lse"], lse)
    again = _launch_np(request, arrays(), ("out", "lse"), scale=scale)
    np.testing.assert_array_equal(again["out"], got["out"])
    np.testing.assert_array_equal(again["lse"], got["lse"])


def _make_bias(seed, shape, bias_dtype):
    values = np.random.default_rng(seed).standard_normal(shape) * 2.0
    return values.astype(np.float32 if bias_dtype == "f32" else np.float16)


def _bias_request(batch, sq, sk, hq, hkv, dim, bias_dtype, mask, *, algorithm, **extra):
    return _dense_direct_request(
        batch,
        sq,
        sk,
        hq,
        hkv,
        dim,
        dim,
        algorithm=algorithm,
        mask_type=mask,
        use_attn_bias=True,
        attn_bias_dtype=bias_dtype,
        **extra,
    )


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize("bias_dtype", ["f32", "q"])
@pytest.mark.parametrize(
    "mask", [AttentionMaskType.NO_MASK, AttentionMaskType.BOTTOM_RIGHT_CAUSAL]
)
@pytest.mark.parametrize(
    "bias_shape", ["1,1,Q,K", "B,1,Q,K", "B,H,Q,K", "B,H,1,K", "1,H,1,K"]
)
def test_dispatch_dense_attn_bias_shapes(bias_dtype, mask, bias_shape, algorithm):
    batch, sq, sk, hq, hkv, dim = 2, 37, 53, 4, 2, 64
    q, k, v = _random_qkv(401, batch, sq, sk, hq, hkv, dim, dim)
    scale = 1.0 / np.sqrt(dim)
    dims = {"B": batch, "H": hq, "Q": sq, "K": sk, "1": 1}
    shape = tuple(dims[d] for d in bias_shape.split(","))
    bias = _make_bias(402, shape, bias_dtype)
    request = _bias_request(
        batch,
        sq,
        sk,
        hq,
        hkv,
        dim,
        bias_dtype,
        mask,
        algorithm=algorithm,
        emit_lse=True,
    )
    arrays = {
        "q": q,
        "k": k,
        "v": v,
        "attn_bias": bias,
        "out": np.full((batch, sq, hq, dim), np.nan, dtype=q.dtype),
        "lse": np.full((batch, hq, sq), np.nan, dtype=np.float32),
    }
    got = _launch_np(request, arrays, ("out", "lse"), scale=scale)
    ctx = sk - sq if mask == AttentionMaskType.BOTTOM_RIGHT_CAUSAL else None
    out, lse = _bias_reference(q, k, v, bias.astype(np.float64), scale, ctx=ctx)
    np.testing.assert_allclose(got["out"], out, rtol=0, atol=2e-2)
    _assert_lse_close(got["lse"], lse)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
def test_dispatch_dense_attn_bias_changes_output_and_composes_with_sinks(algorithm):
    batch, sq, sk, hq, hkv, dim = 2, 33, 47, 4, 2, 64
    q, k, v = _random_qkv(403, batch, sq, sk, hq, hkv, dim, dim)
    scale = 1.0 / np.sqrt(dim)
    sinks = np.random.default_rng(404).standard_normal(hq).astype(np.float16)
    bias = _make_bias(405, (batch, hq, sq, sk), "f32")
    mask = AttentionMaskType.BOTTOM_RIGHT_CAUSAL
    request = _bias_request(
        batch, sq, sk, hq, hkv, dim, "f32", mask, algorithm=algorithm, use_sinks=True
    )
    arrays = {
        "q": q,
        "k": k,
        "v": v,
        "attn_bias": bias,
        "sinks": sinks,
        "out": np.full((batch, sq, hq, dim), np.nan, dtype=q.dtype),
    }
    got = _launch_np(request, arrays, ("out",), scale=scale)["out"]
    out, _ = _bias_reference(
        q, k, v, bias.astype(np.float64), scale, ctx=sk - sq, sinks=sinks
    )
    np.testing.assert_allclose(got, out, rtol=0, atol=2e-2)
    plain, _ = _bias_reference(q, k, v, 0.0, scale, ctx=sk - sq, sinks=sinks)
    assert np.abs(plain - out).max() > 5e-2


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize("bias_dtype", ["f32", "q"])
@pytest.mark.parametrize("view", ["expanded", "permuted", "row_padded"])
def test_dispatch_dense_attn_bias_strided_views(bias_dtype, view, algorithm):
    batch, sq, sk, hq, hkv, dim, pad = 2, 32, 48, 4, 2, 64, 8
    q, k, v = _random_qkv(406, batch, sq, sk, hq, hkv, dim, dim)
    scale = 1.0 / np.sqrt(dim)
    if view == "expanded":
        storage = _make_bias(407, (sq, sk), bias_dtype)
        shape, strides = (batch, hq, sq, sk), (0, 0, sk, 1)
        full = np.broadcast_to(storage.astype(np.float64), shape)
    elif view == "permuted":
        # [B, Sq, H, Sk] storage viewed as [B, H, Sq, Sk].
        storage = _make_bias(407, (batch, sq, hq, sk), bias_dtype)
        shape, strides = (batch, hq, sq, sk), (sq * hq * sk, sk, hq * sk, 1)
        full = storage.astype(np.float64).transpose(0, 2, 1, 3)
    else:
        storage = _make_bias(407, (batch, hq, sq, sk + pad), bias_dtype)
        shape = (batch, hq, sq, sk)
        strides = (hq * sq * (sk + pad), sq * (sk + pad), sk + pad, 1)
        full = storage.astype(np.float64)[..., :sk]
    mask = AttentionMaskType.BOTTOM_RIGHT_CAUSAL
    request = _bias_request(
        batch, sq, sk, hq, hkv, dim, bias_dtype, mask, algorithm=algorithm
    )
    arrays = {
        "q": q,
        "k": k,
        "v": v,
        "attn_bias": storage,
        "out": np.full((batch, sq, hq, dim), np.nan, dtype=q.dtype),
    }
    got = _launch_np(
        request,
        arrays,
        ("out",),
        scale=scale,
        overrides={"attn_bias": (shape, strides)},
    )["out"]
    out, _ = _bias_reference(q, k, v, full, scale, ctx=sk - sq)
    np.testing.assert_allclose(got, out, rtol=0, atol=2e-2)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize("bias_dtype", ["f32", "q"])
def test_dispatch_dense_attn_bias_gapped_batch_stride_uses_runtime_stride(
    bias_dtype, algorithm
):
    """K/V and batched bias preserve their independent runtime batch strides."""
    batch, sq, sk, hq, hkv, dim, pad = 3, 32, 48, 4, 2, 64, 5
    q, k, v = _random_qkv(408, batch, sq, sk, hq, hkv, dim, dim)
    k_store = np.zeros((batch, sk + pad, hkv, dim), dtype=k.dtype)
    v_store = np.zeros_like(k_store)
    k_store[:, :sk], v_store[:, :sk] = k, v
    kv_view = (
        (batch, sk, hkv, dim),
        ((sk + pad) * hkv * dim, hkv * dim, dim, 1),
    )
    bias = _make_bias(409, (batch, hq, sq, sk), bias_dtype)
    scale = 1.0 / np.sqrt(dim)
    mask = AttentionMaskType.NO_MASK
    request = _bias_request(
        batch, sq, sk, hq, hkv, dim, bias_dtype, mask, algorithm=algorithm
    )
    arrays = {
        "q": q,
        "k": k_store,
        "v": v_store,
        "attn_bias": bias,
        "out": np.full((batch, sq, hq, dim), np.nan, dtype=q.dtype),
    }
    got = _launch_np(
        request,
        arrays,
        ("out",),
        scale=scale,
        overrides={"k": kv_view, "v": kv_view},
    )["out"]
    out, _ = _bias_reference(q, k, v, bias.astype(np.float64), scale, ctx=None)
    np.testing.assert_allclose(got, out, rtol=0, atol=2e-2)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize("bias_dtype", ["f32", "q"])
@pytest.mark.parametrize("causal", [False, True])
def test_dispatch_ragged_attn_bias_uses_within_sequence_positions(
    bias_dtype, causal, algorithm
):
    """Ragged bias is ``[B, H|1, max_q, max_k]``; each sequence reads it at its
    own within-sequence (query, key) positions, not packed offsets."""
    hq, hkv, dim = 4, 2, 64
    q_lengths = (5, 17, 1, 25)
    k_lengths = (9, 17, 5, 40)
    batch = len(q_lengths)
    mq, mk = max(q_lengths), max(k_lengths)
    total_q, total_k = sum(q_lengths), sum(k_lengths)
    rng = np.random.default_rng(410)
    q = rng.standard_normal((total_q, hq, dim)).astype(np.float16)
    k = rng.standard_normal((total_k, hkv, dim)).astype(np.float16)
    v = rng.standard_normal((total_k, hkv, dim)).astype(np.float16)
    bias = _make_bias(411, (batch, 1, mq, mk), bias_dtype)
    cu_q = np.concatenate([[0], np.cumsum(q_lengths)]).astype(np.int32)
    cu_k = np.concatenate([[0], np.cumsum(k_lengths)]).astype(np.int32)
    scale = 1.0 / np.sqrt(dim)
    mask = (
        AttentionMaskType.BOTTOM_RIGHT_CAUSAL if causal else AttentionMaskType.NO_MASK
    )
    request = AttentionRequest(
        batch=batch,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=mq,
        seqlen_k=mk,
        hdim_q=dim,
        hdim_v=dim,
        arch=_ARCH,
        dtype="fp16",
        layout="ragged",
        mask_type=mask,
        use_attn_bias=True,
        attn_bias_dtype=bias_dtype,
        algorithm=algorithm,
    )
    arrays = {
        "q": q,
        "k": k,
        "v": v,
        "attn_bias": bias,
        "out": np.full((total_q, hq, dim), np.nan, dtype=q.dtype),
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
    }
    got = _launch_np(request, arrays, ("out",), scale=scale)["out"]
    for i in range(batch):
        qs, qe, ks, ke = cu_q[i], cu_q[i + 1], cu_k[i], cu_k[i + 1]
        lq, lk = int(qe - qs), int(ke - ks)
        out, _ = _bias_reference(
            q[None, qs:qe],
            k[None, ks:ke],
            v[None, ks:ke],
            bias[i : i + 1, :, :lq, :lk].astype(np.float64),
            scale,
            ctx=lk - lq if causal else None,
        )
        np.testing.assert_allclose(got[qs:qe], out[0], rtol=0, atol=2e-2)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
def test_dispatch_bhsd_head96_noncausal_window_with_lse(algorithm):
    """Exercise BHSD plus independently padded Q/K/V/O/LSE strides (row pads
    keep the 16-byte row alignment the kernels load with)."""
    from benchmarks.gfx1151.attention.benchmark_sdpa import DeviceBuffers
    from benchmarks.gfx1151.attention.cases import CaseInputs

    batch, heads_q, heads_kv, sq, sk, dim = 2, 4, 2, 17, 23, 96
    rng = np.random.default_rng(12710)
    q = (rng.standard_normal((batch, heads_q, sq, dim)) * 0.2).astype(np.float16)
    k = (rng.standard_normal((batch, heads_kv, sk, dim)) * 0.2).astype(np.float16)
    v = (rng.standard_normal((batch, heads_kv, sk, dim)) * 0.2).astype(np.float16)
    scale = 1.0 / math.sqrt(dim)
    kh = np.repeat(k.astype(np.float32), heads_q // heads_kv, axis=1)
    vh = np.repeat(v.astype(np.float32), heads_q // heads_kv, axis=1)
    scores = np.einsum("bhsd,bhkd->bhsk", q.astype(np.float32), kh) * scale
    qpos = np.arange(sq)[:, None]
    kpos = np.arange(sk)[None, :]
    visible = (kpos >= qpos - 3) & (kpos <= qpos + 5)
    masked = np.where(visible[None, None], scores, -np.inf)
    row_max = masked.max(axis=-1, keepdims=True)
    weights = np.exp(masked - row_max)
    denom = weights.sum(axis=-1, keepdims=True)
    expected_out = np.einsum("bhsk,bhkd->bhsd", weights / denom, vh)
    expected_lse = (row_max + np.log(denom)).astype(np.float32)

    def padded(array, extra):
        storage = np.zeros((*array.shape[:-1], array.shape[-1] + extra), array.dtype)
        view = storage[..., : array.shape[-1]]
        view[...] = array
        return storage, view

    q_storage, q_view = padded(q, 8)
    k_storage, k_view = padded(k, 16)
    v_storage, v_view = padded(v, 24)
    out_storage = np.full((batch, heads_q, sq, dim + 40), np.nan, np.float16)
    out_view = out_storage[..., :dim]
    lse_storage = np.full((batch, heads_q, sq, 3), np.nan, np.float32)
    lse_view = lse_storage[..., :1]

    rt = Runtime()
    buffers = DeviceBuffers(rt, CaseInputs(q=q, k=k, v=v))
    for name, storage in (
        ("q", q_storage),
        ("k", k_storage),
        ("v", v_storage),
        ("rocke_out", out_storage),
    ):
        rt.free(buffers.ptrs[name])
        buffers.add(name, storage)
    buffers.add("lse", lse_storage)
    try:
        request = AttentionRequest(
            batch=batch,
            nhead_q=heads_q,
            nhead_k=heads_kv,
            seqlen_q=sq,
            seqlen_k=sk,
            hdim_q=dim,
            hdim_v=dim,
            arch=_ARCH,
            dtype="fp16",
            layout="dense",
            tensor_layout="bhsd",
            mask_type=AttentionMaskType.SLIDING_WINDOW,
            window_left=3,
            window_right=5,
            emit_lse=True,
            algorithm=algorithm,
        )
        tensors = {
            "q": _view_as(buffers, "q", q_view),
            "k": _view_as(buffers, "k", k_view),
            "v": _view_as(buffers, "v", v_view),
            "out": _view_as(buffers, "rocke_out", out_view),
            "lse": _view_as(buffers, "lse", lse_view),
        }
        result = _dispatch_pinned(request)
        output = result.bind_torch(tensors, softmax_scale=scale).launch()
        assert output is tensors["out"]
        out_read = buffers.read_output("rocke_out")
        np.testing.assert_allclose(out_read[..., :dim], expected_out, rtol=0, atol=2e-2)
        # Columns past the head width belong to the caller.
        assert np.isnan(out_read[..., dim:]).all()
        np.testing.assert_allclose(
            buffers.read_output("lse")[..., :1],
            expected_lse,
            rtol=0,
            atol=2e-2,
        )
    finally:
        buffers.close()


# ---------------------------------------------------------------------
# Runtime head widths: one bucket object serves every multiple-of-8 width
# inside it (the decode bucket from its own minimum up), and a
# ``runtime_head_dims`` request may give V its own head count.
# ---------------------------------------------------------------------


def _rtdim_qkv(seed, batch, sq, sk, hq, hkv, hv, dq, dv):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((batch, sq, hq, dq)).astype(np.float16),
        rng.standard_normal((batch, sk, hkv, dq)).astype(np.float16),
        rng.standard_normal((batch, sk, hv, dv)).astype(np.float16),
    )


def _rtdim_expected(q, k, v, scale, *, ctx, left, right, bias=None):
    """FP64 ``(out, lse)`` with K and V expanded to the query heads first, so
    the shared references see one head count."""
    hq = q.shape[2]
    k_rep = np.repeat(k, hq // k.shape[2], axis=2)
    v_rep = np.repeat(v, hq // v.shape[2], axis=2)
    if bias is not None:
        return _bias_reference(q, k_rep, v_rep, bias.astype(np.float64), scale, ctx=ctx)
    out = _windowed_reference(
        q, k_rep, v_rep, scale=scale, ctx=ctx, left=left, right=right
    )
    lse = _lse_reference(q, k_rep, scale, ctx=ctx, left=left, right=right)
    return out, lse


def _rtdim_request(batch, sq, sk, hq, hkv, dq, dv, *, algorithm, **extra):
    return _dense_direct_request(
        batch,
        sq,
        sk,
        hq,
        hkv,
        dq,
        dv,
        algorithm=algorithm,
        emit_lse=True,
        runtime_head_dims=True,
        **extra,
    )


def _run_rtdim(request, q, k, v, *, scale, spec=None, bias=None, overrides=None):
    batch, sq, hq, _ = q.shape
    arrays = {
        "q": q,
        "k": k,
        "v": v,
        "out": np.full((batch, sq, hq, v.shape[-1]), np.nan, dtype=np.float16),
        "lse": np.full((batch, hq, sq), np.nan, dtype=np.float32),
    }
    if bias is not None:
        arrays["attn_bias"] = bias
    return _launch_np(
        request, arrays, ("out", "lse"), scale=scale, overrides=overrides, spec=spec
    )


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize(
    "bucket,dim",
    [
        (64, 8),
        (64, 16),
        (64, 56),
        (96, 40),
        (96, 96),
        (128, 16),
        (128, 72),
        (128, 112),
        (128, 128),
        (256, 136),
        (256, 200),
        (256, 248),
    ],
)
def test_runtime_head_dims_serve_every_width_in_a_bucket(bucket, dim, algorithm):
    """An explicit bucket object at widths well below its bucket."""
    if algorithm == _DECODE and not _decode_bucket_serves(bucket, dim):
        pytest.skip(f"the decode bucket {bucket} does not serve hdim_q={dim}")
    batch, sq, sk, hq, hkv = 2, 37, 53, 8, 4
    q, k, v = _rtdim_qkv(501, batch, sq, sk, hq, hkv, hkv, dim, dim)
    scale = 1.0 / np.sqrt(dim)
    request = _rtdim_request(batch, sq, sk, hq, hkv, dim, dim, algorithm=algorithm)
    got = _run_rtdim(
        request, q, k, v, scale=scale, spec=_bucket_spec(algorithm, bucket)
    )
    out, lse = _rtdim_expected(q, k, v, scale, ctx=0, left=0, right=-1)
    np.testing.assert_allclose(got["out"], out, rtol=0, atol=2e-2)
    _assert_lse_close(got["lse"], lse)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize(
    "dq,dv",
    [(80, 48), (48, 112), (96, 48), (176, 80), (128, 64), (192, 128), (8, 24)],
)
def test_runtime_head_dims_unequal_widths(dq, dv, algorithm):
    _require_served(algorithm, dq, dv)
    batch, sq, sk, hq, hkv = 2, 37, 53, 8, 4
    q, k, v = _rtdim_qkv(502, batch, sq, sk, hq, hkv, hkv, dq, dv)
    scale = 1.0 / np.sqrt(dq)
    request = _rtdim_request(batch, sq, sk, hq, hkv, dq, dv, algorithm=algorithm)
    got = _run_rtdim(request, q, k, v, scale=scale)
    out, lse = _rtdim_expected(q, k, v, scale, ctx=0, left=0, right=-1)
    np.testing.assert_allclose(got["out"], out, rtol=0, atol=2e-2)
    _assert_lse_close(got["lse"], lse)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize("hv", [1, 2, 4, 8])
def test_runtime_head_dims_independent_v_heads(hv, algorithm):
    """Hq=8, Hk=4 with Hv in {1, 2, Hk, Hq}: V heads follow Hq/Hv, not Hq/Hk."""
    batch, sq, sk, hq, hkv, dq, dv = 2, 33, 47, 8, 4, 64, 48
    q, k, v = _rtdim_qkv(503, batch, sq, sk, hq, hkv, hv, dq, dv)
    scale = 1.0 / np.sqrt(dq)
    request = _rtdim_request(batch, sq, sk, hq, hkv, dq, dv, algorithm=algorithm)
    got = _run_rtdim(request, q, k, v, scale=scale)
    out, lse = _rtdim_expected(q, k, v, scale, ctx=0, left=0, right=-1)
    np.testing.assert_allclose(got["out"], out, rtol=0, atol=2e-2)
    _assert_lse_close(got["lse"], lse)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize("mask", ["causal", "window"])
def test_runtime_head_dims_causal_and_window(mask, algorithm):
    hq, hkv, hv, dim = 8, 4, 2, 80
    if mask == "causal":
        sq, sk, left, right = 33, 49, 0, 0
        ctx = sk - sq
        extra = dict(mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL)
    else:
        sq, sk, left, right, ctx = 33, 41, 24, 8, 0
        extra = dict(
            mask_type=AttentionMaskType.SLIDING_WINDOW,
            sliding_window=left,
            window_right=right,
        )
    q, k, v = _rtdim_qkv(504, 2, sq, sk, hq, hkv, hv, dim, dim)
    scale = 1.0 / np.sqrt(dim)
    request = _rtdim_request(2, sq, sk, hq, hkv, dim, dim, algorithm=algorithm, **extra)
    got = _run_rtdim(request, q, k, v, scale=scale)
    out, lse = _rtdim_expected(q, k, v, scale, ctx=ctx, left=left, right=right)
    np.testing.assert_allclose(got["out"], out, rtol=0, atol=2e-2)
    _assert_lse_close(got["lse"], lse)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
def test_runtime_head_dims_attn_bias_f32_with_lse(algorithm):
    batch, sq, sk, hq, hkv, hv, dim = 2, 37, 53, 8, 4, 4, 48
    q, k, v = _rtdim_qkv(505, batch, sq, sk, hq, hkv, hv, dim, dim)
    bias = _make_bias(506, (batch, hq, sq, sk), "f32")
    scale = 1.0 / np.sqrt(dim)
    request = _rtdim_request(
        batch,
        sq,
        sk,
        hq,
        hkv,
        dim,
        dim,
        algorithm=algorithm,
        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        use_attn_bias=True,
        attn_bias_dtype="f32",
    )
    got = _run_rtdim(request, q, k, v, scale=scale, bias=bias)
    out, lse = _rtdim_expected(q, k, v, scale, ctx=sk - sq, left=0, right=0, bias=bias)
    np.testing.assert_allclose(got["out"], out, rtol=0, atol=2e-2)
    _assert_lse_close(got["lse"], lse)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
def test_runtime_head_dims_never_touch_columns_past_the_widths(algorithm):
    """Q/K/V live in bucket-wide rows whose padding is NaN, and O in rows
    whose padding holds a sentinel: no padding is read into the result and
    no output column past ``hdim_v`` is written."""
    batch, sq, sk, hq, hkv, hv, dq, dv, bucket = 2, 37, 53, 8, 4, 2, 88, 48, 128
    q, k, v = _rtdim_qkv(507, batch, sq, sk, hq, hkv, hv, dq, dv)

    def padded(x, fill):
        store = np.full(x.shape[:3] + (bucket,), fill, dtype=np.float16)
        store[..., : x.shape[-1]] = x
        return store

    def view(store, dim):
        strides = tuple(s // store.itemsize for s in store.strides)
        return store.shape[:3] + (dim,), strides

    sentinel = 7.0
    arrays = {
        "q": padded(q, np.nan),
        "k": padded(k, np.nan),
        "v": padded(v, np.nan),
        "out": np.full((batch, sq, hq, bucket), sentinel, dtype=np.float16),
        "lse": np.full((batch, hq, sq), np.nan, dtype=np.float32),
    }
    overrides = {
        "q": view(arrays["q"], dq),
        "k": view(arrays["k"], dq),
        "v": view(arrays["v"], dv),
        "out": view(arrays["out"], dv),
    }
    scale = 1.0 / np.sqrt(dq)
    request = _rtdim_request(batch, sq, sk, hq, hkv, dq, dv, algorithm=algorithm)
    got = _launch_np(
        request,
        arrays,
        ("out", "lse"),
        scale=scale,
        overrides=overrides,
        spec=_bucket_spec(algorithm, bucket),
    )
    out, lse = _rtdim_expected(q, k, v, scale, ctx=0, left=0, right=-1)
    np.testing.assert_allclose(got["out"][..., :dv], out, rtol=0, atol=2e-2)
    np.testing.assert_array_equal(
        got["out"][..., dv:], np.full_like(got["out"][..., dv:], sentinel)
    )
    _assert_lse_close(got["lse"], lse)


@pytest.mark.gpu
@_NEEDS_GPU
@_BOTH_KERNELS
@pytest.mark.parametrize("bucket,dim", [(128, 80), (64, 16), (256, 136)])
def test_runtime_head_dims_rows_ending_at_the_allocation(bucket, dim, algorithm):
    """Contiguous BHSD K/V allocated to exactly ``B*H*S*d`` elements: the last
    row of the last head ends at the allocation end, so head-dim tiles past
    ``d`` must be bounded by the K/V extent rather than read past the
    tensor."""
    if algorithm == _DECODE and not _decode_bucket_serves(bucket, dim):
        pytest.skip(f"the decode bucket {bucket} does not serve hdim_q={dim}")
    batch, sq, sk, hq, hkv = 2, 37, 53, 8, 4
    q, k, v = _rtdim_qkv(508, batch, sq, sk, hq, hkv, hkv, dim, dim)

    def bhsd(x):
        return np.ascontiguousarray(x.transpose(0, 2, 1, 3))

    arrays = {
        "q": bhsd(q),
        "k": bhsd(k),
        "v": bhsd(v),
        "out": np.full((batch, hq, sq, dim), np.nan, dtype=np.float16),
        "lse": np.full((batch, hq, sq), np.nan, dtype=np.float32),
    }
    assert arrays["k"].size == batch * hkv * sk * dim
    scale = 1.0 / np.sqrt(dim)
    request = _rtdim_request(
        batch, sq, sk, hq, hkv, dim, dim, algorithm=algorithm, tensor_layout="bhsd"
    )
    got = _launch_np(
        request,
        arrays,
        ("out", "lse"),
        scale=scale,
        spec=_bucket_spec(algorithm, bucket),
    )
    out, lse = _rtdim_expected(q, k, v, scale, ctx=0, left=0, right=-1)
    np.testing.assert_allclose(got["out"].transpose(0, 2, 1, 3), out, rtol=0, atol=2e-2)
    _assert_lse_close(got["lse"], lse)


# ---------------------------------------------------------------------
# Structural launch-safety gate: no GPU, no torch, no compile, no
# ``cu_seqlens``/``seqused_k``/``block_table`` content reads -- only
# shape/dtype/stride/alignment.
# ---------------------------------------------------------------------


def _fake_tensor(shape, dtype="float16", strides=None, device="cuda:0", ptr=0x1000):
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
        data_ptr=lambda: ptr,
    )


def _meta_spec(
    kind,
    *,
    head=64,
    head_v=0,
    bias=None,
    kv_dtype="",
    layout="dense",
    page_block_size=0,
    softcap=False,
    sinks=False,
):
    """A ``kind`` ("prefill" | "decode") fp16 object; the decode bucket is the
    wider of ``head``/``head_v``."""
    if kind == "prefill":
        from kernels.gfx1151.wmma_fmha_fwd_v2 import WmmaFmhaFwdV2Spec

        return WmmaFmhaFwdV2Spec(
            dtype="fp16",
            head_dim=head,
            head_dim_v=head_v,
            bias=bias,
            kv_dtype=kv_dtype,
            layout=layout,
            page_block_size=page_block_size,
            use_softcap=softcap,
            use_sinks=sinks,
        )
    from kernels.gfx1151.wmma_fmha_decode import (
        WmmaFmhaDecodeSpec,
        resolve_wmma_fmha_decode_spec,
    )

    return resolve_wmma_fmha_decode_spec(
        WmmaFmhaDecodeSpec(
            head_size=max(head, head_v),
            dtype="fp16",
            bias=bias or "",
            kv_dtype=kv_dtype,
            layout=layout,
            page_block_size=page_block_size,
            softcap=softcap,
            sinks=sinks,
        ),
        arch="gfx1151",
    )


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
        num_cus=40,
    )
    base.update(kw)
    return AttentionRequest(**base)


def _collect(request, spec, tensors):
    from dispatch.attention.bindings import _rdna_validate_and_collect

    return _rdna_validate_and_collect(request, spec, tensors)


def _bind(kind, request, spec, tensors, **kwargs):
    from dispatch.attention.bindings import (
        bind_rdna_decode_torch,
        bind_rdna_prefill_torch,
    )

    bind = bind_rdna_prefill_torch if kind == "prefill" else bind_rdna_decode_torch
    return bind(request, spec, tensors, **kwargs)


def _dense_tensors(batch=2, seq=32, hq=4, hk=2, dq=64, dv=64, row=0):
    """Contiguous BSHD tensors, or (``row`` > 0) head rows padded to ``row``
    elements so any width keeps 16-byte aligned rows."""

    def make(heads, dim):
        if not row:
            return _fake_tensor((batch, seq, heads, dim))
        return _fake_tensor(
            (batch, seq, heads, dim), strides=(seq * heads * row, heads * row, row, 1)
        )

    return {
        "q": make(hq, dq),
        "k": make(hk, dq),
        "v": make(hk, dv),
        "out": make(hq, dv),
    }


def _ragged_tensors(total_q=10, total_k=12):
    q = _fake_tensor((total_q, 4, 64))
    k = _fake_tensor((total_k, 2, 64))
    cu = _fake_tensor((3,), dtype="int32")
    return {
        "q": q,
        "k": k,
        "v": k,
        "out": q,
        "cu_seqlens_q": cu,
        "cu_seqlens_k": cu,
    }


@pytest.mark.parametrize("kind", ["prefill", "decode"])
class TestMetadataSafety:
    """Direct ``_rdna_validate_and_collect`` and binder calls with both the
    prefill and the decode object -- no GPU, no torch, no compile: every
    rejected case raises before the binder reaches its launcher cache, and
    accepted cases only collect arguments."""

    def test_accepts_independent_dense_batch_strides(self, kind):
        tensors = _dense_tensors()
        gapped_k = _fake_tensor(
            (2, 32, 2, 64), strides=(32 * 2 * 64 + 64, 2 * 64, 64, 1)
        )
        values = _collect(
            _dense_request(),
            _meta_spec(kind),
            {**tensors, "k": gapped_k, "v": gapped_k},
        )
        assert values["stride_k_batch"] == 32 * 2 * 64 + 64
        assert values["stride_q_batch"] == 32 * 4 * 64
        assert (values["head_dim_q"], values["head_dim_v"]) == (64, 64)
        assert values["num_v_heads"] == 2

    def test_lse_tensor_is_required_and_validated(self, kind):
        request = _dense_request(emit_lse=True)
        spec = _meta_spec(kind)
        base = _dense_tensors()
        with pytest.raises(ValueError, match="missing RDNA attention tensors: lse"):
            _collect(request, spec, base)
        good = _fake_tensor((2, 4, 32), dtype="float32")
        values = _collect(request, spec, {**base, "lse": good})
        assert values["LSE"] is good
        assert values["stride_lse_head"] == 32
        assert values["stride_lse_token"] == 1
        assert values["write_lse"] == 1
        for lse in (
            _fake_tensor((2, 4, 32), dtype="float16"),
            _fake_tensor((2, 32, 4), dtype="float32"),
            _fake_tensor((2, 4, 31), dtype="float32"),
        ):
            with pytest.raises(ValueError):
                _collect(request, spec, {**base, "lse": lse})

    def test_packed_lse_layouts(self, kind):
        request = _dense_request(layout="ragged", emit_lse=True)
        spec = _meta_spec(kind, layout="ragged")
        tensors = _ragged_tensors()
        for shape, expected in (
            ((4, 10), (1, 10)),
            ((10, 4), (4, 1)),
            ((10, 4, 1), (4, 1)),
        ):
            lse = _fake_tensor(shape, dtype="float32")
            values = _collect(request, spec, {**tensors, "lse": lse})
            assert (values["stride_lse_token"], values["stride_lse_head"]) == expected
            assert values["stride_lse_batch"] == 0
        with pytest.raises(ValueError, match="packed"):
            _collect(
                request,
                spec,
                {**tensors, "lse": _fake_tensor((10, 5), dtype="float32")},
            )

    def test_attn_bias_is_required_and_validated(self, kind):
        request = _dense_request(use_attn_bias=True)
        spec = _meta_spec(kind, bias="f32")
        base = _dense_tensors()
        with pytest.raises(ValueError, match=r"requires tensors\['attn_bias'\]"):
            _collect(request, spec, base)
        bad = (
            _fake_tensor((2, 4, 32, 32), dtype="float16"),
            _fake_tensor((2, 4, 32), dtype="float32"),
            _fake_tensor((2, 4, 32, 31), dtype="float32"),
            _fake_tensor((3, 4, 32, 32), dtype="float32"),
            _fake_tensor((2, 3, 32, 32), dtype="float32"),
            _fake_tensor((2, 4, 16, 32), dtype="float32"),
            _fake_tensor((2, 4, 32, 32), dtype="float32", strides=(4096, 1024, 32, 2)),
        )
        for bias in bad:
            with pytest.raises(ValueError):
                _collect(request, spec, {**base, "attn_bias": bias})

    def test_attn_bias_strides_and_broadcast_values(self, kind):
        request = _dense_request(use_attn_bias=True)
        spec = _meta_spec(kind, bias="f32")
        base = _dense_tensors()
        cases = (
            ((2, 4, 32, 32), None, (4 * 32 * 32, 32 * 32, 32)),
            ((1, 1, 32, 32), (777, 888, 32, 1), (0, 0, 32)),
            ((2, 1, 32, 32), None, (32 * 32, 0, 32)),
            ((2, 4, 1, 32), None, (4 * 32, 32, 0)),
            ((2, 4, 32, 32), (0, 0, 32, 1), (0, 0, 32)),
        )
        for shape, strides, expected in cases:
            bias = _fake_tensor(shape, dtype="float32", strides=strides)
            values = _collect(request, spec, {**base, "attn_bias": bias})
            assert values["attn_bias_ptr"] is bias
            assert (
                values["bias_stride_b"],
                values["bias_stride_h"],
                values["bias_stride_q"],
            ) == expected

    def test_attn_bias_q_dtype_follows_query_dtype(self, kind):
        request = _dense_request(use_attn_bias=True, attn_bias_dtype="q")
        spec = _meta_spec(kind, bias="q")
        base = _dense_tensors()
        good = _fake_tensor((2, 4, 32, 32), dtype="float16")
        _collect(request, spec, {**base, "attn_bias": good})
        with pytest.raises(ValueError, match="attn_bias dtype"):
            _collect(
                request,
                spec,
                {**base, "attn_bias": _fake_tensor((2, 4, 32, 32), dtype="float32")},
            )

    def test_overlapping_dense_out_batch_stride_is_rejected(self, kind):
        overlapping = _fake_tensor((2, 32, 4, 64), strides=(16 * 4 * 64, 4 * 64, 64, 1))
        with pytest.raises(ValueError, match="strides overlap"):
            _bind(
                kind,
                _dense_request(),
                _meta_spec(kind),
                {**_dense_tensors(), "out": overlapping},
            )

    def test_unequal_head_dims_follow_v_dim(self, kind):
        request = _dense_request(hdim_q=128, hdim_v=64)
        spec = _meta_spec(kind, head=128, head_v=64)
        tensors = _dense_tensors(dq=128, dv=64)
        values = _collect(request, spec, tensors)
        assert values["stride_o_token"] == 4 * 64
        assert values["stride_v_token"] == 2 * 64
        assert (values["head_dim_q"], values["head_dim_v"]) == (128, 64)
        with pytest.raises(ValueError, match="out must have"):
            _collect(request, spec, {**tensors, "out": tensors["q"]})
        with pytest.raises(ValueError, match="v must have"):
            _collect(request, spec, {**tensors, "v": tensors["k"]})

    def test_runtime_head_dims_collect_widths_and_v_heads(self, kind):
        spec = _meta_spec(kind, head=128)
        request = _dense_request(
            nhead_q=8, nhead_k=4, hdim_q=80, hdim_v=48, runtime_head_dims=True
        )
        q = _fake_tensor((2, 32, 8, 80))
        k = _fake_tensor((2, 32, 4, 80))
        out = _fake_tensor((2, 32, 8, 48))
        for hv in (1, 2, 4, 8):
            v = _fake_tensor((2, 32, hv, 48))
            values = _collect(request, spec, {"q": q, "k": k, "v": v, "out": out})
            assert (values["head_dim_q"], values["head_dim_v"]) == (80, 48)
            assert values["num_v_heads"] == hv
            assert values["num_kv_heads"] == 4
            assert values["stride_v_token"] == hv * 48
        with pytest.raises(ValueError, match="must divide"):
            _collect(
                request,
                spec,
                {"q": q, "k": k, "v": _fake_tensor((2, 32, 3, 48)), "out": out},
            )
        # BHSD reads the V head count from axis 1.
        bhsd = _dense_request(
            nhead_q=8,
            nhead_k=4,
            hdim_q=80,
            hdim_v=80,
            tensor_layout="bhsd",
            runtime_head_dims=True,
        )
        values = _collect(
            bhsd,
            spec,
            {
                "q": _fake_tensor((2, 8, 32, 80)),
                "k": _fake_tensor((2, 4, 32, 80)),
                "v": _fake_tensor((2, 2, 32, 80)),
                "out": _fake_tensor((2, 8, 32, 80)),
            },
        )
        assert values["num_v_heads"] == 2

    def test_decode_needs_nested_k_and_v_head_counts(self, kind):
        """Hq=12, Hk=4, Hv=6: both divide Hq, but neither divides the other,
        which only the decode kernel's pack groups require."""
        spec = _meta_spec(kind, head=128)
        request = _dense_request(
            nhead_q=12, nhead_k=4, hdim_q=80, hdim_v=80, runtime_head_dims=True
        )
        tensors = {
            "q": _fake_tensor((2, 32, 12, 80)),
            "k": _fake_tensor((2, 32, 4, 80)),
            "v": _fake_tensor((2, 32, 6, 80)),
            "out": _fake_tensor((2, 32, 12, 80)),
        }
        if kind == "prefill":
            assert _collect(request, spec, tensors)["num_v_heads"] == 6
        else:
            with pytest.raises(ValueError, match="divide the other"):
                _collect(request, spec, tensors)

    def test_v_head_count_is_nhead_k_without_runtime_head_dims(self, kind):
        values = _collect(_dense_request(), _meta_spec(kind), _dense_tensors())
        assert values["num_v_heads"] == 2
        with pytest.raises(ValueError, match="v must have"):
            _collect(
                _dense_request(),
                _meta_spec(kind),
                {**_dense_tensors(), "v": _fake_tensor((2, 32, 1, 64))},
            )

    @pytest.mark.parametrize(
        "dq,dv", [(136, 80), (80, 136), (76, 80), (80, 4), (80, 44), (80, 272)]
    )
    def test_head_widths_must_be_multiples_of_8_inside_the_bucket(self, kind, dq, dv):
        spec = _meta_spec(kind, head=128)
        request = _dense_request(hdim_q=dq, hdim_v=dv)
        tensors = _dense_tensors(dq=dq, dv=dv, row=280)
        with pytest.raises(ValueError, match="must be a multiple of 8 in"):
            _collect(request, spec, tensors)

    @pytest.mark.parametrize(
        "dq,dv", [(80, 8), (72, 120), (128, 128), (64, 64), (8, 64), (16, 128)]
    )
    def test_decode_bucket_serves_q_widths_from_its_minimum(self, kind, dq, dv):
        """Every multiple of 8 inside the bucket runs on the prefill object;
        the decode bucket 128 serves Q/K widths from 72 up (V has no floor)."""
        spec = _meta_spec(kind, head=128)
        request = _dense_request(hdim_q=dq, hdim_v=dv)
        tensors = _dense_tensors(dq=dq, dv=dv)
        if kind == "decode" and dq < 72:
            with pytest.raises(ValueError, match="below what the decode bucket"):
                _collect(request, spec, tensors)
        else:
            values = _collect(request, spec, tensors)
            assert (values["head_dim_q"], values["head_dim_v"]) == (dq, dv)

    def test_runtime_head_dims_packed_v_heads(self, kind):
        spec = _meta_spec(kind, head=128, layout="ragged")
        request = _dense_request(
            batch=1,
            nhead_q=8,
            nhead_k=4,
            hdim_q=96,
            hdim_v=32,
            layout="ragged",
            runtime_head_dims=True,
        )
        cu = _fake_tensor((2,), dtype="int32")
        tensors = {
            "q": _fake_tensor((32, 8, 96)),
            "k": _fake_tensor((32, 4, 96)),
            "v": _fake_tensor((32, 1, 32)),
            "out": _fake_tensor((32, 8, 32)),
            "cu_seqlens_q": cu,
            "cu_seqlens_k": cu,
        }
        values = _collect(request, spec, tensors)
        assert (values["head_dim_q"], values["head_dim_v"], values["num_v_heads"]) == (
            96,
            32,
            1,
        )
        with pytest.raises(ValueError, match="head count and dim"):
            _collect(request, spec, {**tensors, "v": _fake_tensor((31, 1, 32))})

    def test_rejects_wrong_kv_dtype_for_fp8_spec(self, kind):
        tensors = _dense_tensors()  # K/V fp16, the spec expects fp8e4m3
        with pytest.raises(ValueError, match="dtype must be fp8"):
            _bind(
                kind,
                _dense_request(use_fp8=True),
                _meta_spec(kind, kv_dtype="fp8e4m3"),
                tensors,
                k_scale=0.5,
                v_scale=0.25,
            )

    def test_rejects_fnuz_fp8_kv(self, kind):
        k = _fake_tensor((2, 32, 2, 64), dtype="float8_e4m3fnuz")
        with pytest.raises(ValueError, match="not FNUZ"):
            _bind(
                kind,
                _dense_request(use_fp8=True),
                _meta_spec(kind, kv_dtype="fp8e4m3"),
                {**_dense_tensors(), "k": k, "v": k},
                k_scale=0.5,
                v_scale=0.25,
            )

    def test_requires_k_scale_v_scale_for_fp8_kv(self, kind):
        k = _fake_tensor((2, 32, 2, 64), dtype="float8_e4m3fn")
        with pytest.raises(ValueError, match=r"k_scale.*v_scale"):
            _bind(
                kind,
                _dense_request(use_fp8=True),
                _meta_spec(kind, kv_dtype="fp8e4m3"),
                {**_dense_tensors(), "k": k, "v": k},
            )

    def test_softcap_objects_require_a_positive_cap(self, kind):
        spec = _meta_spec(kind, softcap=True)
        request = _dense_request(use_softcap=True)
        with pytest.raises(ValueError, match=r"require kwargs\['softcap'\]"):
            _bind(kind, request, spec, _dense_tensors())
        for cap in (0.0, -1.0, float("inf")):
            with pytest.raises(ValueError, match="finite and positive"):
                _bind(kind, request, spec, _dense_tensors(), softcap=cap)

    def test_sinks_require_query_dtype_per_head_tensor(self, kind):
        spec = _meta_spec(kind, sinks=True)
        request = _dense_request(use_sinks=True)
        with pytest.raises(ValueError, match=r"tensors\['sinks'\]"):
            _collect(request, spec, _dense_tensors())
        for sinks in (
            _fake_tensor((4,), dtype="float32"),
            _fake_tensor((2,), dtype="float16"),
        ):
            with pytest.raises(ValueError, match="sinks"):
                _collect(request, spec, {**_dense_tensors(), "sinks": sinks})
        good = _fake_tensor((4,), dtype="float16")
        assert (
            _collect(request, spec, {**_dense_tensors(), "sinks": good})["sink_ptr"]
            is good
        )

    def test_ragged_requires_cu_seqlens_k(self, kind):
        tensors = _ragged_tensors()
        del tensors["cu_seqlens_k"]
        with pytest.raises(ValueError, match=r"requires tensors\['cu_seqlens_k'\]"):
            _bind(
                kind,
                _dense_request(layout="ragged"),
                _meta_spec(kind, layout="ragged"),
                tensors,
            )

    def test_paged_rejects_kv_cache_page_size_mismatch(self, kind):
        q = _fake_tensor((10, 4, 64))
        k = _fake_tensor((3, 8, 2, 64))  # page size 8, spec expects 16
        with pytest.raises(ValueError, match="page length must be 16"):
            _bind(
                kind,
                _dense_request(layout="paged", kv_block_size=16),
                _meta_spec(kind, layout="paged", page_block_size=16),
                {
                    "q": q,
                    "k": k,
                    "v": k,
                    "out": q,
                    "cu_seqlens_q": _fake_tensor((3,), dtype="int32"),
                    "seqused_k": _fake_tensor((2,), dtype="int32"),
                    "block_table": _fake_tensor((2, 4), dtype="int32"),
                },
            )

    def test_paged_collects_block_strides(self, kind):
        q = _fake_tensor((10, 4, 64))
        k = _fake_tensor((5, 8, 2, 64))
        table = _fake_tensor((2, 4), dtype="int32", strides=(6, 1))
        values = _collect(
            _dense_request(layout="paged", kv_block_size=8),
            _meta_spec(kind, layout="paged", page_block_size=8),
            {
                "q": q,
                "k": k,
                "v": k,
                "out": q,
                "cu_seqlens_q": _fake_tensor((3,), dtype="int32"),
                "seqused_k": _fake_tensor((2,), dtype="int32"),
                "block_table": table,
            },
        )
        assert values["block_table_stride"] == 6
        assert values["stride_k_block"] == 8 * 2 * 64
        assert values["num_blocks"] == 5
        assert (values["stride_k_token"], values["stride_k_head"]) == (2 * 64, 64)

    def test_rejects_undersized_packed_output_and_values(self, kind):
        tensors = _ragged_tensors()
        for name, value in (
            ("out", _fake_tensor((9, 4, 64))),
            ("v", _fake_tensor((11, 2, 64))),
        ):
            with pytest.raises(ValueError):
                _bind(
                    kind,
                    _dense_request(layout="ragged"),
                    _meta_spec(kind, layout="ragged"),
                    dict(tensors, **{name: value}),
                )

    def test_rejects_strided_sequence_metadata(self, kind):
        tensors = _ragged_tensors()
        for name in ("cu_seqlens_q", "cu_seqlens_k"):
            with pytest.raises(ValueError, match=name):
                _bind(
                    kind,
                    _dense_request(layout="ragged"),
                    _meta_spec(kind, layout="ragged"),
                    dict(
                        tensors,
                        **{name: _fake_tensor((3,), dtype="int32", strides=(2,))},
                    ),
                )

    def test_padded_rows_keep_16_byte_alignment(self, kind):
        tensors = _dense_tensors()
        padded = _fake_tensor((2, 32, 4, 64), strides=(32 * 4 * 72, 4 * 72, 72, 1))
        values = _collect(_dense_request(), _meta_spec(kind), {**tensors, "q": padded})
        assert values["stride_q_head"] == 72
        # Non-overlapping, but head rows would start every 130 bytes.
        odd = _fake_tensor((2, 32, 4, 64), strides=(32 * 4 * 65, 4 * 65, 65, 1))
        with pytest.raises(ValueError, match=r"q stride\(\d\)=\d+ elements .*16 bytes"):
            _collect(_dense_request(), _meta_spec(kind), {**tensors, "q": odd})

    def test_rejects_misaligned_data_pointers(self, kind):
        tensors = _dense_tensors()
        for name in ("q", "k", "v", "out"):
            shape = tensors[name].shape
            shifted = _fake_tensor(shape, ptr=0x1008)
            with pytest.raises(
                ValueError, match=f"{name} data pointer must be 16-byte"
            ):
                _collect(_dense_request(), _meta_spec(kind), {**tensors, name: shifted})

    def test_accepts_bhsd_and_lse_strides(self, kind):
        q = _fake_tensor((2, 4, 32, 64))
        k = _fake_tensor((2, 2, 32, 64))
        lse = _fake_tensor((2, 4, 32, 1), dtype="float32")
        request = _dense_request(tensor_layout="bhsd", emit_lse=True)
        values = _collect(
            request,
            _meta_spec(kind),
            {"q": q, "k": k, "v": k, "out": q, "lse": lse},
        )
        assert values["stride_q_token"] == 64
        assert values["stride_q_head"] == 32 * 64
        assert values["stride_lse_token"] == 1
        assert values["stride_lse_head"] == 32
        assert values["write_lse"] == 1

    def test_masks_are_runtime_arguments(self, kind):
        q = _fake_tensor((2, 32, 4, 64))
        k = _fake_tensor((2, 64, 2, 64))
        tensors = {"q": q, "k": k, "v": k, "out": q}
        cases = (
            # Legacy left width: keys (q + ctx - 128, q + ctx].
            (
                dict(
                    mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
                    sliding_window=128,
                ),
                (1, 127, 0),
            ),
            (dict(mask_type=AttentionMaskType.TOP_LEFT_CAUSAL), (0, -1, 0)),
            (dict(mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL), (1, -1, 0)),
            (dict(mask_type=AttentionMaskType.NO_MASK), (0, -1, -1)),
            (
                dict(
                    mask_type=AttentionMaskType.SLIDING_WINDOW,
                    window_left=3,
                    window_right=5,
                ),
                (0, 3, 5),
            ),
            (
                dict(mask_type=AttentionMaskType.TOP_LEFT_CAUSAL, window_left=9),
                (0, 9, 0),
            ),
        )
        for extra, expected in cases:
            values = _collect(
                _dense_request(seqlen_k=64, **extra), _meta_spec(kind), tensors
            )
            assert (
                values["bottom_right"],
                values["window_left"],
                values["window_right"],
            ) == expected, extra
