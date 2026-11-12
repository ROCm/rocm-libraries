# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Narrow uniform-probability subject adapter for the shipped gfx942 dense kernel.

Torch-free end to end: inputs and the output are raw ``uint16`` host arrays,
device buffers are :class:`rocke.runtime.DeviceMem`, and the launch goes through
:func:`launch_attention_dense`, a pointer-level compile, cache and launch
path kept in this module. Dispatch, the runner and the
HIP runtime are imported only inside the functions that use them, so CPU
collection and the analytical contracts need neither a GPU nor ROCm.

The structured-logit lane's own subject adapter: the launch path and the
Hq = Hkv = 4 quick shapes its enclosures are derived against. The uniform lane
uses ``_dense_subject.py``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ._p4_shared_analytical import UniformSpecFingerprint

_POINTER_ALIGNMENT = 16
# Quiet-NaN output prefill: a coordinate the kernel never writes reads back as
# invalid instead of passing as a stale zero.
_NAN_BITS = {"fp16": 0x7E00, "bf16": 0x7FC0}


def launch_attention_dense(
    *,
    spec,
    q,
    k,
    v,
    out,
    scale: float,
    stream: int = 0,
    arch: str = "gfx942",
    sinks=None,
    lse=None,
    segment_workspace=None,
) -> None:
    """Torch-free compile (cached) + launch of the gfx942 dense prefill kernel.

    A test-local pointer-level launcher: the kernel module exports only the
    Torch entry, and this suite is Torch-free. ``q``/``k``/``v``/``out`` (and
    ``sinks``, ``lse`` and the ``segment_workspace`` values) are anything the
    launcher packs as a device pointer: an ``int`` address, a
    :class:`rocke.runtime.DeviceMem`, or an object exposing ``data_ptr()``. The
    layout contract is the torch entry's; this layer checks only spec legality
    and that each spec-required buffer is present, so shape, dtype, placement
    and aliasing are the caller's to guarantee.

    ``spec.kv_segments > 1`` needs ``segment_workspace``, a mapping with one FP32
    buffer per ``SEGMENT_PARAMS`` name; the caller then runs
    ``launch_segment_combine``. The launch is fenced (``LaunchConfig`` default),
    so ``out`` is complete on return."""
    from kernels.gfx942 import attention_dense as kernel
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig

    spec = kernel._as_gfx942_spec(spec)
    ok, why = kernel.supports_attention_dense(spec, arch=arch)
    if not ok:
        raise NotImplementedError(f"attention_dense unsupported for spec: {why}")
    for name, value, required in (
        ("sinks", sinks, spec.use_sinks),
        ("lse", lse, spec.emit_lse),
        ("segment_workspace", segment_workspace, spec.kv_segments > 1),
    ):
        if required and value is None:
            raise ValueError(f"the spec requires {name}, got None")
        if not required and value is not None:
            raise ValueError(f"{name} provided but the spec does not use it")
    if segment_workspace is not None and set(segment_workspace) != set(
        kernel.SEGMENT_PARAMS
    ):
        raise ValueError(
            f"segment_workspace keys {sorted(segment_workspace)} != "
            f"{sorted(kernel.SEGMENT_PARAMS)}"
        )

    key = kernel.attention_dense_cache_key(spec, arch=arch)
    launcher = kernel._DENSE_LAUNCHER_CACHE.get(key)
    if launcher is None:
        art = compile_kernel(
            kernel.build_attention_dense(spec, arch=arch),
            arch=arch,
            backend="python",
            capture_ir_text=False,
        )
        assert art.kernel_name == spec.kernel_name(), (
            art.kernel_name,
            spec.kernel_name(),
        )
        launcher = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=kernel.attention_dense_signature(spec),
        )
        kernel._DENSE_LAUNCHER_CACHE[key] = launcher
    vals = {"q_ptr": q, "k_ptr": k, "v_ptr": v, "o_ptr": out, "scale": float(scale)}
    vals.update(kernel.attention_dense_runtime_args(spec))
    if spec.use_sinks:
        vals["sink_ptr"] = sinks
    if spec.emit_lse:
        vals["lse_ptr"] = lse
    if segment_workspace is not None:
        vals.update(segment_workspace)
    launcher(
        vals,
        config=LaunchConfig(
            grid=kernel.attention_dense_grid(spec),
            block=kernel.attention_dense_block(spec),
            stream=int(stream),
        ),
    )


class DenseCase(Protocol):
    """What the adapter reads from a case: the uniform ``UniformCase`` and the
    structured-logit ``P4Case`` both provide it."""

    output_shape: tuple[int, int, int, int]  # [B, Sq, Hq, D]
    key_shape: tuple[int, int, int, int]  # [B, Sk, Hkv, D]
    dtype: str
    mask: str
    window: int
    case_id: str


@dataclass(frozen=True)
class _Shape:
    batch: int
    seqlen_q: int
    seqlen_k: int
    nhead_q: int
    nhead_k: int
    head_dim: int


def _shape(case: DenseCase) -> _Shape:
    batch, seqlen_q, nhead_q, head_dim = case.output_shape
    key_batch, seqlen_k, nhead_k, key_dim = case.key_shape
    if (key_batch, key_dim) != (batch, head_dim):
        raise ValueError(
            f"inconsistent Q/K shapes {case.output_shape}, {case.key_shape}"
        )
    return _Shape(batch, seqlen_q, seqlen_k, nhead_q, nhead_k, head_dim)


@dataclass(frozen=True)
class ResolvedDenseSubject:
    """Dispatch-resolved dense subject plus its proof-relevant fingerprint."""

    case_id: str
    tuning_spec: object
    candidate_name: str
    fingerprint: UniformSpecFingerprint


def _validate_host_bits(case: DenseCase, q, k, v) -> None:
    """Validate the dense ABI on the raw-bit host inputs."""
    expected_shapes = {
        "q": case.output_shape,
        "k": case.key_shape,
        "v": case.key_shape,
    }
    for name, array in (("q", q), ("k", k), ("v", v)):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a numpy array of raw bits")
        if array.shape != expected_shapes[name]:
            raise ValueError(f"{name} shape {array.shape} != {expected_shapes[name]}")
        if array.dtype != np.uint16:
            raise ValueError(f"{name} dtype {array.dtype} != uint16 raw bits")
        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


def _validate_device_buffers(
    buffers: Mapping[str, object], nbytes: Mapping[str, int]
) -> None:
    """Check each device buffer's size, alignment and separation.

    Duck-typed on ``ptr()`` / ``nbytes()`` so the contract is CPU-testable with
    small fakes; ``run_dense_case`` passes real ``DeviceMem`` objects."""
    spans: dict[str, tuple[int, int]] = {}
    for name, buffer in buffers.items():
        size = int(buffer.nbytes())
        if size != nbytes[name]:
            raise ValueError(f"{name} device size {size} != host size {nbytes[name]}")
        pointer = int(buffer.ptr())
        if pointer == 0:
            raise ValueError(f"{name} device pointer is null")
        if pointer % _POINTER_ALIGNMENT:
            raise ValueError(
                f"{name} pointer must be {_POINTER_ALIGNMENT}-byte aligned"
            )
        spans[name] = (pointer, pointer + size)

    names = tuple(spans)
    for index, left_name in enumerate(names):
        left_start, left_end = spans[left_name]
        for right_name in names[index + 1 :]:
            right_start, right_end = spans[right_name]
            if left_start < right_end and right_start < left_end:
                raise ValueError(f"{left_name} overlaps {right_name}")


def _validate_supported_case(case: DenseCase) -> None:
    """Reject shapes outside the first shipped gfx942 uniform-probability matrix."""
    shape = _shape(case)
    if shape.batch != 1:
        raise ValueError("gfx942 uniform-probability cases require batch 1")
    if shape.nhead_q != 4 or shape.nhead_k != 4:
        raise ValueError("gfx942 uniform-probability cases require Hq=Hkv=4")
    if shape.head_dim != 64:
        raise ValueError("gfx942 uniform-probability cases require D64")
    if shape.seqlen_q != 256:
        raise ValueError("gfx942 uniform-probability cases require query length 256")
    if shape.seqlen_k != 256:
        raise ValueError("gfx942 uniform-probability cases require key length 256")
    if case.mask == "window" and case.window != 64:
        raise ValueError("gfx942 uniform-probability cases require window 64")


def _verify_resolved_spec(case: DenseCase, tuning_spec) -> UniformSpecFingerprint:
    """Check what the adapter needs to launch safely and return the fingerprint.

    Only preconditions of the launch itself are asserted here: dispatch identity,
    the request-shape echo the host buffers are sized from, the buffers the
    adapter does not supply (sinks, LSE), and scope the fingerprint does not
    record. Every other fingerprint field (persistence, runtime shape, tile, fast
    exp2) is left to ``require_calibration``, so a drift there still launches,
    writes calibration records, and fails with the re-ratification remedy."""
    if tuning_spec.path != "dense":
        raise AssertionError(f"dispatch returned path={tuning_spec.path!r}")
    if tuning_spec.arch != "gfx942":
        raise AssertionError(f"dispatch returned arch={tuning_spec.arch!r}")
    if tuning_spec.candidate_name != "attention_gfx942_dense":
        raise AssertionError(
            f"dispatch returned candidate={tuning_spec.candidate_name!r}"
        )
    spec = tuning_spec.kernel_spec
    shape = _shape(case)
    expected = {
        "batch": shape.batch,
        "seqlen_q": shape.seqlen_q,
        "seqlen_kv": shape.seqlen_k,
        "num_query_heads": shape.nhead_q,
        "num_kv_heads": shape.nhead_k,
        "head_size": shape.head_dim,
        "dtype": case.dtype,
        "causal": True,
        "sliding_window": case.window if case.mask == "window" else 0,
    }
    for field, value in expected.items():
        actual = getattr(spec, field)
        if actual != value:
            raise AssertionError(
                f"resolved spec {field}={actual!r}, expected {value!r}"
            )
    # The adapter allocates only Q, K, V and O, so a spec that needs a sink or
    # LSE buffer is outside what it can launch, not a recalibration.
    if spec.use_sinks:
        raise AssertionError(
            "the uniform-probability check requires attention sinks to be disabled"
        )
    if spec.emit_lse:
        raise AssertionError(
            "the uniform-probability check requires the direct non-LSE epilogue"
        )
    # A KV-segment split combines partial softmax states, ragged padding adds
    # masked tail tiles, and bottom-right shifts the band; the single-pass
    # top-left argument behind the calibration covers none of them.
    if spec.kv_segments != 1:
        raise AssertionError(
            "the uniform-probability check requires single-pass KV, "
            f"kv_segments={spec.kv_segments}"
        )
    if spec.ragged:
        raise AssertionError(
            "the uniform-probability check requires a non-ragged shape"
        )
    if spec.causal_bottom_right:
        raise AssertionError(
            "the uniform-probability check requires the top-left causal band"
        )
    return UniformSpecFingerprint(
        path=tuning_spec.path,
        persistent=bool(spec.persistent),
        runtime_shape=bool(spec.runtime_shape),
        head_dim=int(spec.head_size),
        block_m=int(spec.block_m),
        block_n=int(spec.block_n),
        use_exp2_fast=bool(spec.resolved_use_exp2_fast()),
        use_sinks=bool(spec.use_sinks),
        emit_lse=bool(spec.emit_lse),
    )


def _resolve_tuning_spec(case: DenseCase):
    """Resolve ``gfx942_dense`` through the public attention tuning factory."""
    from dispatch.attention import AttentionRequest, tuning_spec_with_knobs

    shape = _shape(case)
    request = AttentionRequest(
        batch=shape.batch,
        nhead_q=shape.nhead_q,
        nhead_k=shape.nhead_k,
        seqlen_q=shape.seqlen_q,
        seqlen_k=shape.seqlen_k,
        hdim_q=shape.head_dim,
        hdim_v=shape.head_dim,
        arch="gfx942",
        mask_type=1,
        sliding_window=case.window if case.mask == "window" else 0,
        dtype=case.dtype,
    )
    tuning_spec = tuning_spec_with_knobs(request, "gfx942_dense", {"persistent": False})
    _verify_resolved_spec(case, tuning_spec)
    return tuning_spec


def resolve_dense_subject(case: DenseCase) -> ResolvedDenseSubject:
    """Resolve one supported quick case and expose its calibration identity."""
    _validate_supported_case(case)
    tuning_spec = _resolve_tuning_spec(case)
    return ResolvedDenseSubject(
        case_id=case.case_id,
        tuning_spec=tuning_spec,
        candidate_name=str(tuning_spec.candidate_name),
        fingerprint=_verify_resolved_spec(case, tuning_spec),
    )


def run_dense_case(
    case: DenseCase,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    subject: ResolvedDenseSubject | None = None,
    scale_override: float | None = None,
) -> np.ndarray:
    """Launch one validated subject on raw input bits and return raw output bits.

    ``q``/``k``/``v`` are ``uint16`` FP16/BF16 bit arrays in the runner layout
    (``[B, Sq, Hq, D]`` and ``[B, Sk, Hkv, D]``). The output is NaN-prefilled and
    read back without any float conversion, so the analyzer judges exactly the
    stored bits. Numerical judgment is left to the analyzer.

    ``scale_override`` replaces the softmax scale ``1/sqrt(D)``; it exists only
    for the subject-side wrong-scale mutation, whose output must be rejected."""
    from rocke.runtime import DeviceMem, Runtime
    from rocke.runtime.host_buffers import as_u8_buffer

    _validate_supported_case(case)
    _validate_host_bits(case, q, k, v)
    resolved = resolve_dense_subject(case) if subject is None else subject
    if resolved.case_id != case.case_id:
        raise ValueError(
            f"resolved subject case {resolved.case_id!r} != {case.case_id!r}"
        )
    tuning_spec = resolved.tuning_spec
    fingerprint = _verify_resolved_spec(case, tuning_spec)
    if fingerprint != resolved.fingerprint:
        raise AssertionError("resolved subject fingerprint changed before launch")

    # Private writable copies: ``as_u8_buffer`` views storage via ``from_buffer``,
    # which refuses read-only arrays.
    host = {
        "q": q.copy(),
        "k": k.copy(),
        "v": v.copy(),
        "out": np.full(case.output_shape, _NAN_BITS[case.dtype], dtype=np.uint16),
    }
    runtime = Runtime()
    device = {name: DeviceMem(array.nbytes) for name, array in host.items()}
    _validate_device_buffers(
        device, {name: int(array.nbytes) for name, array in host.items()}
    )
    for name, array in host.items():
        runtime.memcpy_h2d(device[name].ptr(), as_u8_buffer(array), array.nbytes)
    # Fenced launch (LaunchConfig default): the output is complete on return.
    launch_attention_dense(
        spec=tuning_spec.kernel_spec,
        q=device["q"],
        k=device["k"],
        v=device["v"],
        out=device["out"],
        scale=(
            1.0 / math.sqrt(_shape(case).head_dim)
            if scale_override is None
            else float(scale_override)
        ),
    )
    out = host["out"]
    runtime.memcpy_d2h(as_u8_buffer(out), device["out"].ptr(), out.nbytes)
    return out
