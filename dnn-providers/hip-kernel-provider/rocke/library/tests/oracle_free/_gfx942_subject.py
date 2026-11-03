# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Narrow uniform-probability subject adapter for the shipped gfx942 dense kernel.

Torch-free end to end: inputs and the output are raw ``uint16`` host arrays,
device buffers are :class:`rocke.runtime.DeviceMem`, and the launch goes through
``kernels.gfx942.attention_dense.launch_attention_dense``, the compile, cache and
launch path that ``run_attention_dense_torch`` itself delegates to. Dispatch, the
runner and the HIP runtime are imported only inside the functions that use them,
so CPU collection and the analytical contracts need neither a GPU nor ROCm.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ._analytical import UniformCase, UniformSpecFingerprint

_POINTER_ALIGNMENT = 16
# Quiet-NaN output prefill: a coordinate the kernel never writes reads back as
# invalid instead of passing as a stale zero.
_NAN_BITS = {"fp16": 0x7E00, "bf16": 0x7FC0}


@dataclass(frozen=True)
class ResolvedDenseSubject:
    """Dispatch-resolved dense subject plus its proof-relevant fingerprint."""

    case_id: str
    tuning_spec: object
    candidate_name: str
    fingerprint: UniformSpecFingerprint


def _validate_host_bits(case: UniformCase, q, k, v) -> None:
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


def _validate_supported_case(case: UniformCase) -> None:
    """Reject shapes outside the first shipped gfx942 uniform-probability matrix."""
    if case.batch != 1:
        raise ValueError("gfx942 uniform-probability cases require batch 1")
    if case.nhead_q != 4 or case.nhead_k != 4:
        raise ValueError("gfx942 uniform-probability cases require Hq=Hkv=4")
    if case.head_dim != 64:
        raise ValueError("gfx942 uniform-probability cases require D64")
    if case.seqlen_q != 256:
        raise ValueError("gfx942 uniform-probability cases require query length 256")
    if case.seqlen_k != 256:
        raise ValueError("gfx942 uniform-probability cases require key length 256")
    if case.mask == "window" and case.window != 64:
        raise ValueError("gfx942 uniform-probability cases require window 64")


def _verify_resolved_spec(case: UniformCase, tuning_spec) -> UniformSpecFingerprint:
    """Assert the requested dense path and return its proof fingerprint."""
    if tuning_spec.path != "dense":
        raise AssertionError(f"dispatch returned path={tuning_spec.path!r}")
    if tuning_spec.arch != "gfx942":
        raise AssertionError(f"dispatch returned arch={tuning_spec.arch!r}")
    if tuning_spec.candidate_name != "attention_gfx942_dense":
        raise AssertionError(
            f"dispatch returned candidate={tuning_spec.candidate_name!r}"
        )
    spec = tuning_spec.kernel_spec
    expected = {
        "batch": case.batch,
        "seqlen_q": case.seqlen_q,
        "seqlen_kv": case.seqlen_k,
        "num_query_heads": case.nhead_q,
        "num_kv_heads": case.nhead_k,
        "head_size": case.head_dim,
        "dtype": case.dtype,
        "causal": True,
        "sliding_window": case.window if case.mask == "window" else 0,
        "persistent": False,
    }
    for field, value in expected.items():
        actual = getattr(spec, field)
        if actual != value:
            raise AssertionError(
                f"resolved spec {field}={actual!r}, expected {value!r}"
            )
    if not spec.runtime_shape:
        raise AssertionError("oracle-free cases require the runtime-shape kernel")
    if spec.block_n != 64:
        raise AssertionError(f"unexpected shipped block_n {spec.block_n}")
    if not spec.resolved_use_exp2_fast():
        raise AssertionError("oracle-free quick cases require the fast exp2 path")
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


def _resolve_tuning_spec(case: UniformCase):
    """Resolve ``gfx942_dense`` through the public attention tuning factory."""
    from dispatch.attention import AttentionRequest, tuning_spec_with_knobs

    request = AttentionRequest(
        batch=case.batch,
        nhead_q=case.nhead_q,
        nhead_k=case.nhead_k,
        seqlen_q=case.seqlen_q,
        seqlen_k=case.seqlen_k,
        hdim_q=case.head_dim,
        hdim_v=case.head_dim,
        arch="gfx942",
        mask_type=1,
        sliding_window=case.window if case.mask == "window" else 0,
        dtype=case.dtype,
    )
    tuning_spec = tuning_spec_with_knobs(request, "gfx942_dense", {"persistent": False})
    _verify_resolved_spec(case, tuning_spec)
    return tuning_spec


def resolve_dense_subject(case: UniformCase) -> ResolvedDenseSubject:
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
    case: UniformCase,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    subject: ResolvedDenseSubject | None = None,
) -> np.ndarray:
    """Launch one validated subject on raw input bits and return raw output bits.

    ``q``/``k``/``v`` are ``uint16`` FP16/BF16 bit arrays in the runner layout
    (``[B, Sq, Hq, D]`` and ``[B, Sk, Hkv, D]``). The output is NaN-prefilled and
    read back without any float conversion, so the analyzer judges exactly the
    stored bits. Numerical judgment is left to the analyzer."""
    from kernels.gfx942.attention_dense import launch_attention_dense
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
        scale=1.0 / math.sqrt(case.head_dim),
    )
    out = host["out"]
    runtime.memcpy_d2h(as_u8_buffer(out), device["out"].ptr(), out.nbytes)
    return out
