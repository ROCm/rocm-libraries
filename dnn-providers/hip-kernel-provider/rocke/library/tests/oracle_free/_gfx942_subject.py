# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Narrow P3 subject adapter for the shipped gfx942 dense kernel.

Torch, dispatch, and the runner are imported only inside ``run_dense_case`` so
CPU collection and the analytical contracts remain independent of ROCm Torch.
"""

from __future__ import annotations

import math

from ._analytical import P3Case


_POINTER_ALIGNMENT = 16


def _tensor_span(tensor) -> tuple[int, int]:
    """Return the byte span of a validated contiguous, nonempty tensor."""
    start = int(tensor.data_ptr())
    return start, start + int(tensor.numel()) * int(tensor.element_size())


def _validate_tensors(
    case: P3Case,
    q,
    k,
    v,
    out,
    *,
    expected_dtype,
) -> None:
    """Validate the dense ABI without importing Torch.

    The duck-typed implementation makes the contract CPU-testable with small
    fakes.  ``run_dense_case`` separately requires real ``torch.Tensor`` objects.
    """
    tensors = {"q": q, "k": k, "v": v, "out": out}
    expected_shapes = {
        "q": case.output_shape,
        "k": case.key_shape,
        "v": case.key_shape,
        "out": case.output_shape,
    }
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected_shapes[name]:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} != {expected_shapes[name]}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(f"{name} dtype {tensor.dtype} != {expected_dtype}")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on q's device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if int(tensor.numel()) <= 0:
            raise ValueError(f"{name} must be nonempty")
        if int(tensor.data_ptr()) % _POINTER_ALIGNMENT:
            raise ValueError(
                f"{name} pointer must be {_POINTER_ALIGNMENT}-byte aligned"
            )

    spans = {name: _tensor_span(tensor) for name, tensor in tensors.items()}
    names = tuple(spans)
    for index, left_name in enumerate(names):
        left_start, left_end = spans[left_name]
        for right_name in names[index + 1 :]:
            right_start, right_end = spans[right_name]
            if left_start < right_end and right_start < left_end:
                raise ValueError(f"{left_name} overlaps {right_name}")


def _validate_supported_case(case: P3Case) -> None:
    """Reject shapes outside the first shipped gfx942 P3 matrix."""
    if case.batch != 1:
        raise ValueError("P3 gfx942 tranche requires batch 1")
    if case.nhead_q != 4 or case.nhead_k != 4:
        raise ValueError("P3 gfx942 tranche requires Hq=Hkv=4")
    if case.head_dim != 64:
        raise ValueError("P3 gfx942 tranche requires D64")
    if case.seqlen_q != 256:
        raise ValueError("P3 gfx942 tranche requires query length 256")
    if case.seqlen_k != 256:
        raise ValueError("P3 gfx942 tranche requires key length 256")
    if case.mask == "window" and case.window != 64:
        raise ValueError("P3 gfx942 tranche requires window 64")


def _verify_resolved_spec(case: P3Case, tuning_spec) -> None:
    """Assert that dispatch returned the requested nonpersistent dense subject."""
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
        "causal": case.mask != "none",
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
    if spec.block_m != 256 or spec.block_n != 64:
        raise AssertionError(f"unexpected shipped tile {spec.block_m}x{spec.block_n}")


def _resolve_tuning_spec(case: P3Case):
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
        mask_type=0 if case.mask == "none" else 1,
        sliding_window=case.window if case.mask == "window" else 0,
        dtype=case.dtype,
    )
    tuning_spec = tuning_spec_with_knobs(request, "gfx942_dense", {"persistent": False})
    _verify_resolved_spec(case, tuning_spec)
    return tuning_spec


def run_dense_case(
    case: P3Case,
    q,
    k,
    v,
    out,
):
    """Launch and synchronize one validated production dense-attention case."""
    import torch

    from kernels.gfx942.attention_dense import run_attention_dense_torch

    _validate_supported_case(case)
    for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out)):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
    expected_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[case.dtype]
    _validate_tensors(case, q, k, v, out, expected_dtype=expected_dtype)
    tuning_spec = _resolve_tuning_spec(case)

    stream = torch.cuda.current_stream(device=q.device)
    returned = run_attention_dense_torch(
        spec=tuning_spec.kernel_spec,
        q=q,
        k=k,
        v=v,
        out=out,
        scale=1.0 / math.sqrt(case.head_dim),
        stream=stream.cuda_stream,
    )
    if returned is not out:
        raise AssertionError("dense runner did not return the caller-owned output")
    stream.synchronize()
    invalid_positions = torch.nonzero(~torch.isfinite(out), as_tuple=False)
    invalid = int(invalid_positions.shape[0])
    if invalid:
        batch, query, head, dim = (
            int(value) for value in invalid_positions[0].cpu().tolist()
        )
        raise AssertionError(
            f"P3 {case.dtype} {case.mask} subject output contains {invalid} "
            f"non-finite elements; first at "
            f"b:q:h:d={batch}:{query}:{head}:{dim}"
        )
    return out
