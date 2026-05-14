# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch tensor launcher for CK DSL HSACO kernels.

This runtime is for integrations like AITER where tensors already live on the
GPU. It avoids host staging: kernel arguments are packed from
`torch.Tensor.data_ptr()` and Python scalar values, then launched through the
same hipModule path as `run_manifest`.
"""

from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class TorchLaunchSummary:
    ms: float
    attempts: int


def _require_torch():
    try:
        import torch
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("ck_dsl.runtime.torch_module requires torch") from e
    return torch


def resolve_stream(stream, device=None) -> int:
    """Resolve a kernel launch stream to a torch-tracked HIP stream handle.

    When the caller passes ``stream=0`` (or ``None``), we substitute
    ``int(torch.cuda.current_stream(device).cuda_stream)``. This is
    essential for correctness: workspace tensors are allocated by
    ``torch.empty(..., device=q.device)`` which records the
    allocation against torch's current stream. The torch caching
    allocator's stream-aware free queues a release event on *that*
    stream when the Python reference count drops. If we then launch on
    the literal HIP null stream (handle ``0``), the allocator never
    sees our launch and may hand the workspace memory to the next
    ``torch.empty`` *while our kernel is still writing to it*. Matching
    the launch stream to the allocation stream makes the standard
    FIFO single-stream ordering correctness rules apply, exactly the
    way Triton and AITER's paged-attention shim do it.

    Torch-optional: callers that don't use torch tensors (e.g. the
    numpy ``run_manifest`` runner) can pass ``stream=0`` and we will
    return ``0`` -- there's no torch caching allocator to be racing
    against, so the legacy HIP null stream is fine.
    """
    if stream is not None and int(stream) != 0:
        return int(stream)
    try:
        import torch
    except Exception:
        return 0
    if device is None:
        device = torch.cuda.current_device()
    return int(torch.cuda.current_stream(device).cuda_stream)


def pack_args(
    signature: Sequence[Mapping[str, Any]], values: Mapping[str, Any]
) -> bytes:
    """Pack args from a manifest-style signature.

    Supported `type`s:
      - `ptr<..., global>`: value is a torch tensor or integer device pointer
      - `i32`, `i64`
      - `f32`
    """
    fmt = "<"
    packed: List[Any] = []
    for arg in signature:
        name = str(arg["name"])
        ty = str(arg["type"])
        if name not in values:
            raise KeyError(f"missing kernel arg {name!r}")
        v = values[name]
        if ty.startswith("ptr<"):
            fmt += "Q"
            packed.append(_as_ptr(v))
        elif ty == "i32":
            fmt += "i"
            packed.append(int(v))
        elif ty == "i64":
            fmt += "q"
            packed.append(int(v))
        elif ty == "f32":
            fmt += "f"
            packed.append(float(v))
        else:
            raise ValueError(f"unsupported torch arg type {ty!r} for {name}")
    return struct.pack(fmt, *packed)


def pack_args_kernelparams(
    signature: Sequence[Mapping[str, Any]], values: Mapping[str, Any]
) -> List[Any]:
    """Pack args as a list of individual ``ctypes`` scalars for the
    ``kernelParams`` path of ``hipModuleLaunchKernel``.

    Returning one ``ctypes`` object per kernel argument lets the launcher
    build a ``void* params[]`` array whose entries point to each scalar.
    The CUDA/HIP semantics for ``kernelParams`` guarantee that the
    driver copies each parameter into driver-owned memory at enqueue
    time -- in contrast to the ``extra`` /
    ``HIP_LAUNCH_PARAM_BUFFER_POINTER`` path, whose copy semantics are
    underspecified and have been observed to read the host buffer
    *after* ``hipModuleLaunchKernel`` returns. The ``extra`` race
    produced the parity-harness "max_abs jumps to 2.5 / 512 on the
    second CK call" symptom investigated in
    ``ck/dsl/unified_attention_creative_results.md``.

    This is the same approach Triton's AMD driver uses (see
    ``triton/backends/amd/driver.py:392-402``: ``void *params[] = { ... };
    hipModuleLaunchKernel(..., params, 0);``).
    """
    out: List[Any] = []
    for arg in signature:
        name = str(arg["name"])
        ty = str(arg["type"])
        if name not in values:
            raise KeyError(f"missing kernel arg {name!r}")
        v = values[name]
        if ty.startswith("ptr<"):
            out.append(ctypes.c_uint64(_as_ptr(v)))
        elif ty == "i32":
            out.append(ctypes.c_int32(int(v)))
        elif ty == "i64":
            out.append(ctypes.c_int64(int(v)))
        elif ty == "f32":
            out.append(ctypes.c_float(float(v)))
        else:
            raise ValueError(f"unsupported torch arg type {ty!r} for {name}")
    return out


def _as_ptr(v: Any) -> int:
    if isinstance(v, int):
        return v
    if hasattr(v, "data_ptr"):
        return int(v.data_ptr())
    if v is None:
        return 0
    raise TypeError(f"cannot convert {type(v).__name__} to device pointer")


def empty_workspace(shape: Sequence[int], *, dtype: Any, device: Any):
    """Allocate a fresh torch workspace tensor.

    Prefer :class:`ck_dsl.runtime.launcher.WorkspacePool` for any new
    code that needs workspace tensors across multiple launches -- the
    pool reuses one allocation per (name, shape, dtype, device) and
    avoids the torch caching-allocator race that a per-call
    ``torch.empty`` is vulnerable to when the kernel is launched
    through our ctypes path. ``empty_workspace`` remains here for
    one-off scratch needs and back-compat.
    """
    torch = _require_torch()
    return torch.empty(tuple(int(x) for x in shape), dtype=dtype, device=device)


def launch_torch_kernel(
    *,
    hsaco: bytes,
    kernel_name: str,
    signature: Sequence[Mapping[str, Any]],
    values: Mapping[str, Any],
    grid: Tuple[int, int, int],
    block: Tuple[int, int, int],
    warmup: int = 5,
    attempts: int = 100,
    stream: int = 0,
) -> TorchLaunchSummary:
    """Compile, time, and launch a CK DSL kernel on torch tensors.

    Back-compat shim over :class:`ck_dsl.runtime.launcher.KernelLauncher`
    and :func:`ck_dsl.runtime.launcher.time_launches`.

    For new code, prefer constructing a long-lived
    :class:`KernelLauncher` directly and calling it from your own
    dispatch path -- you'll avoid the per-call module load + arg-
    lifetime book-keeping this shim has to do under the hood, and
    you get :class:`WorkspacePool` and :class:`PipelineLauncher`
    interop for free.

    Special case: when ``warmup == 0 and attempts == 1`` this function
    issues exactly one launch and skips the internal timing entirely
    (no device sync, no event creation). That single-shot mode is
    required for HIP graph capture, where ``hipDeviceSynchronize`` and
    ``hipEventRecord`` are illegal on the captured stream.
    """
    # Local import to avoid a runtime/__init__ circular at module load.
    from .launcher import KernelLauncher, LaunchConfig, time_launches

    launcher = KernelLauncher(
        hsaco=hsaco,
        kernel_name=kernel_name,
        signature=signature,
    )
    cfg = LaunchConfig(grid=grid, block=block, stream=stream)

    def call_once():
        launcher(values, config=cfg)

    if int(warmup) == 0 and int(attempts) == 1:
        call_once()
        return TorchLaunchSummary(ms=0.0, attempts=1)

    ms = time_launches(call_once, warmup=warmup, iters=attempts, stream=stream)
    return TorchLaunchSummary(ms=ms, attempts=int(attempts))
