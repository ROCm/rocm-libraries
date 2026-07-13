# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-tensor launch glue, plus the torch-optional stream resolver.

For integrations like AITER where tensors already live on the GPU: launch
through the same hipModule path as `run_manifest`, without host staging.
This module holds `resolve_stream` (substitute torch's current stream so
its caching allocator stays coherent -- degrades to the HIP null stream
when torch is absent, so the numpy/manifest hip-only path calls it too),
`empty_workspace`, and the `launch_torch_kernel` back-compat shim.

Kernel-argument packing is torch-agnostic and lives in `packing.py`; this
module builds on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class TorchLaunchSummary:
    ms: float
    attempts: int


def _require_torch():
    try:
        import torch
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("rocke.runtime.torch_interop requires torch") from e
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
    try:
        if device is None:
            device = torch.cuda.current_device()
        return int(torch.cuda.current_stream(device).cuda_stream)
    except Exception:
        # This function is also used by the numpy/raw-HIP manifest
        # runner. In that path there are no torch-owned tensors and no
        # torch caching allocator lifetime rules to respect, so the HIP
        # null stream is correct. If torch is importable but cannot
        # initialize in this process (for example after raw HIP has
        # already loaded a module), honor the torch-optional contract in
        # the docstring and fall back to stream 0.
        return 0


def empty_workspace(shape: Sequence[int], *, dtype: Any, device: Any):
    """Allocate a fresh torch workspace tensor.

    Prefer :class:`rocke.runtime.launcher.WorkspacePool` for any new
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

    Back-compat shim over :class:`rocke.runtime.launcher.KernelLauncher`
    and :func:`rocke.runtime.launcher.time_launches`.

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
