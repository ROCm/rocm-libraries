# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-tensor launch glue, plus the torch-optional stream resolver.

For integrations like AITER where tensors already live on the GPU: launch
through the same hipModule path as `run_manifest`, without host staging.
This module holds `resolve_stream` (substitute torch's current stream so
its caching allocator stays coherent -- degrades to the HIP null stream
when torch is absent, so the numpy/manifest hip-only path calls it too),
`empty_workspace`, `torch_dtype`, `device_tensor`, `is_torch_tensor`, the
torch sides of :mod:`rocke.runtime.device_array` (`torch_array_info`,
`torch_host_values`, `torch_empty`), and the `launch_torch_kernel` back-compat
shim.

Kernel-argument packing is torch-agnostic and lives in `packing.py`; this
module builds on it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..numeric.storage import host_dtype

if TYPE_CHECKING:
    from .device_array import ArrayInfo


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


def is_torch_tensor(value: object) -> bool:
    """Whether ``value`` is a torch object, judged by its type's module so torch
    is never imported to answer."""
    return type(value).__module__.split(".", 1)[0] == "torch"


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

    Never imports torch: only a process that has already imported it can
    hold torch tensors, so an unimported torch means there is no allocator
    to stay coherent with. Importing it here would also be unsafe -- a
    torch-free caller in an environment that merely has torch installed
    has already loaded a HIP runtime by the time it launches, and TheRock
    torch preloads its bundled ROCm libraries at import, which aborts the
    process. Library discovery follows the same rule
    (``runtime_coexistence._candidate_lib_paths``).
    """
    if stream is not None and int(stream) != 0:
        return int(stream)
    torch = sys.modules.get("torch")
    if torch is None:
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


#: ``str(torch.dtype)`` -> rocKE element type, for the dtypes a kernel argument can hold.
_TORCH_ELEMENT_TYPE = {
    "torch.float16": "fp16",
    "torch.bfloat16": "bf16",
    "torch.float32": "fp32",
    "torch.float64": "fp64",
    "torch.float8_e4m3fn": "fp8e4m3",
    "torch.float8_e5m2": "bf8e5m2",
    "torch.float8_e4m3fnuz": "fp8e4m3fnuz",
    "torch.float8_e5m2fnuz": "bf8e5m2fnuz",
    "torch.int8": "i8",
    "torch.int16": "i16",
    "torch.int32": "i32",
    "torch.int64": "i64",
    "torch.uint8": "u8",
    "torch.bool": "i1",
}


def torch_array_info(tensor: Any) -> ArrayInfo:
    """The :class:`rocke.runtime.device_array.ArrayInfo` of a torch tensor.

    Reads only tensor attributes, so it never imports torch. ``device`` is the
    CUDA/HIP ordinal, or ``None`` for a CPU tensor. The FNUZ fp8 types are named
    by their storage encodings (see :class:`ArrayInfo`)."""
    from .device_array import ArrayInfo

    name = str(tensor.dtype)
    if name not in _TORCH_ELEMENT_TYPE:
        raise ValueError(
            f"torch_array_info failed - dtype={name}, expected one of "
            f"{sorted(_TORCH_ELEMENT_TYPE)}"
        )
    device = tensor.device
    return ArrayInfo(
        shape=tuple(int(extent) for extent in tensor.shape),
        strides=tuple(int(stride) for stride in tensor.stride()),
        dtype=_TORCH_ELEMENT_TYPE[name],
        element_size=int(tensor.element_size()),
        data_ptr=int(tensor.data_ptr()),
        device=int(device.index or 0) if device.type == "cuda" else None,
        backed_elements=int(tensor.untyped_storage().nbytes())
        // int(tensor.element_size())
        - int(tensor.storage_offset()),
    )


def _torch_dtype_of(element_type: str):
    """The ``torch.dtype`` holding rocKE element type ``element_type``."""
    for torch_name, name in _TORCH_ELEMENT_TYPE.items():
        if name == element_type:
            return torch_dtype(torch_name.removeprefix("torch."))
    raise ValueError(
        f"torch dtype failed - element type={element_type!r}, expected one of "
        f"{sorted(_TORCH_ELEMENT_TYPE.values())}"
    )


def torch_host_values(tensor: Any):
    """``tensor``'s elements as a host numpy array of its shape (the torch side
    of :func:`rocke.runtime.device_array.host_values`).

    Copied by ``Tensor.cpu()``, so ordered on torch's current stream. Element
    types numpy has no type for (bf16, fp8, FNUZ fp8) come back as raw bits of
    the same width. Only reached with a live tensor, so torch is already
    imported."""
    info = torch_array_info(tensor)
    torch = sys.modules["torch"]
    flat = tensor.detach().contiguous().cpu().reshape(-1)
    raw = flat.view(torch.uint8).numpy()
    try:
        values_dtype = host_dtype(info.dtype)
    except ValueError:  # a FNUZ storage encoding: raw bits
        values_dtype = np.dtype(f"u{info.element_size}")
    return raw.view(values_dtype).reshape(info.shape)


def torch_empty(shape: Sequence[int], element_type: str, *, device: Any):
    """An uninitialized torch tensor of rocKE element type ``element_type`` on
    ``device`` (the torch side of
    :func:`rocke.runtime.device_array.device_workspace`)."""
    return empty_workspace(shape, dtype=_torch_dtype_of(element_type), device=device)


def torch_dtype(name: str):
    """The ``torch.dtype`` named ``name`` (e.g. ``"float16"``, ``"int32"``)."""
    dtype = getattr(_require_torch(), name, None)
    if dtype is None:
        raise ValueError(f"torch has no dtype {name!r}")
    return dtype


def device_tensor(values: Sequence[Any], *, dtype: Any, device: Any):
    """A 1-D tensor holding ``values`` on ``device``, e.g. a launch's int32
    lengths or block table built from host-side Python ints."""
    torch = _require_torch()
    return torch.tensor(list(values), dtype=dtype, device=device)


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
    cluster_dims: Optional[Sequence[int]] = None,
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

    ``cluster_dims`` is the kernel's compiled cluster shape, if any; every
    launch then uses it (see :class:`KernelLauncher`).
    """
    # Local import to avoid a runtime/__init__ circular at module load.
    from .launcher import KernelLauncher, LaunchConfig, time_launches

    launcher = KernelLauncher(
        hsaco=hsaco,
        kernel_name=kernel_name,
        signature=signature,
        cluster_dims=cluster_dims,
    )
    cfg = LaunchConfig(grid=grid, block=block, stream=stream)

    def call_once():
        launcher(values, config=cfg)

    if int(warmup) == 0 and int(attempts) == 1:
        call_once()
        return TorchLaunchSummary(ms=0.0, attempts=1)

    ms = time_launches(call_once, warmup=warmup, iters=attempts, stream=stream)
    return TorchLaunchSummary(ms=ms, attempts=int(attempts))
