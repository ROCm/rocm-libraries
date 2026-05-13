# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch tensor launcher for CK DSL HSACO kernels.

This runtime is for integrations like AITER where tensors already live on the
GPU. It avoids host staging: kernel arguments are packed from
`torch.Tensor.data_ptr()` and Python scalar values, then launched through the
same hipModule path as `run_manifest`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple

from .hip_module import Runtime


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


def _as_ptr(v: Any) -> int:
    if isinstance(v, int):
        return v
    if hasattr(v, "data_ptr"):
        return int(v.data_ptr())
    if v is None:
        return 0
    raise TypeError(f"cannot convert {type(v).__name__} to device pointer")


def empty_workspace(shape: Sequence[int], *, dtype: Any, device: Any):
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
) -> TorchLaunchSummary:
    rt = Runtime()
    module = rt.load_module(hsaco)
    fn = module.get_function(kernel_name)
    args = pack_args(signature, values)

    for _ in range(int(warmup)):
        rt.launch(fn, grid, block, args)
    rt.sync()

    e0 = rt.event()
    e1 = rt.event()
    e0.record()
    for _ in range(int(attempts)):
        rt.launch(fn, grid, block, args)
    e1.record()
    e1.synchronize()
    ms = e0.elapsed_to(e1) / int(attempts)
    e0.destroy()
    e1.destroy()
    module.unload()
    return TorchLaunchSummary(ms=ms, attempts=int(attempts))
