# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dense SDPA launch and export mechanics, independent of candidate selection."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernels.common.attention_dense_spec import AttentionDenseSpec


def launch(
    module: ModuleType, spec: AttentionDenseSpec, buffers: dict, scale: float
) -> None:
    """Bind HIP buffers to the selected production module's public launch entry."""
    module.run_attention_dense_torch(
        spec=spec,
        q=buffers["q"],
        k=buffers["k"],
        v=buffers["v"],
        out=buffers["out"],
        scale=scale,
    )


def exported_kernel(
    module: ModuleType, spec: AttentionDenseSpec, architecture: str
) -> tuple[bytes, dict]:
    """Export the cached launcher that executed, using production geometry helpers."""
    from kernels.common.attention_dense_spec import attention_dense_cache_key

    launcher = module._DENSE_LAUNCHER_CACHE[
        attention_dense_cache_key(spec, arch=architecture)
    ]
    return launcher._hsaco, {
        "name": launcher.kernel_name,
        "signature": launcher.signature,
        "grid": module.attention_dense_grid(spec),
        "block": module.attention_dense_block(spec),
        "runtime_shape": spec.runtime_shape,
    }
