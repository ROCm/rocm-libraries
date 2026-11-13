# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The LLVM IR the Python builders give for a ``<arch>_attention_tiled_3d_lower_llvm``
call of the ``rocke_engine`` binding.

On gfx942 and gfx950 the binding lowers the segment kernel, then the split-KV
reduce it derives from the segment spec: head size, head counts, dtype, segment
count and ``waves_per_eu`` copied over, no LSE output. The native-builder tests
compare the binding against this text, so the derivation has one home here.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

if TYPE_CHECKING:
    from kernels.gfx942.attention_tiled_3d import (
        UnifiedAttention3DTiledSpec as Gfx942SegmentSpec,
    )
    from kernels.gfx950.attention_tiled_3d import (
        UnifiedAttention3DTiledSpec as Gfx950SegmentSpec,
    )


def lower_like_tiled_3d_binding(
    segment_spec: Gfx942SegmentSpec | Gfx950SegmentSpec, arch: str
) -> str:
    """Python-lowered segment kernel of ``segment_spec``, then its derived reduce."""
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    reduce_spec = module.UnifiedAttentionReduceTiledSpec(
        head_size=segment_spec.head_size,
        num_query_heads=segment_spec.num_query_heads,
        num_kv_heads=segment_spec.num_kv_heads,
        dtype=segment_spec.dtype,
        num_segments=segment_spec.num_segments,
        waves_per_eu=segment_spec.waves_per_eu,
    )
    segment = module.build_unified_attention_3d_tiled(segment_spec, arch=arch)
    reduce = module.build_unified_attention_reduce_tiled(reduce_spec, arch=arch)
    return _lower_kernel_to_llvm_python(
        segment, arch=arch
    ) + _lower_kernel_to_llvm_python(reduce, arch=arch)
