# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 3D decode LDS footprint: head_size 192 stages Q through the V slot."""

from __future__ import annotations

import re

import pytest
from kernels.gfx942.attention_tiled_3d import (
    UnifiedAttention3DTiledSpec,
    build_unified_attention_3d_tiled,
)
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python


def _lds_bytes(head_size: int) -> int:
    spec = UnifiedAttention3DTiledSpec(
        head_size=head_size,
        block_size=16,
        num_query_heads=96,
        num_kv_heads=96,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_segments=16,
        num_seqs=1,
        use_decode_grid=True,
    )
    ll = _lower_kernel_to_llvm_python(
        build_unified_attention_3d_tiled(spec, arch="gfx942"),
        arch="gfx942",
        llvm_flavor="llvm20",
    )
    sizes = re.findall(r"addrspace\(3\) global \[(\d+) x i8\]", ll)
    assert sizes, "no LDS pool in the lowered kernel"
    return sum(int(s) for s in sizes)


@pytest.mark.parametrize(
    "head_size, expected",
    [
        # Q (16 x HD) + K (2 x 16 x HD) + V (16 x HD), bf16.
        (128, (16 + 32 + 16) * 128 * 2),
        # Q staged through the V slot: K (2 x 16 x HD) + V (16 x HD).
        (192, (32 + 16) * 192 * 2),
    ],
)
def test_3d_lds_footprint(head_size, expected):
    assert _lds_bytes(head_size) == expected
