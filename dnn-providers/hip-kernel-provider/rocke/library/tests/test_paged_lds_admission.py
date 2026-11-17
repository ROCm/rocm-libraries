# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Bounded LDS accounting charges the emitted pools and uses architecture capacity."""

import re
from importlib import import_module
from types import SimpleNamespace

import pytest

from kernels.common._attention_paged_bounds import (
    _paged_segment_lds_bytes,
    _validate_paged_segment_lds,
)
from rocke.core.arch import ArchTarget
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python


def _spec(arch, head_size=256, dtype="fp16"):
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    return module.UnifiedAttention3DTiledSpec(
        head_size=head_size,
        block_size=16,
        tile_size_override=32,
        num_query_heads=8,
        num_kv_heads=2,
        dtype=dtype,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_segments=8,
        num_seqs=3,
        use_decode_grid=True,
        metadata_containment=True,
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "head_size,expected", [(64, 14592), (128, 28928), (256, 57600)]
)
def test_bounded_lds_estimate_matches_emitted_pool(
    arch, dtype, head_size, expected, monkeypatch
):
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm23")
    spec = _spec(arch, head_size, dtype)
    assert _paged_segment_lds_bytes(spec) == expected
    _validate_paged_segment_lds(spec, arch)
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    text = _lower_kernel_to_llvm_python(
        module.build_unified_attention_3d_tiled(spec, arch=arch), arch=arch
    )
    pools = re.findall(r"addrspace\(3\) global \[(\d+) x i8\]", text)
    assert pools == [str(expected)]


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("capacity,accepted", [(57600, True), (57584, False)])
def test_exact_capacity_and_over_budget_rejection(
    arch, capacity, accepted, monkeypatch
):
    spec = _spec(arch)
    # A synthetic resource profile tests the capacity predicate independently
    # of today's catalog. It is not a claim of GPU validation on that profile.
    target = SimpleNamespace(
        lds_capacity_bytes=capacity,
        fits_lds=lambda size: size <= capacity,
    )
    monkeypatch.setattr(ArchTarget, "from_gfx", lambda name: target)
    if accepted:
        _validate_paged_segment_lds(spec, arch)
    else:
        with pytest.raises(ValueError, match=rf"requires 57600 B LDS.*{capacity} B"):
            _validate_paged_segment_lds(spec, arch)
