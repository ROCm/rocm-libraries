# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only contract for the optional unified split-KV (3D) LSE output."""

import importlib

import pytest

from rocke.core.lower_llvm import lower_kernel_to_llvm

_ARCHES = ("gfx942", "gfx950")


def _reduce_module(arch):
    return importlib.import_module(f"kernels.{arch}.attention_tiled_3d")


def _reduce_spec(arch, **overrides):
    mod = _reduce_module(arch)
    fields = dict(
        head_size=128,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        num_segments=16,
    )
    fields.update(overrides)
    return mod.UnifiedAttentionReduceTiledSpec(**fields)


def _reduce_ir(arch, spec):
    mod = _reduce_module(arch)
    kernel = mod.build_unified_attention_reduce_tiled(spec, arch=arch)
    return lower_kernel_to_llvm(kernel, arch=arch)


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("bad", ["false", 0, 1, None])
def test_reduce_spec_rejects_non_boolean_emit_lse(arch, bad):
    with pytest.raises(ValueError, match="emit_lse must be a bool"):
        _reduce_spec(arch, emit_lse=bad)


@pytest.mark.parametrize("arch", _ARCHES)
def test_reduce_lse_is_a_distinct_kernel_with_one_more_pointer(arch):
    off = _reduce_spec(arch)
    on = _reduce_spec(arch, emit_lse=True)
    assert on.kernel_name() == off.kernel_name() + "_lse"
    off_ir = _reduce_ir(arch, off)
    on_ir = _reduce_ir(arch, on)
    assert "lse_ptr" not in off_ir
    assert "lse_ptr" in on_ir
    assert "@llvm.log2.f32" in on_ir
