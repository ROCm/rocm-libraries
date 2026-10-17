# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Cache-key contract for the scalar unified-attention fallback.

The scalar 2D / 3D / reduce kernels read the dynamic shape (``total_q``,
``num_seqs``, ``max_seqlen_q``, ``max_seqlen_k``) only through kernargs and the
launch grid. The block-table row stride is the ``block_table_stride`` kernarg,
so a caller's table may be wider than ``ceil(max_seqlen_k / block_size)``.
``_cache_key`` therefore omits the dynamic shape, and one binary serves every
batch.

Layers:
  * Shapes that differ only in the dynamic shape share a key and lower to
    byte-identical IR at every LLVM flavor.
  * Every static field the build reads splits the key.
  * The launch signature matches the kernel's params, including the stride.

No GPU required. ``test_attention_scalar_block_table_numeric.py`` runs the
launch path on hardware.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from kernels.common import attention_unified as au
from rocke.core.lower_llvm import LLVM_FLAVORS, _lower_kernel_to_llvm_python

P = au.UnifiedAttentionProblem

_BASE = P(
    total_q=64,
    num_seqs=4,
    num_query_heads=8,
    num_kv_heads=2,
    head_size=128,
    block_size=16,
    max_seqlen_q=16,
    max_seqlen_k=256,
    dtype="bf16",
)

# Each entry moves only the dynamic shape. max_seqlen_k=1000 changes the block
# count, which used to be compiled into the block-table row stride.
_DYNAMIC_VARIANTS = (
    dict(total_q=200),
    dict(num_seqs=1, total_q=16),
    dict(num_seqs=37, total_q=37, max_seqlen_q=1),
    dict(max_seqlen_q=50),
    dict(max_seqlen_k=250),
    dict(max_seqlen_k=1000),
    dict(total_q=1, num_seqs=1, max_seqlen_q=1, max_seqlen_k=65536),
)

# Each entry moves one static field the scalar build reads.
_STATIC_VARIANTS = (
    dict(num_query_heads=16),
    dict(num_kv_heads=8),
    dict(head_size=64),
    dict(block_size=32),
    dict(dtype="fp16"),
    dict(sliding_window=128),
    dict(use_sinks=True),
    dict(softcap=30.0),
)


def _lower_all(problem: P, flavor: str) -> tuple[str, str, str]:
    kernels = (
        au.build_unified_attention_2d(au.UnifiedAttention2DSpec(problem)),
        au.build_unified_attention_3d(au.UnifiedAttention3DSpec(problem)),
        au.build_unified_attention_reduce(au.UnifiedAttentionReduceSpec(problem, 8)),
    )
    return tuple(
        _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor=flavor)
        for k in kernels
    )


@pytest.mark.parametrize("change", _DYNAMIC_VARIANTS, ids=repr)
def test_dynamic_shape_shares_key(change):
    other = dataclasses.replace(_BASE, **change)
    assert au._cache_key(other, "gfx950") == au._cache_key(_BASE, "gfx950")


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_dynamic_shape_lowers_identical_ir(flavor):
    """Same key means same binary: every variant must lower byte-identically
    for the 2D, 3D and reduce scalar kernels, kernel name included."""
    base = _lower_all(_BASE, flavor)
    for change in _DYNAMIC_VARIANTS:
        other = _lower_all(dataclasses.replace(_BASE, **change), flavor)
        for kind, a, b in zip(("2d", "3d", "reduce"), base, other):
            assert a == b, f"{kind} IR differs for {change} at {flavor}"


@pytest.mark.parametrize("change", _STATIC_VARIANTS, ids=repr)
def test_static_field_splits_key_and_ir(change):
    other = dataclasses.replace(_BASE, **change)
    assert au._cache_key(other, "gfx950") != au._cache_key(_BASE, "gfx950")
    a = _lower_all(_BASE, "llvm20")[0]
    b = _lower_all(other, "llvm20")[0]
    assert a != b, f"{change} splits the key but not the 2D IR"


def test_arch_splits_key():
    assert au._cache_key(_BASE, "gfx942") != au._cache_key(_BASE, "gfx950")


def test_block_table_stride_is_a_kernarg():
    """The row stride must come from the launch, not ``max_seqlen_k``."""
    ll = _lower_all(_BASE, "llvm20")
    for kind, text in zip(("2d", "3d"), ll[:2]):
        assert "i32 %block_table_stride)" in text, kind
        assert re.search(r"mul nsw i32 %\w+, %block_table_stride$", text, re.M), kind


def test_launch_signature_matches_kernel_params():
    """``_get_scalar_launcher`` packs kernargs by this signature; a name or
    order mismatch with the kernel's params silently shifts every argument."""
    kernel = au.build_unified_attention_2d(au.UnifiedAttention2DSpec(_BASE))
    signature = au._attn_signature(
        _BASE.dtype, include_bt_stride=True, kv_dtype=au._kv_storage_dtype(_BASE)
    )
    assert [item["name"] for item in signature] == [p.name for p in kernel.params]
