# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side checks of the dense-attention ``score_identity`` spec field:
the back-to-back GEMM ``E = scale * (Q K^T) V`` with the intermediate kept in
registers (no running max, no exp, no 1/l).

Pure codegen (no GPU): the field is off by default (every existing symbol and
body unchanged), renames the kernel, enters the cache key, removes the softmax
exponentials, and rejects non-bool values and the modes it does not support.
"""

from __future__ import annotations

import pytest
from kernels.common.attention_dense_spec import attention_dense_cache_key
from kernels.gfx950.attention_dense import (
    Gfx950AttentionDenseSpec,
    build_attention_dense,
)
from rocke.core.lower_llvm import lower_kernel_to_llvm


def _spec(**kw):
    d = dict(
        batch=1,
        seqlen_q=1024,
        seqlen_kv=1024,
        num_query_heads=8,
        num_kv_heads=8,
        head_size=64,
        causal=False,
        dtype="bf16",
        block_m=128,
        block_n=64,
        lazy_rescale=False,
    )
    d.update(kw)
    return Gfx950AttentionDenseSpec(**d)


@pytest.mark.parametrize("head_size", [64, 128])
def test_score_identity_builds_without_softmax(head_size):
    plain_spec = _spec(head_size=head_size)
    b2b_spec = _spec(head_size=head_size, score_identity=True)
    plain = build_attention_dense(plain_spec)
    b2b = build_attention_dense(b2b_spec)
    assert plain.name == plain_spec.kernel_name() and "b2b" not in plain.name
    assert b2b.name == b2b_spec.kernel_name() == plain.name + "_b2b"
    ir0 = lower_kernel_to_llvm(plain, arch="gfx950")
    ir1 = lower_kernel_to_llvm(b2b, arch="gfx950")
    assert "exp2" in ir0
    assert "exp2" not in ir1
    assert ir1.count("mfma") == ir0.count("mfma")
    assert [p.name for p in b2b.params] == [p.name for p in plain.params]


def test_score_identity_is_in_the_cache_key():
    plain, b2b = _spec(), _spec(score_identity=True)
    assert attention_dense_cache_key(plain, arch="gfx950") != (
        attention_dense_cache_key(b2b, arch="gfx950")
    )


@pytest.mark.parametrize("value", ["false", 1, 0, None])
def test_score_identity_rejects_non_bool(value):
    with pytest.raises(ValueError, match="score_identity must be a bool"):
        _spec(score_identity=value)


@pytest.mark.parametrize(
    "kw",
    [
        dict(lazy_rescale=True),
        dict(causal=True),
        dict(emit_lse=True),
        dict(sliding_window=128),
        dict(seqlen_kv=1000, ragged=True),
        dict(use_sinks=True),
        dict(persistent=True),
        dict(kv_segments=2),
    ],
)
def test_score_identity_rejects_unsupported_modes(kw):
    with pytest.raises(ValueError, match="score_identity is not supported with"):
        _spec(score_identity=True, **kw)
