# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side checks of the dense-attention ``spec._score_identity`` side
channel: the back-to-back GEMM ``E = scale * (Q K^T) V`` with the
intermediate kept in registers (no running max, no exp, no 1/l).

Pure codegen (no GPU): the side channel is absent by default, renames the
kernel, removes the softmax exponentials, and rejects the modes it does not
support.
"""

from __future__ import annotations

import pytest
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


def _ident(sp):
    object.__setattr__(sp, "_score_identity", True)
    return sp


@pytest.mark.parametrize("head_size", [64, 128])
def test_score_identity_builds_without_softmax(head_size):
    plain = build_attention_dense(_spec(head_size=head_size))
    b2b = build_attention_dense(_ident(_spec(head_size=head_size)))
    assert b2b.name == plain.name + "_b2b"
    ir0 = lower_kernel_to_llvm(plain, arch="gfx950")
    ir1 = lower_kernel_to_llvm(b2b, arch="gfx950")
    assert "exp2" in ir0
    assert "exp2" not in ir1
    assert ir1.count("mfma") == ir0.count("mfma")
    assert [p.name for p in b2b.params] == [p.name for p in plain.params]


@pytest.mark.parametrize(
    "kw", [dict(lazy_rescale=True), dict(causal=True), dict(emit_lse=True)]
)
def test_score_identity_rejects_unsupported_modes(kw):
    with pytest.raises(ValueError, match="_score_identity"):
        build_attention_dense(_ident(_spec(**kw)))
