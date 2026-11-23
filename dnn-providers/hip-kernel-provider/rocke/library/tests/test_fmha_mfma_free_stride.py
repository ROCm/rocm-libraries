# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU build tests for the fmha_mfma free-stride (layout-general) path.

fmha_mfma gains an opt-in ``free_stride`` mode: per-tensor batch strides plus a
base-pointer rebase so one kernel reads any layout (BSHD / BHSD / general /
packed), replacing the BSHD-contiguous seqlen fold. The default path must stay
byte-identical, so these tests lock the ABI contract: ``free_stride=False`` is
unchanged and ``free_stride=True`` appends exactly the four batch strides and
tags the kernel name.

On-GPU numeric parity is the verify harness
(``builders/common/sdpa_fwd_free_stride_verify.py``); byte-identity of the
default path is the repo's own byte-identity gate.
"""

from __future__ import annotations

import pytest

from kernels.common._fmha_common import FmhaCommonSpec, FmhaShape
from kernels.common.fmha_mfma import (
    FmhaMfmaSpec,
    build_fmha_fwd_mfma,
    fmha_fwd_mfma_signature,
)

_BATCH = ("stride_q_batch", "stride_k_batch", "stride_v_batch", "stride_o_batch")


def _spec(free_stride=False, **over):
    shape = FmhaShape(
        head_size=over.pop("head_size", 64),
        num_query_heads=over.pop("num_query_heads", 8),
        num_kv_heads=over.pop("num_kv_heads", 8),
    )
    common = FmhaCommonSpec(shape=shape, mask_mode=over.pop("mask_mode", "none"))
    return FmhaMfmaSpec(
        common=common, seqlen_q=128, seqlen_k=128, free_stride=free_stride
    )


def test_default_abi_has_no_batch_strides():
    names = [p["name"] for p in fmha_fwd_mfma_signature(_spec())]
    assert len(names) == 15
    assert not any(n in _BATCH for n in names)


def test_default_name_untagged():
    assert not _spec().kernel_name().endswith("_fs")


def test_free_stride_abi_appends_batch_strides():
    names = [p["name"] for p in fmha_fwd_mfma_signature(_spec(free_stride=True))]
    assert len(names) == 19
    assert tuple(names[-4:]) == _BATCH


def test_free_stride_name_tagged():
    assert _spec(free_stride=True).kernel_name().endswith("_fs")


@pytest.mark.parametrize("arch", ["gfx950", "gfx942"])
@pytest.mark.parametrize("mask", ["none", "causal"])
def test_free_stride_builds(arch, mask):
    kd = build_fmha_fwd_mfma(_spec(free_stride=True, mask_mode=mask), arch)
    assert kd.name.endswith("_fs")


def test_free_stride_gqa_builds():
    kd = build_fmha_fwd_mfma(
        _spec(free_stride=True, num_query_heads=8, num_kv_heads=2), "gfx950"
    )
    assert kd is not None
