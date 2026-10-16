# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only build/smoke tests for the free-stride SDPA forward shell.

``kernels.common.fmha_sdpa_fwd`` is the net-new MFMA-body layout shell
(hipDNN requirement 2). These tests exercise the host-side contract with no
GPU and no torch: the spec validator gates, the kernel ABI signature, the
launch grid, and that a valid spec lowers to a ``KernelDef`` via the Python
engine. On-GPU numeric parity and byte-identity against the C++ engine are
separate (hardware / engine) lanes.
"""

from __future__ import annotations

import pytest

from kernels.common._fmha_common import FmhaCommonSpec, FmhaShape
from kernels.common.fmha_sdpa_fwd import (
    FmhaSdpaFwdSpec,
    build_fmha_sdpa_fwd,
    fmha_sdpa_fwd_grid,
    fmha_sdpa_fwd_signature,
    is_valid_spec,
)


def _spec(**over):
    kw = dict(head_size=64, num_query_heads=8, num_kv_heads=8)
    shape_kw = {k: over.pop(k) for k in ("head_size", "num_query_heads", "num_kv_heads") if k in over}
    kw.update(shape_kw)
    common = FmhaCommonSpec(shape=FmhaShape(**kw), dtype=over.pop("dtype", "f16"))
    base = dict(common=common, batch=2, seqlen_q=128, seqlen_k=128)
    base.update(over)
    return FmhaSdpaFwdSpec(**base)


# --- validator gates -------------------------------------------------------


def test_valid_spec_accepted():
    ok, why = is_valid_spec(_spec(), "gfx950")
    assert ok, why


@pytest.mark.parametrize(
    "over, needle",
    [
        (dict(seqlen_q=130), "BLOCK_M"),
        (dict(seqlen_k=130), "BLOCK_K"),
        (dict(head_size=40), "head_size"),
        (dict(batch=0), "batch"),
        (dict(seqlen_q=0), "seqlen_q"),
        (dict(layout="bshd"), "not yet implemented"),
        (dict(packed_qkv=True), "packed_qkv"),
    ],
)
def test_invalid_specs_rejected(over, needle):
    ok, why = is_valid_spec(_spec(**over), "gfx950")
    assert not ok
    assert needle in why


def test_bshd_names_itself_planned():
    _, why = is_valid_spec(_spec(layout="bshd"), "gfx950")
    assert "Phase 3" in why


# --- signature / grid ------------------------------------------------------


def test_signature_param_order():
    sig = fmha_sdpa_fwd_signature(_spec())
    names = [p["name"] for p in sig]
    assert names == [
        "Q", "K", "V", "O", "scale_log2",
        "stride_q_token", "stride_q_head",
        "stride_k_token", "stride_k_head",
        "stride_v_token", "stride_v_head",
        "stride_o_token", "stride_o_head",
        "stride_q_batch", "stride_k_batch", "stride_v_batch", "stride_o_batch",
    ]


def test_grid_is_tile_head_batch():
    assert fmha_sdpa_fwd_grid(_spec(seqlen_q=256, batch=3)) == (16, 8, 3)


# --- build (Python engine, CPU-only) --------------------------------------


def test_build_returns_named_kernel():
    kd = build_fmha_sdpa_fwd(_spec(), "gfx950")
    assert kd.name == _spec().kernel_name()


def test_build_rejects_invalid_spec():
    with pytest.raises(ValueError, match="invalid fmha_sdpa_fwd spec"):
        build_fmha_sdpa_fwd(_spec(layout="bshd"), "gfx950")


def test_build_gqa_shape():
    # num_query_heads != num_kv_heads exercises the decode_grid GQA divisor.
    kd = build_fmha_sdpa_fwd(_spec(num_query_heads=8, num_kv_heads=2), "gfx950")
    assert kd is not None


def test_build_bf16():
    kd = build_fmha_sdpa_fwd(_spec(dtype="bf16"), "gfx950")
    assert kd is not None


def test_signature_matches_built_kernel():
    # The throwaway-probe signature must match the real built kernel's params.
    spec = _spec()
    probe = fmha_sdpa_fwd_signature(spec)
    built = build_fmha_sdpa_fwd(spec, "gfx950").params
    assert [p["name"] for p in probe] == [p.name for p in built]
