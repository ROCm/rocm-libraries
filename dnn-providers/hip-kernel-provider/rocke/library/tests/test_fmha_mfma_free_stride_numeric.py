# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric parity for the fmha_mfma free-stride (layout-general) path.

Runs only on a CDNA device (gfx942 / gfx950); skips elsewhere. For each physical
placement it builds ``fmha_mfma(free_stride=True)``, launches it, and checks the
output against the numpy dense-attention reference.

The placements are chosen so the kernel, not numpy, is what is under test:
besides the two contiguous orders, a padded head dim makes every stride carry
the padding, and packed QKV puts the three operands in one buffer at non-zero
base offsets. Feeding contiguous strides for a padded buffer, or dropping the
packed base offsets, makes these cases fail, which is what separates them from
the host-side layout tests in ``test_sdpa_free_stride_layouts.py``.

The build/ABI contract is covered CPU-only by ``test_fmha_mfma_free_stride.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocke.runtime.hip_module import get_device_arch

_ARCH = get_device_arch()
_CDNA = ("gfx942", "gfx950")

pytestmark = pytest.mark.skipif(
    _ARCH not in _CDNA, reason=f"needs a CDNA GPU (gfx942/gfx950); got {_ARCH}"
)

# One logical problem, reused so every placement is the same arithmetic.
_PROBLEM = dict(
    heads=8,
    head_size=64,
    seqlen_q=128,
    seqlen_k=128,
    batch=2,
)


@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
@pytest.mark.parametrize("causal", [False, True])
def test_free_stride_matches_reference(layout, causal):
    from builders.common.sdpa_fwd_free_stride_verify import run_verify

    r = run_verify(
        arch=_ARCH, layout=layout, kv_heads=2, causal=causal, verbose=False, **_PROBLEM
    )
    assert (
        r.passed
    ), f"{layout} causal={causal}: max_abs_diff={r.max_abs:.3e} bad={r.bad}/{r.total}"


@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
@pytest.mark.parametrize("causal", [False, True])
def test_padded_head_dim_matches_reference(layout, causal):
    # [B, H, S, D+8] sliced to D: the head dim keeps unit stride, but the batch,
    # head and token strides all carry the padding, so a kernel that recomputed
    # them from D instead of reading the kernargs would read the wrong rows.
    from builders.common.sdpa_fwd_free_stride_verify import run_verify

    r = run_verify(
        arch=_ARCH,
        layout=layout,
        kv_heads=2,
        causal=causal,
        head_pad=8,
        verbose=False,
        **_PROBLEM,
    )
    assert r.passed, (
        f"{layout} pad=8 causal={causal}: max_abs_diff={r.max_abs:.3e} "
        f"bad={r.bad}/{r.total}"
    )


@pytest.mark.parametrize("causal", [False, True])
def test_packed_qkv_matches_reference(causal):
    # One [B, S, 3, H, D] buffer: Q/K/V alias a single allocation and differ only
    # by base offset (0, H*D, 2*H*D), so the body's aligned vector loads have to
    # hold at a non-zero base. MHA, because the packed buffer has one head count.
    from builders.common.sdpa_fwd_free_stride_verify import run_verify

    r = run_verify(
        arch=_ARCH,
        layout="bshd",
        kv_heads=0,
        causal=causal,
        packed=True,
        verbose=False,
        **_PROBLEM,
    )
    assert (
        r.passed
    ), f"packed causal={causal}: max_abs_diff={r.max_abs:.3e} bad={r.bad}/{r.total}"


def test_placements_agree_bit_for_bit():
    # The free-stride invariant: one logical problem, four physical placements,
    # one answer. Changing where the operands live changes addresses, not the
    # order the kernel accumulates in, so the outputs must match exactly -- a
    # stronger and less flaky check than re-asserting each placement's tolerance.
    from builders.common.sdpa_fwd_free_stride_verify import run_verify

    common = dict(arch=_ARCH, kv_heads=0, causal=False, verbose=False, **_PROBLEM)
    placements = {
        "bhsd": dict(layout="bhsd"),
        "bshd": dict(layout="bshd"),
        "bhsd+pad8": dict(layout="bhsd", head_pad=8),
        "packed": dict(layout="bshd", packed=True),
    }
    outs = {name: run_verify(**common, **kw) for name, kw in placements.items()}
    for name, r in outs.items():
        assert r.passed, f"{name}: max_abs_diff={r.max_abs:.3e}"
    base = outs["bhsd"].out
    for name, r in outs.items():
        np.testing.assert_array_equal(
            r.out, base, err_msg=f"{name} disagrees with bhsd"
        )
