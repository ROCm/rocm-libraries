# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric parity for the fmha_mfma free-stride (layout-general) path.

Runs only on a CDNA device (gfx942 / gfx950); skips elsewhere. For each
contiguous layout it builds ``fmha_mfma(free_stride=True)``, launches it, and
checks the output against the numpy dense-attention reference. The point is
layout invariance: BSHD and BHSD must both match the reference (and therefore
each other), proving the per-batch base-pointer rebase reads and writes every
layout correctly. The build/ABI contract is covered CPU-only by
``test_fmha_mfma_free_stride.py``.
"""

from __future__ import annotations

import pytest

from rocke.runtime.hip_module import get_device_arch

_ARCH = get_device_arch()
_CDNA = ("gfx942", "gfx950")

pytestmark = pytest.mark.skipif(
    _ARCH not in _CDNA, reason=f"needs a CDNA GPU (gfx942/gfx950); got {_ARCH}"
)


@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
@pytest.mark.parametrize("causal", [False, True])
def test_free_stride_matches_reference(layout, causal):
    from builders.common.sdpa_fwd_free_stride_verify import run_verify

    passed, max_abs, bad, total = run_verify(
        arch=_ARCH,
        layout=layout,
        heads=8,
        kv_heads=2,  # GQA
        head_size=64,
        seqlen_q=128,
        seqlen_k=128,
        batch=2,
        causal=causal,
        verbose=False,
    )
    assert (
        passed
    ), f"{layout} causal={causal}: max_abs_diff={max_abs:.3e} bad={bad}/{total}"


def test_bshd_and_bhsd_agree():
    # Same logical problem, two layouts: the kernel outputs must match bit-for-
    # bit-close, which is the core free-stride invariant.
    from builders.common.sdpa_fwd_free_stride_verify import run_verify

    common = dict(
        arch=_ARCH,
        heads=8,
        kv_heads=2,
        head_size=64,
        seqlen_q=128,
        seqlen_k=128,
        batch=2,
        causal=False,
        verbose=False,
    )
    ok_a, a, _, _ = run_verify(layout="bhsd", **common)
    ok_b, b, _, _ = run_verify(layout="bshd", **common)
    assert ok_a and ok_b, f"bhsd={a:.3e} bshd={b:.3e}"
