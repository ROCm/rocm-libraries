# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The score hook on the MFMA and WMMA attention forward bodies costs no IR.

``extra_score_transform`` receives each score cell's query and key positions.
They are the values the mask already computes, so an identity transform must
leave the IR unchanged. The byte-identity gate covers the default (no-callback)
IR; this test drives the real ``fmha_mfma`` kernel with the callback injected.
Row epilogues (LSE output) go through ``AttnFwdExt.epilogue_hook``, covered by
``platform/tests/instances/test_attention_fwd_ext_ir.py``.
"""

from __future__ import annotations

import pytest

from kernels.common import fmha_mfma
from kernels.common._fmha_common import FmhaCommonSpec, FmhaShape
from rocke.core.ir import Value
from rocke.core.lower_llvm import lower_kernel_to_llvm

# MFMA (gfx950, gfx942) and WMMA (gfx1151) bodies.
ARCHES = ["gfx950", "gfx942", "gfx1151"]


def _spec(mask_mode: str) -> fmha_mfma.FmhaMfmaSpec:
    common = FmhaCommonSpec(
        shape=FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=8),
        dtype="f16",
        mask_mode=mask_mode,
        sliding_window=32 if mask_mode == "sliding_window" else 0,
        scale_log2=0.0,
    )
    return fmha_mfma.FmhaMfmaSpec(common=common, seqlen_q=128, seqlen_k=256)


def _build_ir(monkeypatch, arch: str, mask_mode: str, **callbacks) -> str:
    body = fmha_mfma.mfma_attention_fwd_inner_body
    if callbacks:
        monkeypatch.setattr(
            fmha_mfma,
            "mfma_attention_fwd_inner_body",
            lambda *args, **kwargs: body(*args, **kwargs, **callbacks),
        )
    kernel = fmha_mfma.build_fmha_fwd_mfma(_spec(mask_mode), arch=arch)
    monkeypatch.undo()
    return lower_kernel_to_llvm(kernel, arch=arch)


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("mask_mode", ["none", "causal", "sliding_window"])
def test_score_coordinates_cost_no_ir(monkeypatch, arch, mask_mode):
    seen = []

    def identity(b, score, kt, row_in_atom, query_pos, key_pos):
        seen.append((row_in_atom, query_pos, key_pos))
        return score

    baseline = _build_ir(monkeypatch, arch, mask_mode)
    with_hook = _build_ir(monkeypatch, arch, mask_mode, extra_score_transform=identity)
    assert with_hook == baseline
    assert seen and all(isinstance(v, Value) for _, q, k in seen for v in (q, k))
