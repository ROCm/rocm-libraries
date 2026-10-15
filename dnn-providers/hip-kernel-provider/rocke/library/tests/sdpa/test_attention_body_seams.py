# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Optional hooks on the MFMA and WMMA attention forward bodies.

``extra_score_transform`` receives each score cell's query and key positions,
and ``epilogue_fn`` receives each row's final max and denominator. Neither may
cost IR when unused: the coordinates are the values the mask already computes,
and ``epilogue_fn=None`` emits nothing. The byte-identity gate covers the
default (no-callback) IR; these tests drive the real ``fmha_mfma`` kernel with
the callbacks injected.
"""

from __future__ import annotations

import pytest

from kernels.common import fmha_mfma
from kernels.common._fmha_common import FmhaCommonSpec, FmhaShape
from rocke.core.ir import Value
from rocke.core.lower_llvm import lower_kernel_to_llvm

# (arch, accumulator row slots per lane): MFMA 16x16x16 holds 4, WMMA holds 8.
ARCHES = [("gfx950", 4), ("gfx942", 4), ("gfx1151", 8)]


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


@pytest.mark.parametrize("arch, _slots", ARCHES)
@pytest.mark.parametrize("mask_mode", ["none", "causal", "sliding_window"])
def test_score_coordinates_cost_no_ir(monkeypatch, arch, _slots, mask_mode):
    seen = []

    def identity(b, score, kt, row_in_atom, query_pos, key_pos):
        seen.append((row_in_atom, query_pos, key_pos))
        return score

    baseline = _build_ir(monkeypatch, arch, mask_mode)
    with_hook = _build_ir(monkeypatch, arch, mask_mode, extra_score_transform=identity)
    assert with_hook == baseline
    assert seen and all(isinstance(v, Value) for _, q, k in seen for v in (q, k))


@pytest.mark.parametrize("arch, slots", ARCHES)
def test_epilogue_sees_every_row_slot(monkeypatch, arch, slots):
    calls = []

    def epilogue(b, *, m_log2, l, row, row_valid):
        calls.append((m_log2, l, row, row_valid))

    _build_ir(monkeypatch, arch, "causal", epilogue_fn=epilogue)
    assert len(calls) == slots
    for m_log2, l, row, row_valid in calls:
        assert all(isinstance(v, Value) for v in (m_log2, l, row))
        assert row_valid is None
    assert len({id(c[2]) for c in calls}) == slots  # one row index per slot


@pytest.mark.parametrize("arch, _slots", ARCHES)
def test_epilogue_store_reaches_ir(monkeypatch, arch, _slots):
    """An epilogue that stores each row's fp32 ``l`` adds those stores to the IR."""

    def store_l(b, *, m_log2, l, row, row_valid):
        b.global_store(store_l.out, row, l, align=4)

    def capture_out(*args, **kwargs):
        store_l.out = kwargs["O"]
        return body(*args, **kwargs, epilogue_fn=store_l)

    body = fmha_mfma.mfma_attention_fwd_inner_body
    baseline = _build_ir(monkeypatch, arch, "none")
    monkeypatch.setattr(fmha_mfma, "mfma_attention_fwd_inner_body", capture_out)
    kernel = fmha_mfma.build_fmha_fwd_mfma(_spec("none"), arch=arch)
    monkeypatch.undo()
    with_store = lower_kernel_to_llvm(kernel, arch=arch)
    assert with_store != baseline
    assert with_store.count("store float") > baseline.count("store float")


def test_wmma_epilogue_gets_the_row_bound(monkeypatch):
    """With a Q-length bound, ``row_valid`` is the predicate the O store uses."""
    calls = []

    def epilogue(b, *, m_log2, l, row, row_valid):
        calls.append(row_valid)

    body = fmha_mfma.mfma_attention_fwd_inner_body

    def bounded(*args, **kwargs):
        # Any i32 Value works as the bound; the test checks the plumbing only.
        return body(
            *args, **kwargs, wmma_seqlen_q=kwargs["seqlen_k"], epilogue_fn=epilogue
        )

    monkeypatch.setattr(fmha_mfma, "mfma_attention_fwd_inner_body", bounded)
    fmha_mfma.build_fmha_fwd_mfma(_spec("none"), arch="gfx1151")
    assert len(calls) == 8
    assert all(isinstance(v, Value) for v in calls)
