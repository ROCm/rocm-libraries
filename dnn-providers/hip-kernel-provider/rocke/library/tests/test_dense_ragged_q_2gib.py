# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 dense ragged Q: the preflight bounds Q in elements, the load in bytes.

With ``ragged=True`` the gfx950 dense kernels (grid and persistent) load Q
through a buffer resource sized ``B * Sq * Hq * D * 2`` bytes, emitted through
``const_i32``, with ``voffset = (q_base + q_tok * stride + col) * 2`` built
from ``mul nsw`` / ``add nsw`` i32. ``check_dense_spec_preflight`` rejects
K/V at ``>= 2^31`` *bytes* but Q/O only at ``>= 2^31`` *elements*, although its
own docstring says the signed i32 bound is the binding one on both paths.

So a bf16 shape with ``2^30 <= Q elements < 2^31`` is admitted and emits

- an ``i32`` literal above ``INT32_MAX`` for ``num_records``
  (B=64: ``i32 2149580800``), and
- a ``mul nsw i32`` that overflows for every Q token at or past
  ``2^31 / (Hq * D * 2)`` = token 262144 = batch 63, row 3844: signed
  overflow, i.e. poison in LLVM IR.

What ROCm 7.1 COMGR (LLVM 20) makes of it for gfx950: the literal truncates
to the same u32 (``s_mov_b32 s26, 0x80200000``), and the voffset is a plain 32-bit
``v_lshlrev_b32``/``v_add_lshl_u32``, so the byte offset wraps to the
correct unsigned value. The buffer range check compares unsigned (probed on
gfx90a: voffset ``0x80000000`` with ``num_records`` ``0x80001000`` reads
memory). The output is therefore expected to be correct today; the IR is
nevertheless undefined, and any LLVM that exploits the ``nsw`` may break it.
``test_admitted_ragged_q_rsrc_fits_i32`` pins the IR defect (CPU, fails now);
``test_ragged_q_past_2gib_bytes`` checks the rows past the boundary on a
gfx950 (expected to pass on the current toolchain; it is the numeric
backstop if that changes). B=63 keeps Q under 2^31 bytes and is the control.
"""

from __future__ import annotations

import re

import pytest

_S, _HQ, _HKV, _D = 4100, 32, 8, 128  # 4100 is not a tile multiple -> ragged
_INT32_MAX = (1 << 31) - 1
_SPEC_IDS = ["gfx950_dense_grid", "gfx950_dense_persist"]
# llvm20: ``i32 <lit>``; llvm22+: ``i64 %nb`` with ``%nb = zext i32 <lit> to i64``.
_RSRC_Q = re.compile(
    r"make\.buffer\.rsrc\.(?:p8\.)?p1\(ptr addrspace\(1\) %q_ptr, i16 0, "
    r"i(?:32|64) (%?[\w.-]+),"
)


def _spec(batch: int, spec_id: str):
    from dispatch.attention import AttentionRequest, attention_tuning_spec

    return attention_tuning_spec(
        AttentionRequest(
            batch=batch,
            nhead_q=_HQ,
            nhead_k=_HKV,
            seqlen_q=_S,
            seqlen_k=_S,
            hdim_q=_D,
            hdim_v=_D,
            arch="gfx950",
            mask_type=1,
            dtype="bf16",
            use_sinks=False,
            sliding_window=0,
        ),
        spec_id,
    ).kernel_spec


def _q_rsrc_sizes(ll: str) -> list[int]:
    """The constant ``num_records`` operands of every Q buffer resource."""
    sizes = []
    for operand in _RSRC_Q.findall(ll):
        if operand.startswith("%"):
            m = re.search(rf"{re.escape(operand)} = zext i32 (-?\d+) to i64", ll)
            assert m, f"Q resource size {operand} is not a zext'd i32 constant"
            operand = m.group(1)
        sizes.append(int(operand))
    return sizes


def _q_bytes(batch: int) -> int:
    return batch * _S * _HQ * _D * 2


# ---------------------------------------------------------------------------
# IR check (no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec_id", _SPEC_IDS)
@pytest.mark.parametrize("batch", [63, 64], ids=["q_under_2gib", "q_over_2gib"])
def test_admitted_ragged_q_rsrc_fits_i32(batch, spec_id):
    """An admitted ragged spec must not size its Q resource past INT32_MAX."""
    from kernels.gfx950.attention_dense import (
        build_attention_dense,
        supports_attention_dense,
    )
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    spec = _spec(batch, spec_id)
    assert spec.ragged
    ok, _why = supports_attention_dense(spec, arch="gfx950")
    if not ok:
        return  # refused: acceptable
    ll = lower_kernel_to_llvm(build_attention_dense(spec, arch="gfx950"), arch="gfx950")
    sizes = _q_rsrc_sizes(ll)
    assert sizes, "ragged spec emitted no Q buffer resource"
    q_elems = batch * _S * _HQ * _D
    assert max(sizes) <= _INT32_MAX, (
        f"{spec_id} B={batch}: preflight admitted Q = {q_elems} elements "
        f"(< 2^31) = {_q_bytes(batch)} bytes (>= 2^31); the Q resource is "
        f"i32 {max(sizes)} and the byte voffset (mul nsw i32 ..., 2) overflows "
        f"from token {(1 << 31) // (_HQ * _D * 2)} on"
    )


# ---------------------------------------------------------------------------
# On-GPU check (gfx950)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gfx950_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from rocke.runtime.hip_module import get_device_arch

    if get_device_arch() != "gfx950":
        pytest.skip("the ragged dense kernels are gfx950-only")
    return torch


def _reference_rows(torch, q, k, v, scale, r0):
    """fp32 causal attention for query rows [r0, S) of one batch ([1,S,H,D])."""
    import torch.nn.functional as F

    rep = _HQ // _HKV
    qf = q[:, r0:].transpose(1, 2).float()
    kf = k.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    vf = v.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    qi = torch.arange(r0, _S, device=q.device).view(-1, 1)
    ki = torch.arange(_S, device=q.device).view(1, -1)
    ref = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=ki <= qi, scale=scale)
    return ref.transpose(1, 2)  # [1, S - r0, Hq, D]


@pytest.mark.gpu
@pytest.mark.parametrize("spec_id", _SPEC_IDS)
@pytest.mark.parametrize("batch", [63, 64], ids=["q_under_2gib", "q_over_2gib"])
def test_ragged_q_past_2gib_bytes(gfx950_torch, batch, spec_id):
    """Rows whose Q byte offset is >= 2^31 must match the fp32 reference."""
    from kernels.gfx950.attention_dense import run_attention_dense_torch

    torch = gfx950_torch
    spec = _spec(batch, spec_id)
    torch.cuda.empty_cache()  # earlier cases' cached blocks count as used
    need = 2 * _q_bytes(batch) + 2 * batch * _S * _HKV * _D * 2
    free_bytes, _total = torch.cuda.mem_get_info()
    if free_bytes < need + (1 << 30):
        pytest.skip(f"needs {need >> 20} MiB free")
    torch.manual_seed(0)
    q = torch.randn(batch, _S, _HQ, _D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, _S, _HKV, _D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, _S, _HKV, _D, dtype=torch.bfloat16, device="cuda")
    out = torch.full_like(q, float("nan"))
    scale = _D**-0.5
    run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
    torch.cuda.synchronize()

    first_tok = (1 << 31) // (_HQ * _D * 2)  # 262144
    r0 = 3584  # tile-aligned start below the predicted row 3844 of batch 63
    report = []
    for bi in (0, batch - 1):
        ref = _reference_rows(
            torch, q[bi : bi + 1], k[bi : bi + 1], v[bi : bi + 1], scale, r0
        )
        err = (out[bi : bi + 1, r0:].float() - ref).abs().amax(dim=(0, 2, 3))
        bad = (err > 4e-2) | err.isnan()  # bf16 tolerance used by the repo
        if bad.any():
            rows = (bad.nonzero().flatten() + r0).tolist()
            report.append(
                f"batch {bi}: {len(rows)} bad rows from {rows[0]} "
                f"(max err {err.nan_to_num(1e9).max().item():.3e})"
            )
    assert not report, (
        f"{spec_id} B={batch} (Q {_q_bytes(batch)} B; first token past 2^31 "
        f"bytes = {first_tok} = batch {first_tok // _S} row {first_tok % _S}): "
        + "; ".join(report)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
