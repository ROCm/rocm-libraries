# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shape cases, input construction, and the comparison helper for the
gated-delta-rule state-scan study.

A deliberately small subset of the FlyDSL harness's case table — enough to
cover the four behaviours that actually distinguish code paths:

===========  ==========================================================
``gdn``      scalar gate; GQA ratio > 1
``kda``      per-channel gate; GQA ratio 1
``varlen``   several ragged segments through ``cu_seqlens``
``tail``     ``T % BT != 0`` — exercises the tail-chunk row mask
===========  ==========================================================

Input construction mirrors the FlyDSL harness exactly (same magnitudes, same
gate cumsum semantics), so a divergence against the reference is a kernel bug
rather than a distribution difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


# Gate cumsum is global over T_total (not per segment) — matches the FlyDSL
# harness and the upstream pipeline.
@dataclass(frozen=True)
class Case:
    """One problem shape."""

    name: str
    K: int = 128
    V: int = 128
    Hk: int = 16          # key heads, pre-TP
    Hv: int = 64          # value heads, pre-TP
    tp: int = 8
    BT: int = 64
    gate: str = "g"       # "g" (scalar / GDN) | "gk" (per-channel / KDA)
    context_lens: List[int] = field(default_factory=lambda: [1024])
    dtype: torch.dtype = torch.bfloat16
    ssm_state_dtype: torch.dtype = torch.float32
    with_initial_state: bool = True
    output_final_state: bool = True

    @property
    def Hg(self) -> int:
        return self.Hk // self.tp

    @property
    def H(self) -> int:
        return self.Hv // self.tp

    @property
    def is_varlen(self) -> bool:
        return len(self.context_lens) > 1

    def __repr__(self) -> str:  # pragma: no cover - display only
        T = sum(self.context_lens)
        return (
            f"{self.name}[{self.gate},K{self.K},V{self.V},"
            f"H{self.H},Hg{self.Hg},T{T},n{len(self.context_lens)}]"
        )


# A small, fast, behaviour-covering set. Kept tiny on purpose: the study cares
# about "does it compute the right thing", not about sweep coverage.
CASES: List[Case] = [
    # scalar gate, GQA ratio 4 (Hv/Hk = 64/16), single segment
    Case(name="gdn", gate="g", context_lens=[512]),
    # per-channel gate, GQA ratio 1 — the KDA shape
    Case(name="kda", gate="gk", Hk=96, Hv=96, tp=8, context_lens=[512]),
    # ragged multi-segment
    Case(name="varlen", gate="g", context_lens=[256, 512, 128, 640]),
    # T % BT != 0 on every segment -> tail-chunk row mask must be right
    Case(name="tail", gate="gk", Hk=96, Hv=96, tp=8, context_lens=[160, 96]),
    # bf16 SSM state
    Case(
        name="gdn_state_bf16",
        gate="g",
        context_lens=[512],
        ssm_state_dtype=torch.bfloat16,
    ),
]

CASES_BY_NAME = {c.name: c for c in CASES}


def build_cu_seqlens(context_lens, device="cuda") -> torch.Tensor:
    return torch.tensor(
        [0] + torch.cumsum(torch.tensor(context_lens), 0).tolist(),
        dtype=torch.int32,
        device=device,
    )


def make_inputs(case: Case, *, device: str = "cuda", seed: int = 0) -> dict:
    """Build the K5 input set for *case*.

    Magnitudes and gate construction follow the FlyDSL harness:
    ``k``/``w``/``u`` scaled by 0.1; ``g`` is a non-positive cumsum over T
    (head-major ``[H, T]``); ``gk`` is a non-positive cumsum over T
    (``[T, H, K]``); the initial state is built in f32 and cast down.
    """
    torch.manual_seed(seed)

    H, Hg = case.H, case.Hg
    T = sum(case.context_lens)
    B = 1
    dt, dev = case.dtype, device

    cu = build_cu_seqlens(case.context_lens, device=dev) if case.is_varlen else None
    N = (len(case.context_lens)) if case.is_varlen else B

    k = torch.randn(B, T, Hg, case.K, dtype=dt, device=dev) * 0.1
    w = torch.randn(B, T, H, case.K, dtype=dt, device=dev) * 0.1
    u = torch.randn(B, T, H, case.V, dtype=dt, device=dev) * 0.1

    g = gk = None
    if case.gate == "g":
        # head-major [H, T], non-positive, cumulative along T
        g = (torch.randn(H, T, dtype=torch.float32, device=dev).abs() * -0.5).cumsum(1)
    else:
        # [T, H, K], non-positive, cumulative along T
        gk = (
            torch.randn(T, H, case.K, dtype=torch.float32, device=dev)
            .abs()
            .mul(-0.1)
            .cumsum(dim=0)
            .contiguous()
        )

    h0 = None
    if case.with_initial_state:
        h0 = torch.randn(N, H, case.V, case.K, dtype=torch.float32, device=dev) * 0.01
        if case.ssm_state_dtype != torch.float32:
            h0 = h0.to(case.ssm_state_dtype)

    return {
        "k": k, "w": w, "u": u, "g": g, "gk": gk,
        "initial_state": h0, "cu_seqlens": cu,
        "chunk_size": case.BT, "output_final_state": case.output_final_state,
    }


def reference_for(case: Case, inputs: dict):
    """Run the fp32 reference on *inputs*."""
    from .reference import ref_chunk_gated_delta_rule_fwd_h

    return ref_chunk_gated_delta_rule_fwd_h(
        inputs["k"], inputs["w"], inputs["u"],
        g=inputs["g"], gk=inputs["gk"],
        initial_state=inputs["initial_state"],
        output_final_state=inputs["output_final_state"],
        chunk_size=inputs["chunk_size"],
        cu_seqlens=inputs["cu_seqlens"],
    )


# The FlyDSL harness uses one tolerance for every dtype and output. Kept here
# so the study is not accidentally a *stricter* oracle than the kernel it is
# being compared against.
DEFAULT_ATOL = 5e-2
DEFAULT_RTOL = 5e-2


def assert_k5_matches(
    h_out, vn_out, fs_out,
    h_ref, vn_ref, fs_ref,
    *,
    output_final_state: bool,
    label: str,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    vn_is_head_major: bool = False,
) -> None:
    """Compare a backend's ``(h, v_new, final_state)`` against the reference.

    ``vn_is_head_major`` permutes ``[B, H, T, V]`` back to ``[B, T, H, V]``
    before comparing — the FlyDSL kernel emits head-major ``v_new``.
    """
    torch.testing.assert_close(
        h_out.float(), h_ref.float(), atol=atol, rtol=rtol,
        msg=f"{label}: h mismatch",
    )
    vn = vn_out.permute(0, 2, 1, 3).contiguous() if vn_is_head_major else vn_out
    torch.testing.assert_close(
        vn.float(), vn_ref.float(), atol=atol, rtol=rtol,
        msg=f"{label}: v_new mismatch",
    )
    if output_final_state:
        torch.testing.assert_close(
            fs_out.float(), fs_ref.float(), atol=atol, rtol=rtol,
            msg=f"{label}: final_state mismatch",
        )
    else:
        assert fs_out is None, f"{label}: expected None final_state"
