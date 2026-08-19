# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""FP32 reference for the gated-delta-rule chunked state scan (the "K5" stage).

Ported from the FlyDSL kernel's test harness so the rocKE port can be validated
against the *same* oracle rather than a re-derivation. Two gate variants:

* **Scalar gate** (``g`` given) — the GDN family::

      b_v[t] = u[t] - w[t] @ h^T
      gate[t] = exp(g_last - g_cumsum[t])          # scalar, broadcast over V
      h       = h * exp(g_last) + (b_v * gate)^T @ k[t]

* **Per-channel gate** (``gk`` given) — the KDA family::

      b_v[t]  = u[t] - w[t] @ h^T                  # same delta correction
      h[:, j] *= exp(gk_last[j])                   # per-K decay, no v_new gating
      h       = h + b_v^T @ k[t]

  (``k`` is pre-gated upstream; ``v_new`` is not gated on this path.)

``w`` and ``u`` are token-major ``[B, T, H, *]``.

Deliberately dependency-light: numpy-free, triton-free, torch-only, so it runs
anywhere the numeric lane runs. ``torch`` is imported lazily by the caller.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def ref_chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Return ``(h_out, v_new, final_state)`` in fp32.

    ``h_out[b, chunk, head]`` is the state snapshot *before* that chunk's
    update — which is what the kernel drains to HBM, and what the fused K6
    stage consumes.
    """
    assert (g is None) != (gk is None), "exactly one of g, gk must be provided"

    B, T, Hg_dim, K_dim = k.shape
    H_dim, V_dim = u.shape[-2], u.shape[-1]
    BT = chunk_size

    if cu_seqlens is None:
        NT = _cdiv(T, BT)
    else:
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        NT = sum(_cdiv(int(s), BT) for s in seq_lens)

    gqa_ratio = H_dim // Hg_dim

    h_out = k.new_zeros(B, NT, H_dim, V_dim, K_dim, dtype=torch.float32)
    v_new_out = torch.zeros_like(u, dtype=torch.float32)

    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    final_state = (
        torch.zeros(N, H_dim, V_dim, K_dim, dtype=torch.float32, device=k.device)
        if output_final_state
        else None
    )

    for b_idx in range(B):
        if cu_seqlens is not None:
            seqs = [
                (s, int(cu_seqlens[s].item()), int(cu_seqlens[s + 1].item()))
                for s in range(N)
            ]
        else:
            seqs = [(b_idx, 0, T)]

        chunk_offset = 0
        for seq_idx, bos, eos in seqs:
            seq_len = eos - bos
            seq_nt = _cdiv(seq_len, BT)

            for i_h in range(H_dim):
                i_hg = i_h // gqa_ratio
                h_state = torch.zeros(
                    V_dim, K_dim, dtype=torch.float32, device=k.device
                )
                if initial_state is not None:
                    h_state = initial_state[seq_idx, i_h].float().clone()

                for i_t in range(seq_nt):
                    t_start = i_t * BT
                    t_end = min(t_start + BT, seq_len)
                    actual_bt = t_end - t_start

                    # snapshot BEFORE the update
                    h_out[b_idx, chunk_offset + i_t, i_h] = h_state.clone()

                    w_chunk = w[b_idx, bos + t_start : bos + t_end, i_h].float()
                    u_chunk = u[b_idx, bos + t_start : bos + t_end, i_h].float()
                    b_v = u_chunk - w_chunk @ h_state.T
                    v_new_out[b_idx, bos + t_start : bos + t_end, i_h] = b_v

                    k_chunk = k[b_idx, bos + t_start : bos + t_end, i_hg].float()
                    last_idx = bos + t_end - 1

                    if gk is not None:
                        # per-channel decay on h; v_new ungated
                        gk_last = gk[last_idx, i_h].float()  # [K]
                        h_state = h_state * torch.exp(gk_last).unsqueeze(0)
                        h_state = h_state + b_v.T @ k_chunk
                    else:
                        # scalar decay on h + per-token v_new gate
                        g_last = g[i_h, last_idx].float()
                        g_chunk = g[i_h, bos + t_start : bos + t_end].float()

                        mask = torch.zeros(BT, device=k.device)
                        mask[:actual_bt] = 1.0
                        gate = torch.where(
                            mask[:actual_bt].bool(),
                            torch.exp(g_last - g_chunk),
                            torch.zeros_like(g_chunk),
                        )
                        b_v_gated = b_v * gate.unsqueeze(-1)
                        h_state = h_state * torch.exp(g_last)
                        # round-trip through the storage dtype: the kernel writes
                        # bf16 into LDS before GEMM2, so the reference must too or
                        # it is a stricter oracle than the kernel can satisfy.
                        b_v_gated_cast = b_v_gated.to(k.dtype).float()
                        h_state = h_state + b_v_gated_cast.T @ k_chunk

                if output_final_state:
                    final_state[seq_idx, i_h] = h_state

            chunk_offset += seq_nt

    return h_out, v_new_out.to(u.dtype), final_state
