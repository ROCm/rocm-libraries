# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Arch-neutral vocabulary for the GDN *prefill* dispatcher family.

GDN prefill is the chunkwise KDA kernel run in its GDN mode: a gated
delta-rule recurrence whose forget gate is one scalar per (token, head)
(``-exp(A_log) * softplus(a + dt_bias)`` broadcast across the key dimension),
rather than KDA's per-channel vector gate. Everything else -- the six per-chunk
tiles, the triangular solve, the serial state scan -- is the KDA machinery.

Because the in-kernel GDN gate lives only on the split path's raw prep kernel
(the fused kernel is packed-only and cannot emit it), this family ships the two
split halves and no fused candidate. The two halves are selected by an explicit
``algorithm`` pin (``"chunk_prep"`` then ``"chunk_scan"``); there is no single
fused default to fall back on, so ``algorithm="auto"`` is rejected rather than
silently resolved to one half of a two-launch path.

This module holds only what describes a *problem*: the request dataclass, the
family identity, the dimension vocabulary, and the shared request/selector
validation. Arch-specific capability, spec construction and the tuned
``value_splits`` table live in the per-arch module (``prefill_gfx950.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

from rocke.dispatch.core import KernelCandidate, OperatorRequest, selector_matches

from .common import normalize_dtype

FAMILY_PREFILL = "gdn_prefill"

# Bumped when the kernel's argument contract changes, so a cached compile from
# an older layout can never be reused against a newer launcher. The prefill
# family rides the KDA chunkwise kernel's ABI but carries its own GQA-shaped
# request, so it versions independently of both KDA and GDN decode.
GDN_PREFILL_ABI_VERSION = "rocke-gdn-prefill/v1"


@dataclass(frozen=True)
class GdnPrefillRequest(OperatorRequest):
    """One chunkwise GDN prefill problem (many tokens at once).

    ``seqlen`` is the padded per-sequence length; this family has no varlen
    path, so a ragged batch must be padded by the caller to a multiple of the
    chunk size. An omitted ``chunk_size`` selects the gfx950 tuned default (32).

    Head counts are GQA: ``num_v_heads`` value heads are the recurrence streams,
    each reading the key head ``head // kv_group``. ``kv_group = 1`` is MHA.
    The field names mirror :class:`GdnDecodeRequest` so the two GDN operators
    can share one request lineage when the families are consolidated.

    Supported decay range (accepted limit, 2026-09-04): the GDN gate
    ``-exp(A_log)*softplus(a+dt_bias)`` is unbounded, but the chunkwise decay
    stabilization is sized for the KDA ``gate_lower_bound=-5`` reference
    (~160 nats over a 32-token chunk). A head whose per-token decay is steeper
    degrades (one head, silently). Trained GDN keeps ``exp(A_log)*dt`` small, well
    inside this; steeper decay is out of range by design. See the vault "KNOWN
    NUMERICAL LIMIT" note (option (c), nested chunking, is the backlog fix).
    """

    batch: int
    seqlen: int
    arch: str
    num_k_heads: int = 16
    num_v_heads: int = 32
    head_k_dim: int = 128
    head_v_dim: int = 128
    chunk_size: int | None = None
    op: str = "gdn_prefill"
    dtype: str = "bf16"
    algorithm: str = "auto"
    spec_id: str = "auto"
    has_initial_state: bool = False
    store_final_state: bool = True

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = normalize_dtype(self.dtype)
        d["chunk_size"] = self.effective_chunk_size
        return d

    def dims(self) -> dict:
        return {
            "batch": int(self.batch),
            "seqlen": int(self.seqlen),
            "num_k_heads": int(self.num_k_heads),
            "num_v_heads": int(self.num_v_heads),
            "head_k_dim": int(self.head_k_dim),
            "head_v_dim": int(self.head_v_dim),
            "chunk_size": self.effective_chunk_size,
            "num_chunks": self.num_chunks,
        }

    def features(self) -> frozenset:
        active = set()
        if bool(self.has_initial_state):
            active.add("initial_state")
        if bool(self.store_final_state):
            active.add("final_state")
        return frozenset(active)

    @property
    def effective_chunk_size(self) -> int:
        """Requested chunk, or the gfx950 tuned default (32)."""
        if self.chunk_size is not None:
            return int(self.chunk_size)
        return 32

    @property
    def kv_group(self) -> int:
        """Value heads per key head. 1 = MHA; >1 = GQA gather."""
        if self.num_k_heads <= 0:
            return 0
        return int(self.num_v_heads) // int(self.num_k_heads)

    @property
    def num_chunks(self) -> int:
        """Chunks per sequence. Zero when ``seqlen`` does not tile exactly."""
        chunk = self.effective_chunk_size
        if chunk <= 0 or int(self.seqlen) % chunk:
            return 0
        return int(self.seqlen) // chunk

    @property
    def workgroups(self) -> int:
        """Independent recurrence streams: one per (batch, value head).

        The scan's ``value_splits`` fan each of these out into more workgroups;
        that multiplication is the scan grid helper's job, not the request's.
        """
        return int(self.batch) * int(self.num_v_heads)

    @property
    def batch_heads(self) -> int:
        """``BH`` -- the axis the tuned ``value_splits`` table bands on."""
        return self.workgroups


GDN_PREFILL_DIM_VOCABULARY = (
    "batch",
    "seqlen",
    "num_k_heads",
    "num_v_heads",
    "head_k_dim",
    "head_v_dim",
    "chunk_size",
    "num_chunks",
)


def prefill_request_errors(req: OperatorRequest) -> list:
    """Shape-level rejections independent of any candidate or arch."""
    if not isinstance(req, GdnPrefillRequest):
        return [f"expected GdnPrefillRequest, got {type(req).__name__}"]
    errors = []
    if req.batch <= 0:
        errors.append(f"batch must be positive, got {req.batch}")
    if req.seqlen <= 0:
        errors.append(f"seqlen must be positive, got {req.seqlen}")
    if req.num_k_heads <= 0 or req.num_v_heads <= 0:
        errors.append("head counts must be positive")
    elif req.num_v_heads % req.num_k_heads:
        errors.append(
            f"num_v_heads {req.num_v_heads} must be a multiple of "
            f"num_k_heads {req.num_k_heads}"
        )
    if req.head_k_dim <= 0 or req.head_v_dim <= 0:
        errors.append("head dims must be positive")
    # GDN prefill rides the bf16-only KDA chunkwise kernel.
    if normalize_dtype(req.dtype) != "bf16":
        errors.append(f"unsupported dtype {req.dtype!r} (bf16 only)")
    if req.seqlen > 0 and req.num_chunks == 0:
        errors.append(
            f"seqlen {req.seqlen} must be a multiple of chunk "
            f"{req.effective_chunk_size}; this family has no varlen path"
        )
    return errors


# Shared pin-selector, re-exported under this family's name (see dispatch core).
prefill_selector_matches = selector_matches
