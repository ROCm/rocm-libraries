# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Turns the caller's torch tensors into the tensors a hipDNN graph declares.

A :class:`Problem` holds, for one forward or backward call:

* ``slots``: role -> declared dims, strides, dtype and ragged-offset role. The
  roles, in insertion order, fix the tensor UIDs; ``key`` (everything that shapes
  the graph) is the plan-cache key.
* ``tensors``: role -> the torch tensor whose pointer feeds that slot at execute
  time, including the ``[B,1,1,1]`` sequence-length views and the ragged-offset
  tensors built here.

Nothing in this module touches hipDNN or a device stream.

Packed varlen (``pos_*`` set) arrives as ``[1, H, T, D]`` views. The RFC-0014
span rule (``projects/hipdnn/docs/rfcs/0014_RaggedTensors.md`` item 5;
``data_sdk/.../RaggedTensor.hpp`` ``validateSequenceOutermostLayout``) requires one
token row's footprint to fit inside the token stride. A tensor that fails it
raises :class:`EngineDeclined` before any graph exists. One that passes is declared
``[B, H, max_S, D]`` with its own head, token and feature strides, a batch stride
of ``max_S * token_stride``, and a ``[B+1, 1, 1, 1]`` INT32 ragged offset of
``pos * token_stride`` elements.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._types import Dropout, EngineDeclined, Mask, Varlen

_FWD_OUTPUTS = ("o", "stats")
_BWD_OUTPUTS = ("dq", "dk", "dv", "dbias")


@dataclass(frozen=True)
class Slot:
    dims: tuple
    strides: tuple
    dtype: torch.dtype
    ragged: str | None  # role of the ragged-offset slot, or None


class Problem:
    def __init__(
        self, direction: str, scale: float, mask: Mask | None, dropout_p: float | None
    ):
        self.direction = direction
        self.scale = float(scale)
        self.mask = mask
        self.dropout_p = dropout_p
        self.padding = False
        self.packed_q = False
        self.packed_kv = False
        self.slots: dict[str, Slot] = {}
        self.tensors: dict[str, torch.Tensor] = {}
        self._ragged: dict[tuple, str] = {}

    @property
    def outputs(self) -> tuple:
        names = _FWD_OUTPUTS if self.direction == "fwd" else _BWD_OUTPUTS
        return tuple(r for r in names if r in self.slots)

    @property
    def key(self) -> tuple:
        mask = (
            None
            if self.mask is None
            else (self.mask.alignment, self.mask.left, self.mask.right)
        )
        slots = tuple(
            (role, s.dims, s.strides, s.dtype, s.ragged)
            for role, s in self.slots.items()
        )
        return (
            self.direction,
            self.scale,
            mask,
            self.dropout_p,
            self.padding,
            self.packed_q,
            self.packed_kv,
            slots,
        )

    def uids(self) -> dict[str, int]:
        return {role: i + 1 for i, role in enumerate(self.slots)}

    def _add(
        self, role: str, t: torch.Tensor, dims=None, strides=None, ragged=None
    ) -> None:
        self.slots[role] = Slot(
            tuple(t.shape) if dims is None else dims,
            tuple(t.stride()) if strides is None else strides,
            t.dtype,
            ragged,
        )
        self.tensors[role] = t

    def add_dense(self, role: str, t: torch.Tensor) -> None:
        self._add(role, t)

    def add_side(
        self, role: str, t: torch.Tensor, side: str, varlen: Varlen | None
    ) -> None:
        """Add a Q-side (``side='q'``) or K-side (``'kv'``) tensor, packed when its ``pos`` is set."""
        pos = None
        if varlen is not None:
            pos = varlen.pos_q if side == "q" else varlen.pos_kv
        if pos is None:
            self._add(role, t)
            return
        max_s = varlen.max_sq if side == "q" else varlen.max_sk
        if t.dim() != 4 or t.shape[0] != 1:
            raise ValueError(
                f"packed {role} must be a [1, H, T, D] view, got {tuple(t.shape)}"
            )
        _, heads, _, feat = t.shape
        _, s_head, s_tok, s_feat = t.stride()
        if not _sequence_outermost(heads, feat, s_head, s_feat, s_tok):
            raise EngineDeclined(
                self.direction,
                f"adapter: packed layout not sequence-outermost (RFC-0014): {role}",
            )
        batch = pos.numel() - 1
        ragged = self._ragged_offset(side, pos, s_tok)
        self._add(
            role,
            t,
            dims=(batch, heads, max_s, feat),
            strides=(max_s * s_tok, s_head, s_tok, s_feat),
            ragged=ragged,
        )

    def _ragged_offset(self, side: str, pos: torch.Tensor, token_stride: int) -> str:
        key = (side, token_stride)
        role = self._ragged.get(key)
        if role is None:
            role = f"ragged_offset_{side}_{token_stride}"
            self._ragged[key] = role
            offsets = (pos.to(torch.int64) * token_stride).to(torch.int32)
            self._add(role, offsets.view(-1, 1, 1, 1))
        return role


def _sequence_outermost(
    heads: int, feat: int, s_head: int, s_feat: int, s_tok: int
) -> bool:
    """RFC-0014 span rule over the non-batch, non-sequence axes of a ``[1, H, T, D]`` view."""
    if s_tok <= 0:
        return False
    if heads == 0 or feat == 0:
        return True  # no elements to place
    span = 1 + (heads - 1) * s_head + (feat - 1) * s_feat
    return span <= s_tok


def _add_common(
    prob: Problem,
    bias: torch.Tensor | None,
    dropout: Dropout | None,
    varlen: Varlen | None,
) -> None:
    if bias is not None:
        if varlen is not None and (
            varlen.pos_q is not None or varlen.pos_kv is not None
        ):
            raise ValueError("bias is not expressible with packed varlen")
        prob.add_dense("bias", bias)
    if varlen is not None:
        prob.padding = True
        prob.packed_q = varlen.pos_q is not None
        prob.packed_kv = varlen.pos_kv is not None
        batch = varlen.seq_len_q.numel()
        prob.add_dense("seq_len_q", varlen.seq_len_q.view(batch, 1, 1, 1))
        prob.add_dense(
            "seq_len_kv", varlen.seq_len_kv.view(varlen.seq_len_kv.numel(), 1, 1, 1)
        )
    if prob.dropout_p is not None:
        prob.add_dense("seed", dropout.seed)
        prob.add_dense("offset", dropout.offset)


def _dropout_p(dropout: Dropout | None) -> float | None:
    return float(dropout.p) if dropout is not None and dropout.p != 0 else None


def forward_problem(q, k, v, o, stats, scale, bias, mask, dropout, varlen) -> Problem:
    prob = Problem("fwd", scale, mask, _dropout_p(dropout))
    prob.add_side("q", q, "q", varlen)
    prob.add_side("k", k, "kv", varlen)
    prob.add_side("v", v, "kv", varlen)
    prob.add_side("o", o, "q", varlen)
    if stats is not None:
        prob.add_side("stats", stats, "q", varlen)
    _add_common(prob, bias, dropout, varlen)
    return prob


def backward_problem(
    q, k, v, o, do, stats, dq, dk, dv, scale, dbias, bias, mask, dropout, varlen
):
    prob = Problem("bwd", scale, mask, _dropout_p(dropout))
    prob.add_side("q", q, "q", varlen)
    prob.add_side("k", k, "kv", varlen)
    prob.add_side("v", v, "kv", varlen)
    prob.add_side("o", o, "q", varlen)
    prob.add_side("do", do, "q", varlen)
    prob.add_side("stats", stats, "q", varlen)
    prob.add_side("dq", dq, "q", varlen)
    prob.add_side("dk", dk, "kv", varlen)
    prob.add_side("dv", dv, "kv", varlen)
    _add_common(prob, bias, dropout, varlen)
    if dbias is not None:
        if bias is None:
            raise ValueError("dbias requires bias")
        prob.add_dense("dbias", dbias)
    return prob
