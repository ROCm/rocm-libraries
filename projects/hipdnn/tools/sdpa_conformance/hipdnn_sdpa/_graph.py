# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Builds the hipDNN SDPA graph for a :class:`~._problem.Problem`.

Needs the ``hipdnn_frontend`` bindings but no handle and no device: every tensor is
declared with ``Graph.tensor_like`` from the caller's torch tensor, then re-declared
with the slot's dims and strides (they differ only for packed varlen). The graph can
be serialized with ``Graph.to_json()`` straight after :func:`build`.

Bindings used (``projects/hipdnn/python/frontend_bindings/src``):

* ``Graph.tensor_like`` ``graph_bindings.cpp:771``; ``Graph.sdpa`` ``:701``;
  ``Graph.sdpa_backward`` ``:711``; ``Graph.set_{compute,intermediate,io}_data_type``
  ``:603-609``
* ``Tensor`` ``set_{uid,name,data_type,stride,dim,output,is_pass_by_value}`` and
  ``get_{name,data_type,stride,dim}`` ``tensor_bindings.cpp:131-154``;
  ``Tensor.set_ragged_offset`` is Stream A's binding (``set_ragged_offset(tensor | None)``)
* ``SdpaAttributes`` ``attributes_bindings.cpp:1039-1186`` (``set_bias`` ``:1056``,
  ``set_seq_len_{q,kv}`` ``:1064,1067``, ``set_generate_stats`` ``:1146``,
  ``set_padding_mask`` ``:1152``, ``set_dropout`` ``:1161``, ``set_attn_scale(float)``
  ``:1169``, ``set_diagonal_band_{left,right}_bound`` ``:1173,1176``,
  ``set_diagonal_alignment`` ``:1182``)
* ``SdpaBackwardAttributes`` ``attributes_bindings.cpp:1238-1353`` (same setter names:
  ``set_bias`` ``:1275``, ``set_seq_len_{q,kv}`` ``:1279,1283``, ``set_dbias`` ``:1319``,
  ``set_padding_mask`` ``:1326``, ``set_dropout`` ``:1335`` -- the backward attribute
  has its own ``set_dropout(p, seed, offset)``, ``set_attn_scale(float)`` ``:1340``,
  bounds ``:1344,1347``, ``set_diagonal_alignment`` ``:1350``)
* ``DiagonalAlignment`` ``types_bindings.cpp:123``
"""

from __future__ import annotations

import torch

from ._problem import Problem


def frontend():
    """The hipdnn_frontend module (imported lazily so the package imports without it)."""
    import hipdnn_frontend

    return hipdnn_frontend


def _declare(hipdnn, prob: Problem, role: str, uid: int, declared: dict):
    """The graph tensor for ``role``: torch metadata, slot dims/strides, uid, ragged offset."""
    slot = prob.slots[role]
    t = hipdnn.Graph.tensor_like(prob.tensors[role].detach(), role)
    # A CPU producer would otherwise become a runtime pass-by-value tensor.
    t.set_is_pass_by_value(False)
    t.set_dim(list(slot.dims)).set_stride(list(slot.strides)).set_uid(uid)
    if slot.ragged is not None:
        t.set_ragged_offset(declared[slot.ragged])
    declared[role] = t
    return t


def _configure_output(out, meta, ragged) -> None:
    """Give a frontend-created output tensor its slot's metadata and ragged offset."""
    out.set_name(meta.get_name())
    out.set_dim(meta.get_dim()).set_stride(meta.get_stride())
    out.set_data_type(meta.get_data_type())
    out.set_uid(meta.get_uid())
    out.set_output(True)
    if ragged is not None:
        out.set_ragged_offset(ragged)


def _apply_common(hipdnn, attrs, prob: Problem, t) -> None:
    """Attributes shared by the forward and backward node (same binding names on both)."""
    attrs.set_attn_scale(prob.scale)
    mask = prob.mask
    if mask is not None:
        attrs.set_diagonal_alignment(
            hipdnn.DiagonalAlignment.TOP_LEFT
            if mask.alignment == "tl"
            else hipdnn.DiagonalAlignment.BOTTOM_RIGHT
        )
        if mask.left is not None:
            attrs.set_diagonal_band_left_bound(int(mask.left))
        if mask.right is not None:
            attrs.set_diagonal_band_right_bound(int(mask.right))
    if "bias" in t:
        attrs.set_bias(t["bias"])
    if prob.padding:
        attrs.set_padding_mask(True)
        attrs.set_seq_len_q(t["seq_len_q"])
        attrs.set_seq_len_kv(t["seq_len_kv"])
    if prob.dropout_p is not None:
        attrs.set_dropout(prob.dropout_p, t["seed"], t["offset"])


def build(prob: Problem):
    """Graph for ``prob``, not yet built against a handle. Returns ``(graph, uids)``.

    ``uids`` maps every slot role to its tensor UID, for the variant pack.
    """
    hipdnn = frontend()
    uids = prob.uids()
    t: dict = {}
    for role in prob.slots:
        if role in prob.outputs:
            continue
        _declare(hipdnn, prob, role, uids[role], t)

    g = hipdnn.Graph()
    g.set_io_data_type(t["q"].get_data_type())
    g.set_compute_data_type(hipdnn.DataType.FLOAT)
    g.set_intermediate_data_type(hipdnn.DataType.FLOAT)

    if prob.direction == "fwd":
        attrs = hipdnn.SdpaAttributes()
        _apply_common(hipdnn, attrs, prob, t)
        has_stats = "stats" in prob.slots
        attrs.set_generate_stats(has_stats)
        o, stats = g.sdpa(t["q"], t["k"], t["v"], attrs)
        meta = {r: _declare(hipdnn, prob, r, uids[r], t) for r in prob.outputs}
        _configure_output(o, meta["o"], t.get(prob.slots["o"].ragged))
        if has_stats:
            _configure_output(stats, meta["stats"], t.get(prob.slots["stats"].ragged))
    else:
        attrs = hipdnn.SdpaBackwardAttributes()
        _apply_common(hipdnn, attrs, prob, t)
        meta = {r: _declare(hipdnn, prob, r, uids[r], t) for r in prob.outputs}
        if "dbias" in meta:
            meta["dbias"].set_output(True)
            attrs.set_dbias(meta["dbias"])
        dq, dk, dv = g.sdpa_backward(
            t["q"], t["k"], t["v"], t["o"], t["do"], t["stats"], attrs
        )
        for out, role in ((dq, "dq"), (dk, "dk"), (dv, "dv")):
            _configure_output(out, meta[role], t.get(prob.slots[role].ragged))
    return g, uids


__all__ = ["build", "frontend"]
