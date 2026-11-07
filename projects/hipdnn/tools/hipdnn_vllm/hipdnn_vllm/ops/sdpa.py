# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Python entry points for ``torch.ops.hipdnn_vllm.sdpa_fwd``.

``sdpa_fwd`` writes ``out`` in place and raises on a decline (decoded by
:func:`hipdnn_vllm.declines.parse_op_error`). :func:`sdpa_fwd_probe` builds and
caches the plan for a signature without executing it, and reports the engine
that would serve it; :func:`sdpa_fwd_layout` returns the graph the builder
describes (tensor uids, dims, strides, ragged multipliers, slot order).

Layouts (see ``csrc/graphs/sdpa.cpp``):

* dense: ``q``/``out`` ``[B, Hq, Sq, D]``, ``k``/``v`` ``[B, Hkv, Skv, D]``, any strides;
* ragged Q/O: packed ``[T, Hq, D]`` plus ``ragged_offset_q`` ``[B+1]`` (token
  units), ``seq_len_q`` ``[B]`` and ``max_seq_len_q``;
* ragged K/V: packed ``[Tk, Hkv, D]`` plus ``ragged_offset_kv``, ``seq_len_kv``
  and ``max_seq_len_kv``;
* paged K/V: containers ``[num_blocks, Hkv, block_size, D]`` plus ``page_table``
  ``[B, M]``, ``seq_len_kv`` ``[B]`` and ``max_seq_len_kv``;
* attention sink (any layout): ``sink`` ``[1, Hq, 1, 1]`` in FLOAT or the Q
  dtype, one learned logit per query head that joins the softmax max and
  denominator (not multiplied by ``scale``).

Masks use hipDNN's diagonal band: ``causal`` sets ``right_bound = 0``,
``bottom_right`` picks the alignment, and ``left_bound``/``right_bound`` are
vLLM's ``sliding_window`` pair as is (``-1`` = open).
"""

from __future__ import annotations

from typing import Any

import torch

from .. import _bootstrap

__all__ = ["SDPA_DEFAULTS", "sdpa_fwd", "sdpa_fwd_layout", "sdpa_fwd_probe"]

SDPA_DEFAULTS: dict[str, Any] = {
    "causal": False,
    "bottom_right": True,
    "left_bound": -1,
    "right_bound": -1,
    "ragged_offset_q": None,
    "seq_len_q": None,
    "max_seq_len_q": 0,
    "ragged_offset_kv": None,
    "page_table": None,
    "seq_len_kv": None,
    "max_seq_len_kv": 0,
    "sink": None,
}


def _kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    unknown = set(kwargs) - set(SDPA_DEFAULTS)
    if unknown:
        raise TypeError(f"sdpa_fwd: unexpected argument(s) {sorted(unknown)}")
    return {**SDPA_DEFAULTS, **kwargs}


def sdpa_fwd(q, k, v, out, scale: float, **kwargs) -> torch.Tensor:
    _bootstrap.load()
    torch.ops.hipdnn_vllm.sdpa_fwd(q, k, v, out, float(scale), **_kwargs(kwargs))
    return out


def sdpa_fwd_probe(q, k, v, out, scale: float, **kwargs) -> tuple[str, str]:
    """``(reason, detail)``: reason ``""`` and the engine name when hipDNN runs
    this signature; otherwise the decline reason and the backend message."""
    _bootstrap.load()
    from .. import _C

    return tuple(_C.sdpa_fwd_probe(q, k, v, out, float(scale), **_kwargs(kwargs)))


def sdpa_fwd_layout(q, k, v, out, scale: float, **kwargs) -> dict[str, Any]:
    _bootstrap.load()
    from .. import _C

    return _C.sdpa_fwd_layout(q, k, v, out, float(scale), **_kwargs(kwargs))
