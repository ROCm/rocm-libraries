# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build the ``SdpaRequest`` a hipDNN graph would carry for a ``cases.py`` case.

Strides come from ``helpers.layout_tensors`` and the bias descriptor from
``helpers._bias_desc``, so a request addresses the same buffers the kernel-facing
tests materialise.
"""

from __future__ import annotations

from dispatch.sdpa import SdpaBias, SdpaRequest, SdpaTensor
from tests.sdpa.helpers import SdpaCase, _bias_desc, layout_tensors
from tests.sdpa.request_cases import packed

__all__ = ["request_from_case"]


def _ragged(case: SdpaCase, name: str, heads: int) -> SdpaTensor:
    """A THD operand as hipDNN describes it: logical BHSD over packed BSHD rows.

    The batch stride is unused for a ragged operand; the offsets are in tokens,
    scaled to elements by the row stride (``ragged_offset_multiplier``).
    """
    s = case.s_q if name in "qo" else case.s_kv
    d = case.d
    return SdpaTensor(
        case.dtype,
        (case.b, heads, s, d),
        (s * heads * d, d, heads * d, 1),
        ragged_offset=packed((case.b + 1, 1, 1, 1), "int32"),
        ragged_offset_multiplier=heads * d,
    )


def request_from_case(case: SdpaCase, arch: str) -> SdpaRequest:
    descs, _ = layout_tensors(case)
    heads = {"q": case.h_q, "k": case.h_k, "v": case.h_v, "o": case.h_q}
    if case.layout == "thd":
        ops = {n: _ragged(case, n, heads[n]) for n in "qkvo"}
    else:
        ops = {
            n: SdpaTensor(case.dtype, descs[n].dims, descs[n].strides) for n in "qkvo"
        }

    left, right, top_left = case.band
    kwargs = dict(
        arch=arch,
        **ops,
        scale_source="value",
        scale=case.resolved_scale,
        diagonal="top_left" if top_left else "bottom_right",
        left_bound=left,
        right_bound=right,
    )
    if case.length_mode == "padded":
        lens = packed((case.b, 1, 1, 1), "int32")
        kwargs.update(padding_mask=True, seq_len_q=lens, seq_len_kv=lens)
    if case.bias is not None:
        desc, _ = _bias_desc(case)
        dtype = case.dtype if case.bias.dtype == "same" else case.bias.dtype
        kwargs["bias"] = SdpaBias(SdpaTensor(dtype, desc.dims, desc.strides))
    if case.emit_lse:
        kwargs["stats"] = packed((case.b, case.h_q, case.s_q, 1), "fp32")
    return SdpaRequest(**kwargs)
