# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side case builder for the shipped tuned general keys of the backward.

:func:`general_key_case` turns a tuned general key into a fixed-length bhsd /
bshd numeric case whose request lands in that key. The device suite
(``test_attention_bwd_dispatch_numeric.py``) runs the cases; the host test
``test_attention_bwd_key_cases.py`` checks that every shipped fixed-length
key of every arch gets one. No device and no torch needed here.
"""

from __future__ import annotations

import dataclasses

from dispatch.attention_bwd import tuned_tables
from kernels.common.attention_bwd_plan import attn_bwd_support, attn_bwd_tuning_key

from .sdpa.bwd_cases import _mkb
from .sdpa.bwd_kernel_cases import request_from_case

# float64 oracle budget of the device suite: batches * heads * s_q * s_kv
ORACLE_CELLS = 160 * 1024 * 1024

_MASK_CASE = {
    "none": {},
    "causal_tl": {"diagonal": "top_left", "right_bound": 0},
    "causal_br": {"diagonal": "bottom_right", "right_bound": 0},
}


def mask_case(mask, req):
    """The case mask fields of a ``mask`` request (a window takes the
    request's own band), or None for a mask kind with no case form."""
    if mask in _MASK_CASE:
        return _MASK_CASE[mask]
    if mask == "window" and req.left_bound is not None and req.left_bound >= 0:
        return {
            "diagonal": "bottom_right" if req.bottom_right else "top_left",
            "left_bound": req.left_bound,
            "right_bound": req.right_bound,
        }
    return None


def general_key_case(arch, key, *, oracle_cells=ORACLE_CELLS):
    """A fixed-length bhsd / bshd case of tuned key ``key`` and the request
    fields that put it in the key (``num_cus``, ``num_xcds``), or None."""
    for req in tuned_tables.general_key_requests(arch, key):
        if req.layout != "dense" or req.padding or req.num_cus is None:
            continue
        mask_kw = mask_case(key[4], req)
        if mask_kw is None or req.b * req.h_q * req.s_q * req.s_kv > oracle_cells:
            continue
        layout = "bhsd" if req.strides is None else "bshd"
        tag = "_".join(str(x) for x in key[2:]).replace("*", "any")
        case = _mkb(
            f"disp_{arch}_{tag}_b{req.b}h{req.h_q}k{req.h_k}s{req.s_q}x{req.s_kv}",
            reqs=(1,),
            tier="full",
            dtype=req.dtype,
            d=req.d_qk,
            b=req.b,
            h_q=req.h_q,
            h_k=req.h_k,
            h_v=req.h_v,
            s_q=req.s_q,
            s_kv=req.s_kv,
            layout=layout,
            **mask_kw,
        )
        extra = {"num_cus": req.num_cus, "num_xcds": req.num_xcds}
        got = dataclasses.replace(request_from_case(case), **extra)
        if attn_bwd_tuning_key(got, arch) == key and attn_bwd_support(got, arch).ok:
            return case, extra
    return None
