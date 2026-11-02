# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The unified SDPA binding's data movement, on CPU.

The kernel is replaced by a stand-in that reads only what the binding hands
it -- packed Q, the paged K/V cache through its block table, ``cu_seqlens_q``
and ``seqused_k`` -- and computes attention with ``sdpa_reference`` using the
request's band and scale. Comparing the scattered-back O and LSE with the
case-table oracle then checks packing, paging, lengths and write-back for
every layout and length mode. The kernels themselves are checked on GPU by
``test_unified_sdpa_numeric.py``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rocke.numeric.sdpa_reference import sdpa_reference

from dispatch.sdpa.bind_unified import _LaunchPlan, sdpa_unified_admits
from tests.sdpa import cases as C
from tests.sdpa import helpers as H
from tests.sdpa.binding_harness import bind_tensors, outputs
from tests.sdpa.request_bridge import request_from_case
from tests.sdpa.request_cases import bhsd, make_request, packed

ARCH = "gfx950"


def fake_kernel(plan, req):
    """Attention over exactly the tensors the binding prepared."""
    kt = plan.kernel_tensors()
    used = kt["seqused_k"].tolist()
    table = kt["block_table"]
    k_pages, v_pages = [], []
    for b, n in enumerate(used):
        k_pages.append(kt["k"][table[b].long()].reshape(-1, *kt["k"].shape[2:])[:n])
        v_pages.append(kt["v"][table[b].long()].reshape(-1, *kt["v"].shape[2:])[:n])
    o, lse = sdpa_reference(
        kt["q"].double().numpy(),
        torch.cat(k_pages).double().numpy(),
        torch.cat(v_pages).double().numpy(),
        scale=float(req.scale),
        left_bound=req.left_bound,
        right_bound=req.right_bound,
        top_left=req.diagonal == "top_left",
        q_offsets=np.asarray(kt["cu_seqlens_q"].tolist()),
        kv_offsets=np.concatenate([[0], np.cumsum(used)]),
    )
    kt["out"].copy_(torch.from_numpy(o).to(kt["out"].dtype))
    if plan.lse is not None:
        plan.lse.copy_(torch.from_numpy(lse)[..., None].float())


_CASES = [
    c
    for c in C.all_cases()
    if ARCH in c.archs
    and ("smoke" in c.tags or "full" in c.tags)
    and c.cost <= 5e6
    and sdpa_unified_admits(request_from_case(c, ARCH))[0]
]


def test_the_cpu_table_covers_every_layout_and_length_mode():
    assert {c.length_mode for c in _CASES} == set(H.LENGTH_MODES)
    assert {c.layout for c in _CASES} >= {"bhsd", "bshd", "thd"}
    assert any(c.emit_lse for c in _CASES)


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_binding_moves_data_faithfully(case):
    req = request_from_case(case, ARCH)
    inputs = H.materialize(case)
    tensors = bind_tensors(case, H.to_torch(inputs, device="cpu"), req)

    plan = _LaunchPlan(req, tensors)
    plan.pack()
    fake_kernel(plan, req)
    plan.unpack()

    o, lse = outputs(case, tensors, req)
    assert H.compare(case, o, lse, H.evaluate(inputs)) == []


def _dense(req, name):
    desc = getattr(req, name)
    buf = torch.randn(1 + sum((n - 1) * s for n, s in zip(desc.dims, desc.strides)))
    return torch.as_strided(buf.half(), desc.dims, desc.strides)


@pytest.mark.parametrize("layout,inplace", [(None, True), (bhsd, False)])
def test_packed_bshd_q_and_o_are_used_in_place(layout, inplace):
    req = make_request(d=64, **({"layout": layout} if layout else {}))
    tensors = {n: _dense(req, n) for n in ("q", "k", "v", "o")}
    plan = _LaunchPlan(req, tensors)
    assert plan.q_inplace is inplace and plan.o_inplace is inplace
    if inplace:
        assert plan.q_packed.data_ptr() == tensors["q"].data_ptr()


def test_padded_rows_are_zero_and_neg_inf():
    lens = torch.tensor([1, 0], dtype=torch.int32).view(2, 1, 1, 1)
    req = replace(
        make_request(d=64, s_q=1, s_kv=16),
        padding_mask=True,
        seq_len_q=packed((2, 1, 1, 1), "int32"),
        seq_len_kv=packed((2, 1, 1, 1), "int32"),
        stats=packed((2, 4, 1, 1), "fp32"),
    )
    tensors = {n: _dense(req, n) for n in ("q", "k", "v", "o")}
    tensors["o"].fill_(7)
    tensors.update(
        seq_len_q=lens,
        seq_len_kv=torch.tensor([16, 16], dtype=torch.int32).view(2, 1, 1, 1),
        stats=torch.full((2, 4, 1, 1), 7.0),
    )
    plan = _LaunchPlan(req, tensors)
    assert plan.cu == [0, 1, 1]
    plan.pack()
    fake_kernel(plan, req)
    plan.unpack()
    assert torch.all(tensors["o"][1] == 0)
    assert torch.all(torch.isneginf(tensors["stats"][1]))
    assert torch.all(torch.isfinite(tensors["stats"][0]))


def test_a_tensor_that_disagrees_with_the_request_is_refused():
    req = make_request(d=64)  # BSHD
    tensors = {n: _dense(req, n) for n in ("q", "k", "v", "o")}
    tensors["q"] = torch.empty_strided(
        req.q.dims, bhsd(2, 4, 8, 64).strides, dtype=torch.float16
    )
    with pytest.raises(ValueError, match="strides"):
        _LaunchPlan(req, tensors)
