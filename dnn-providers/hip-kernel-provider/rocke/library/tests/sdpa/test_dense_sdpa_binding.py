# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The dense SDPA binding's data movement, on CPU.

The kernel is replaced by a stand-in that reads only what the binding hands
it -- contiguous ``[B, S, H, D]`` Q/K/V -- and writes O (and LSE) through the
same tensors, computing attention with ``sdpa_reference`` using the request's
band and scale. Comparing the written-back O and LSE with the case-table
oracle then checks the layout copies and write-back for every layout. The
kernels themselves are checked on GPU by ``test_dense_sdpa_numeric.py``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rocke.numeric.sdpa_reference import sdpa_reference

from dispatch.sdpa.bind_dense import _DensePlan, sdpa_dense_admits
from tests.sdpa import cases as C
from tests.sdpa import helpers as H
from tests.sdpa.binding_harness import bind_tensors, outputs
from tests.sdpa.request_bridge import request_from_case
from tests.sdpa.request_cases import bhsd, make_request, packed

ARCH = "gfx950"


def fake_kernel(plan, req):
    """Attention over exactly the tensors the binding prepared."""
    kt = plan.kernel_tensors()
    b, s_q = kt["q"].shape[:2]
    s_kv = kt["k"].shape[1]

    def flat(t):
        return t.reshape(-1, *t.shape[2:]).double().numpy()

    o, lse = sdpa_reference(
        flat(kt["q"]),
        flat(kt["k"]),
        flat(kt["v"]),
        scale=float(req.scale),
        left_bound=req.left_bound,
        right_bound=req.right_bound,
        top_left=req.diagonal == "top_left",
        q_offsets=np.arange(b + 1) * s_q,
        kv_offsets=np.arange(b + 1) * s_kv,
    )
    kt["out"].copy_(torch.from_numpy(o).view(kt["out"].shape).to(kt["out"].dtype))
    if "lse" in kt:
        # The reference's LSE is (B * S_q, H); the kernel's is (B, H, S_q, 1).
        lse = torch.from_numpy(lse).view(b, s_q, -1).transpose(1, 2)
        kt["lse"].copy_(lse[..., None].float())


_CASES = [
    c
    for c in C.all_cases()
    if ARCH in c.archs
    and ("smoke" in c.tags or "full" in c.tags)
    and c.cost <= 2e7
    and sdpa_dense_admits(request_from_case(c, ARCH))[0]
]


def test_the_cpu_table_covers_the_layouts():
    assert {c.layout for c in _CASES} >= {
        "bhsd",
        "bshd",
        "strided_bshd",
        "strided_bhsd",
        "mixed",
    }
    assert any(c.emit_lse for c in _CASES)


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_binding_moves_data_faithfully(case):
    req = request_from_case(case, ARCH)
    inputs = H.materialize(case)
    tensors = bind_tensors(case, H.to_torch(inputs, device="cpu"), req)

    plan = _DensePlan(req, tensors)
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
def test_contiguous_bshd_operands_are_used_in_place(layout, inplace):
    req = make_request(d=64, **({"layout": layout} if layout else {}))
    tensors = {n: _dense(req, n) for n in ("q", "k", "v", "o")}
    plan = _DensePlan(req, tensors)
    assert (not plan.copied) is inplace
    if inplace:
        assert plan.views["q"].data_ptr() == tensors["q"].data_ptr()
        assert plan.views["o"].data_ptr() == tensors["o"].data_ptr()


def test_non_contiguous_stats_are_written_back():
    req = replace(make_request(d=64), stats=packed((2, 4, 8, 1), "fp32"))
    tensors = {n: _dense(req, n) for n in ("q", "k", "v", "o")}
    base = torch.full((2, 4, 8, 2), 7.0)
    tensors["stats"] = base[..., :1]  # the request's shape, but not contiguous
    req = replace(req, stats=replace(req.stats, strides=tuple(base[..., :1].stride())))
    plan = _DensePlan(req, tensors)
    assert plan.lse is not plan.stats
    plan.pack()
    fake_kernel(plan, req)
    plan.unpack()
    assert torch.all(torch.isfinite(base[..., 0]))
    assert torch.all(base[..., 1] == 7.0)


def test_a_tensor_that_disagrees_with_the_request_is_refused():
    req = make_request(d=64)  # BSHD
    tensors = {n: _dense(req, n) for n in ("q", "k", "v", "o")}
    tensors["q"] = torch.empty_strided(
        req.q.dims, bhsd(2, 4, 8, 64).strides, dtype=torch.float16
    )
    with pytest.raises(ValueError, match="strides"):
        _DensePlan(req, tensors)
