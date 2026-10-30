# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The dense SDPA binding on GPU, against the case-table oracle.

Every ``cases.py`` case the dense binding admits on the live arch, all tiers,
is bound with ``sdpa_dense_bind``, launched twice (the second launch reuses
the bound buffers) and compared with ``sdpa_reference``, fully masked rows
included (O = 0, LSE = -inf).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dispatch.sdpa.bind_dense import sdpa_dense_admits, sdpa_dense_bind
from tests.sdpa import cases as C
from tests.sdpa import helpers as H
from tests.sdpa.binding_harness import bind_tensors, outputs
from tests.sdpa.request_bridge import request_from_case


def _arch():
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_properties(0).gcnArchName.lower()
    return next((a for a in ("gfx942", "gfx950") if a in name), None)


ARCH = _arch()

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH is None, reason="needs a gfx942 or gfx950 GPU with ROCm torch"
    ),
]

_CASES = (
    []
    if ARCH is None
    else [
        c
        for c in C.all_cases()
        if ARCH in c.archs and sdpa_dense_admits(request_from_case(c, ARCH))[0]
    ]
)


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_admitted_case_matches_sdpa_reference(case):
    req = request_from_case(case, ARCH)
    inputs = H.materialize(case)
    tensors = bind_tensors(case, H.to_torch(inputs, device="cuda"), req)

    binding = sdpa_dense_bind(req, tensors)
    binding.launch()
    binding.launch()
    torch.cuda.synchronize()

    o, lse = outputs(case, tensors, req)
    assert H.compare(case, o, lse, H.evaluate(inputs)) == []
