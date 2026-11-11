# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The unified SDPA binding on GPU, against the case-table oracle.

Every ``cases.py`` case the binding admits on the live arch, all tiers, is
bound with ``sdpa_unified_bind``, launched twice (the second launch
reuses the bound buffers, as graph replay does) and compared with
``sdpa_reference`` on the valid rows, fully masked rows included (O = 0,
LSE = -inf).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dispatch.sdpa.bind_unified import (
    _LaunchPlan,
    sdpa_unified_admits,
    sdpa_unified_bind,
)
from rocke.core.arch import base_arch_from_target_id

from tests.sdpa import cases as C
from tests.sdpa import helpers as H
from tests.sdpa.binding_harness import bind_tensors, outputs
from tests.sdpa.request_bridge import request_from_case
from tests.sdpa.request_cases import make_request


def _arch():
    if not torch.cuda.is_available():
        return None
    arch = base_arch_from_target_id(torch.cuda.get_device_properties(0).gcnArchName)
    return arch if arch in ("gfx942", "gfx950") else None


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
        if ARCH in c.archs and sdpa_unified_admits(request_from_case(c, ARCH))[0]
    ]
)


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_admitted_case_matches_sdpa_reference(case):
    req = request_from_case(case, ARCH)
    inputs = H.materialize(case)
    tensors = bind_tensors(case, H.to_torch(inputs, device="cuda"), req)

    binding = sdpa_unified_bind(req, tensors)
    binding.launch()
    binding.launch()
    torch.cuda.synchronize()

    o, lse = outputs(case, tensors, req)
    assert H.compare(case, o, lse, H.evaluate(inputs)) == []


def _dense_cuda(req, name):
    desc = getattr(req, name)
    buf = torch.randn(
        1 + sum((n - 1) * s for n, s in zip(desc.dims, desc.strides)), device="cuda"
    )
    return torch.as_strided(buf.half(), desc.dims, desc.strides)


def test_an_out_of_range_block_table_physical_id_is_rejected_before_launch():
    """The binding's own paged K/V cache is identity-mapped, so a physical id
    the binding writes is always in range; this proves the bind-time check
    (``validate_tuning_attention_tensors``, which the real ``sdpa_unified_bind``
    calls before any kernel runs, not device-side) still catches a corrupted
    entry, rather than trusting the block table blindly.
    """
    from dispatch.attention.bindings import validate_tuning_attention_tensors

    req = make_request(arch=ARCH, d=64)
    tensors = {n: _dense_cuda(req, n) for n in ("q", "k", "v", "o")}
    plan = _LaunchPlan(req, tensors)
    num_blocks = plan.k_cache.shape[0]
    plan.block_table[0, 0] = num_blocks  # one past the last valid physical block
    with pytest.raises(ValueError, match=r"outside \[0"):
        validate_tuning_attention_tensors(plan.problem, plan.kernel_tensors())
