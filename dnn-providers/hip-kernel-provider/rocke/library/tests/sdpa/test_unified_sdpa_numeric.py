# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The unified SDPA binding on GPU, against the case-table oracle.

Every ``cases.py`` case the binding admits on the live arch, all tiers, is
bound with ``sdpa_unified_bind``, launched twice (the second launch
reuses the bound buffers, as graph replay does) and compared with
``sdpa_reference`` on the valid rows, fully masked rows included (O = 0,
LSE = -inf). The arrays are rocKE ``DeviceArray`` s, so no torch is needed; a
few cases also run with torch tensors, the production callers' kind.
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.runtime.device_array import DeviceArray, host_values
from rocke.tests.gpu import probe_device_arch
from rocke.tests.lanes import TORCH_IN_CI, arch_skip_reason, import_or_skip_until

from dispatch.sdpa.bind_unified import (
    _LaunchPlan,
    sdpa_unified_admits,
    sdpa_unified_bind,
)
from tests.sdpa import cases as C
from tests.sdpa.binding_harness import bind_and_check, torch_caller_cases
from tests.sdpa.request_bridge import request_from_case
from tests.sdpa.request_cases import make_request

_ARCHES = ("gfx942", "gfx950")
_FOUND, _ABSENCE = probe_device_arch()
ARCH = _FOUND if _FOUND in _ARCHES else None

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH is None,
        reason=arch_skip_reason(
            _ARCHES,
            f"the unified SDPA binding numerics need one of {list(_ARCHES)}; "
            f"found {_FOUND or _ABSENCE}",
        ),
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
    assert bind_and_check(sdpa_unified_bind, case, req, kind="device_array") == []


_TORCH_CASES = torch_caller_cases(_CASES)


@pytest.mark.parametrize("case", _TORCH_CASES, ids=[c.id for c in _TORCH_CASES])
def test_torch_callers_match_sdpa_reference(case):
    """The production callers pass torch tensors: a few cases with them."""
    import_or_skip_until("torch", TORCH_IN_CI)
    req = request_from_case(case, ARCH)
    assert bind_and_check(sdpa_unified_bind, case, req, kind="torch") == []


def _dense_device(req, name):
    desc = getattr(req, name)
    span = 1 + sum((n - 1) * s for n, s in zip(desc.dims, desc.strides))
    values = np.random.default_rng(0).standard_normal(span).astype(np.float16)
    return DeviceArray(values).view(desc.dims, desc.strides)


def test_an_out_of_range_block_table_physical_id_is_rejected_before_launch():
    """The binding's own paged K/V cache is identity-mapped, so a physical id
    the binding writes is always in range; this proves the bind-time check
    (``validate_tuning_attention_tensors``, which the real ``sdpa_unified_bind``
    calls before any kernel runs, not device-side) still catches a corrupted
    entry, rather than trusting the block table blindly.
    """
    from dispatch.attention.bindings import validate_tuning_attention_tensors

    req = make_request(arch=ARCH, d=64)
    tensors = {n: _dense_device(req, n) for n in ("q", "k", "v", "o")}
    plan = _LaunchPlan(req, tensors)
    num_blocks = plan.k_cache.shape[0]
    table = host_values(plan.block_table)
    table[0, 0] = num_blocks  # one past the last valid physical block
    plan.block_table.write(table)
    with pytest.raises(ValueError, match=r"outside \[0"):
        validate_tuning_attention_tensors(plan.problem, plan.kernel_tensors())
