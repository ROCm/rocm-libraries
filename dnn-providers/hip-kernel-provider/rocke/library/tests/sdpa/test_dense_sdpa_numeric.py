# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The dense SDPA binding on GPU, against the case-table oracle.

Every ``cases.py`` case the dense binding admits on the live arch, all tiers,
is bound with ``sdpa_dense_bind``, launched twice (the second launch reuses
the bound buffers) and compared with ``sdpa_reference``, fully masked rows
included (O = 0, LSE = -inf). The arrays are rocKE ``DeviceArray`` s, so no
torch is needed; a few cases also run with torch tensors, the production
callers' kind.
"""

from __future__ import annotations

import pytest
from rocke.tests.gpu import probe_device_arch
from rocke.tests.lanes import TORCH_IN_CI, arch_skip_reason, import_or_skip_until

from dispatch.sdpa.bind_dense import sdpa_dense_admits, sdpa_dense_bind
from tests.sdpa import cases as C
from tests.sdpa.binding_harness import bind_and_check, torch_caller_cases
from tests.sdpa.request_bridge import request_from_case

_ARCHES = ("gfx942", "gfx950")
_FOUND, _ABSENCE = probe_device_arch()
ARCH = _FOUND if _FOUND in _ARCHES else None

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH is None,
        reason=arch_skip_reason(
            _ARCHES,
            f"the dense SDPA binding numerics need one of {list(_ARCHES)}; "
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
        if ARCH in c.archs and sdpa_dense_admits(request_from_case(c, ARCH))[0]
    ]
)


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_admitted_case_matches_sdpa_reference(case):
    req = request_from_case(case, ARCH)
    assert bind_and_check(sdpa_dense_bind, case, req, kind="device_array") == []


_TORCH_CASES = torch_caller_cases(_CASES)


@pytest.mark.parametrize("case", _TORCH_CASES, ids=[c.id for c in _TORCH_CASES])
def test_torch_callers_match_sdpa_reference(case):
    """The production callers pass torch tensors: a few cases with them."""
    import_or_skip_until("torch", TORCH_IN_CI)
    req = request_from_case(case, ARCH)
    assert bind_and_check(sdpa_dense_bind, case, req, kind="torch") == []
