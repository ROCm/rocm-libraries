# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""All ordered gfx950 mixed-format pairs with independent K32 scales."""

from itertools import product

import pytest
from rocke.tests.arch_marks import arch_specific

pytestmark = pytest.mark.gpu

FORMATS = ("fp8", "bf8", "fp6", "bf6", "fp4")


@arch_specific(
    ("gfx950",),
    reason="scaled MFMA exists on gfx950 only; gfx942 refuses it (test_wrong_target_rejection in tests/core/test_gfx950_scaled_mfma.py checks the refusal)",
)
@pytest.mark.parametrize(
    "a,b", [(a, b) for a, b in product(FORMATS, repeat=2) if a != b]
)
@pytest.mark.parametrize("output", ["bf16", "fp16"])
@pytest.mark.parametrize("route", ["comgr", "hip"])
@pytest.mark.parametrize("M,N,K", [(32, 64, 128), (64, 96, 256)])
def test_numeric(gfx950_numeric, a, b, output, route, M, N, K):
    gfx950_numeric(a, b, output, route, M, N, K)


@arch_specific(
    ("gfx950",),
    reason="scaled MFMA exists on gfx950 only; gfx942 refuses it (test_wrong_target_rejection in tests/core/test_gfx950_scaled_mfma.py checks the refusal)",
)
@pytest.mark.parametrize(
    "a,b", [("fp8", "fp8"), ("bf8", "fp4"), ("fp6", "bf6"), ("fp4", "fp4")]
)
@pytest.mark.parametrize("route", ["comgr", "hip"])
@pytest.mark.parametrize("output", ["bf16", "fp16"])
def test_single_k64_atom(gfx950_numeric, a, b, output, route):
    gfx950_numeric(a, b, output, route, 64, 96, 64)
