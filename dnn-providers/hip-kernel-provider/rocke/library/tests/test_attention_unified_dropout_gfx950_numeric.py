# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dropout of the gfx950 unified tiled 2D kernel on the GPU: the mask bit for
bit, O and LSE (checks in ``_unified_dropout_gpu``) on the transposed 32x32
body and the 16x16 body with P in registers or read back from LDS.
"""

from __future__ import annotations

import pytest
from rocke.tests.gpu import current_device_arch_in

from ._unified_dropout_gpu import Case, check_mask, check_output_and_lse

pytestmark = pytest.mark.gpu

_ARCH = "gfx950"


@pytest.fixture
def gfx950():
    current_device_arch_in((_ARCH,), "the gfx950 tiled 2D dropout kernels")


def _body(spec) -> str:
    if spec.use_mfma_32x32:
        return "t32"
    return "regpv" if spec.use_register_pv else "plds"


# V = identity: non-causal, every kv_len <= head_size.
_MASK_CASES = [
    Case("t32_llvm", (300, 40, 260), (128, 100, 128), False, "bf16", 128, "t32", 0.5, "value", "llvm"),
    Case("t32_hipcc", (300, 40, 260), (128, 100, 128), False, "bf16", 128, "t32", 0.5, "device", "hipcc"),
    Case("regpv", (200, 33, 77), (128, 77, 120), False, "bf16", 128, "regpv", 0.5, "mixed"),
    Case("plds_fp16", (200, 33, 77), (128, 77, 120), False, "fp16", 128, "plds", 0.5, "value"),
]  # fmt: skip
_OUTPUT_CASES = [
    Case("t32_causal", (300, 77, 513), (300, 500, 600), True, "bf16", 128, "t32", 0.1, "device"),
    Case("regpv", (200, 33, 128), (256, 700, 128), False, "bf16", 128, "regpv", 0.5, "mixed"),
    Case("plds_fp16_d64_causal", (64, 5, 90), (64, 90, 200), True, "fp16", 64, "plds", 0.3, "value"),
]  # fmt: skip


def _ids(cases):
    return [pytest.param(c, id=c.name) for c in cases]


@pytest.mark.parametrize("case", _ids(_MASK_CASES))
def test_mask_bit_exact(gfx950, case):
    check_mask(case, _ARCH, _body)


@pytest.mark.parametrize("case", _ids(_OUTPUT_CASES))
def test_output_and_lse(gfx950, case):
    check_output_and_lse(case, _ARCH, _body)
