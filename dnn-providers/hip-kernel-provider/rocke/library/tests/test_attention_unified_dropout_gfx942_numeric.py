# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dropout of the gfx942 unified tiled 2D kernel on the GPU: the mask bit for
bit, O and LSE (checks in ``_unified_dropout_gpu``) on the transposed 32x32x8
body (plain and with the sliced-K ring) and the 16x16x16 body with P in
registers (D256) or read back from LDS. gfx942 tiled 2D takes causal masks only.
"""

from __future__ import annotations

import pytest
from rocke.tests.gpu import current_device_arch_in

from ._unified_dropout_gpu import Case, check_mask, check_output_and_lse

pytestmark = pytest.mark.gpu

_ARCH = "gfx942"


@pytest.fixture
def gfx942():
    current_device_arch_in((_ARCH,), "the gfx942 tiled 2D dropout kernels")


def _body(spec) -> str:
    if spec.use_mfma_32x32x8:
        return "t32ring" if spec.use_k_sliced_ring else "t32"
    return "regpv" if spec.use_register_pv else "plds"


# V = identity: every kv_len <= head_size. Rows with q_len > kv_len start with
# keyless queries, whose O must stay 0.
_MASK_CASES = [
    Case("t32_llvm", (1024, 40, 700), (128, 100, 128), True, "bf16", 128, "t32", 0.5, "value", "llvm"),
    Case("t32ring", (1024, 40, 700), (128, 100, 128), True, "fp16", 128, "t32ring", 0.5, "device"),
    Case("regpv_d256", (200, 33, 77), (256, 77, 200), True, "bf16", 256, "regpv", 0.5, "mixed"),
    Case("plds_fp16", (128, 40, 100), (128, 100, 128), True, "fp16", 128, "plds", 0.5, "value"),
]  # fmt: skip
_OUTPUT_CASES = [
    Case("t32", (1024, 77, 513), (1024, 500, 600), True, "bf16", 128, "t32", 0.1, "device"),
    Case("t32ring", (1024, 77, 513), (1200, 500, 600), True, "fp16", 128, "t32ring", 0.3, "mixed"),
    Case("regpv_d256", (600, 40, 500), (600, 100, 500), True, "bf16", 256, "regpv", 0.5, "mixed"),
    Case("plds_fp16_d64", (64, 5, 90), (64, 90, 200), True, "fp16", 64, "plds", 0.3, "value"),
]  # fmt: skip


def _ids(cases):
    return [pytest.param(c, id=c.name) for c in cases]


@pytest.mark.parametrize("case", _ids(_MASK_CASES))
def test_mask_bit_exact(gfx942, case):
    check_mask(case, _ARCH, _body)


@pytest.mark.parametrize("case", _ids(_OUTPUT_CASES))
def test_output_and_lse(gfx942, case):
    check_output_and_lse(case, _ARCH, _body)
