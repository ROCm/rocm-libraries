# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Only a hipBLASLt build without rocRoller may select the checked-in rocRoller kernels.

A rocRoller build serves these problems through rocRoller itself, and its Tensile
paths (HIPBLASLT_USE_ROCROLLER=0, the hipblaslt_ext APIs, index lookups) must
behave as if the captures were not checked in. Each capture therefore carries the
RocRollerCaptures hardware predicate, which only hardware marked by such a build
satisfies. Being a hardware predicate, it is evaluated by every library search
type, including the GEMM-type-only enumeration that skips problem predicates.
"""

import pytest

from tensilelite import Hardware
from tensilelite.Common import state
from tensilelite.Contractions import _isCheckedInRocRollerCapture

pytestmark = pytest.mark.unit

_RR_KERNEL = "RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_"
_LEGACY_RR_KERNEL = (
    "RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_"
    "LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_"
    "SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e"
)
_AITER_KERNEL = "_ZN5aiter24bf16gemm_bf16_tn_256x256E"


def _gpu_terms(predicate):
    value = state(predicate)["value"]
    return value["value"] if value["type"] == "And" else [value]


def test_a_capture_requires_rocroller_captures():
    predicate = Hardware.HardwarePredicate.FromHardware(
        (9, 5, 0), None, ["Device 75a0"], rocRollerCapture=True
    )
    assert state(predicate)["type"] == "AMDGPU"
    assert {"type": "RocRollerCaptures"} in _gpu_terms(predicate)


def test_the_capture_term_adds_to_the_device_terms():
    plain = Hardware.HardwarePredicate.FromHardware((9, 5, 0), None, ["Device 75a0"])
    capture = Hardware.HardwarePredicate.FromHardware(
        (9, 5, 0), None, ["Device 75a0"], rocRollerCapture=True
    )
    assert _gpu_terms(capture) == _gpu_terms(plain) + [{"type": "RocRollerCaptures"}]


def test_other_solutions_keep_their_hardware_predicate():
    for deviceNames in (None, ["Device 75a0"]):
        predicate = Hardware.HardwarePredicate.FromHardware((9, 5, 0), None, deviceNames)
        assert {"type": "RocRollerCaptures"} not in _gpu_terms(predicate)


@pytest.mark.parametrize(
    "customKernel,expected",
    [
        ({"name": _RR_KERNEL}, True),
        ({"name": _RR_KERNEL, "generated": True}, False),
        ({"name": _LEGACY_RR_KERNEL}, False),
        ({"name": _AITER_KERNEL}, False),
        (None, False),
    ],
)
def test_only_checked_in_rocroller_mx_captures_get_it(customKernel, expected):
    solutionState = {} if customKernel is None else {"CustomKernel": customKernel}
    assert _isCheckedInRocRollerCapture(solutionState) is expected
