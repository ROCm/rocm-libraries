# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LdsXorSwizzle is part of the kernel name only when enabled."""

import pytest

from tensilelite.SolutionStructs.Naming import getKernelNameMin


@pytest.mark.unit
@pytest.mark.parametrize(
    "value, expected",
    [
        (True, "Cij_Aik_Bjk_S_B_UserArgs_SN_LXS1_WGMXCC1"),
        (False, "Cij_Aik_Bjk_S_B_UserArgs_SN_WGMXCC1"),
        (None, "Cij_Aik_Bjk_S_B_UserArgs_SN_WGMXCC1"),
    ],
    ids=["enabled", "disabled", "absent"],
)
def test_lds_xor_swizzle_named_only_when_enabled(value, expected):
    state = {
        "ProblemType": {"DataType": 0, "GroupedGemm": False},
        "GlobalSplitU": 1,
        "WorkGroupMappingXCC": 1,
        "UseCustomMainLoopSchedule": False,
        "SpaceFillingAlgo": [],
    }
    if value is not None:
        state["LdsXorSwizzle"] = value

    assert getKernelNameMin(state, splitGSU=False) == expected
