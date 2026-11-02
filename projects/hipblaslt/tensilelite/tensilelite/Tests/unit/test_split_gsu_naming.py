# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression tests for split-GSU kernel names."""

from copy import deepcopy

import pytest

from tensilelite.SolutionStructs.Naming import getKernelNameMin


@pytest.mark.unit
@pytest.mark.parametrize(
    "gsu, expected",
    [
        (2, "Cij_Aik_Bjk_S_B_UserArgs_SN_GSUM_WGMXCC1"),
        (-1, "Cij_Aik_Bjk_S_B_UserArgs_SN_GSUM_WGMXCC1"),
        # Values that are not split variants keep their own name.
        (1, "Cij_Aik_Bjk_S_B_UserArgs_SN_WGMXCC1"),
        (0, "Cij_Aik_Bjk_S_B_UserArgs_SN_GSU0_WGMXCC1"),
    ],
    ids=["first_split_value", "automatic", "single_split", "disabled"],
)
def test_split_gsu_kernel_name_uses_multi_variant_and_preserves_state(gsu, expected):
    state = {
        "ProblemType": {"DataType": 0, "GroupedGemm": False},
        "GlobalSplitU": gsu,
        "WorkGroupMappingXCC": 1,
        "UseCustomMainLoopSchedule": False,
        "SpaceFillingAlgo": [],
    }
    original = deepcopy(state)

    assert getKernelNameMin(state, splitGSU=True) == expected
    assert state == original
