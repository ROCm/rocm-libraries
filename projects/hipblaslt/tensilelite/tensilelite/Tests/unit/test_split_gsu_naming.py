# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression tests for split-GSU kernel names."""

from copy import deepcopy

import pytest

from Tensile.SolutionStructs.Naming import getKernelNameMin


@pytest.mark.unit
@pytest.mark.parametrize("gsu", [2, -1], ids=["first_split_value", "automatic"])
def test_split_gsu_kernel_name_uses_multi_variant_and_preserves_state(gsu):
    state = {
        "ProblemType": {"DataType": 0, "GroupedGemm": False},
        "GlobalSplitU": gsu,
        "WorkGroupMappingXCC": 1,
        "UseCustomMainLoopSchedule": False,
        "SpaceFillingAlgo": [],
    }
    original = deepcopy(state)

    assert getKernelNameMin(state, splitGSU=True) == (
        "Cij_Aik_Bjk_S_B_UserArgs_SN_GSUM_WGMXCC1"
    )
    assert state == original
