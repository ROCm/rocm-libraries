################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import pytest

from Tensile.Common.ValidParameters import checkParametersAreValid, validParameters


# Parameters whose value is a geometry-derived byte size (auto-derived when set
# to -1). Their valid set is irregular and effectively unbounded, so they use the
# scalar -1 sentinel to opt out of the discrete value-list check. Real constraints
# are enforced downstream in Solution.calcLdsBlockSizePerPad / checkLdsBlockSizePerPadForTDM.
LDS_BLOCK_SIZE_PER_PAD_PARAMS = [
    "LdsBlockSizePerPadA",
    "LdsBlockSizePerPadMXSA",
    "LdsBlockSizePerPadB",
    "LdsBlockSizePerPadMXSB",
    "LdsBlockSizePerPadMetadata",
]


class TestLdsBlockSizePerPadSentinel:
    """LdsBlockSizePerPad* use the -1 sentinel, so their value check is skipped."""

    @pytest.mark.parametrize("param", LDS_BLOCK_SIZE_PER_PAD_PARAMS)
    def test_param_uses_minus_one_sentinel(self, param):
        assert validParameters[param] == -1

    @pytest.mark.parametrize("param", LDS_BLOCK_SIZE_PER_PAD_PARAMS)
    # Values that the generator can derive from problem geometry. The old discrete
    # list ([-1, 0, 64, 128, 256, 512, 1024, 2048]) wrongly rejected anything above
    # 2048 and every non-power-of-two (e.g. 96, 3072), breaking configs that loaded
    # generated library logic back through Tensile. They must all be accepted now.
    @pytest.mark.parametrize("value", [-1, 0, 16, 96, 2048, 3072, 4096, 7680, 8192])
    def test_derived_values_accepted(self, param, value):
        # Must not raise.
        checkParametersAreValid((param, [value]), validParameters)

    @pytest.mark.parametrize("param", LDS_BLOCK_SIZE_PER_PAD_PARAMS)
    def test_list_of_values_accepted(self, param):
        # checkParametersAreValid iterates over all values in the list.
        checkParametersAreValid((param, [0, 96, 4096, 8192]), validParameters)


class TestCheckParametersAreValidFramework:
    """The sentinel only opts these params out; the rest of the check still works."""

    def test_unknown_parameter_name_rejected(self):
        with pytest.raises(Exception, match="Invalid parameter name"):
            checkParametersAreValid(("LdsBlockSizePerPadBogus", [4096]), validParameters)

    def test_discrete_param_still_rejects_invalid_value(self):
        # LdsPadA is a genuine discrete-choice param; its list check is unaffected.
        with pytest.raises(Exception, match="Invalid parameter value"):
            checkParametersAreValid(("LdsPadA", [7]), validParameters)

    def test_discrete_param_accepts_valid_value(self):
        checkParametersAreValid(("LdsPadA", [8]), validParameters)
