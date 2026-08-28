################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
import pytest

from rocisa.enum import DataTypeEnum

from Tensile.Common.DataType import DataType
from Tensile.SolutionStructs.Solution import Solution


def _build_state(*, data_type_a, data_type_b, tdm_inst=3, num_threads=256, sparse=0):
    """Minimal state dict for Solution.setGlobalReadVectorWidth TDM tests."""
    return {
        "TDMInst": tdm_inst,
        "NumThreads": num_threads,
        "ProblemType": {
            "DataTypeA": data_type_a,
            "DataTypeB": data_type_b,
            "Sparse": sparse,
        },
    }


class TestTDMGlobalReadVectorWidth:
    """Verify setGlobalReadVectorWidth picks a valid GRVW for TDM tensors.

    TDM loads via tensor_load_to_lds, and GRVW is kept small so small-size
    solution rejection still fires. But the global-read width (GRVW*bpe/bpr)
    must map to a real load instruction: the smallest is b8 = 0.25 dwords
    (1 byte), and the width must be a whole multiple of it, so GRVW*bpe must
    be an integer number of bytes. GRVW=1 is too small for sub-byte types, so
    it is bumped to the minimum that yields an integer byte count:
        FP4 (0.5 B/elem) -> 2  (1 byte)
        FP6 (0.75 B/elem) -> 4 (3 bytes = exactly 4 packed 6-bit elements)
    All other types stay at 1. NumLoads*/NumLoadsCoalesced*/NumLoadsPerpendicular*
    are always normalized to 1 for TDM.
    """

    @pytest.mark.parametrize("tc", ["A", "B"])
    def test_fp4_bumped_to_2(self, tc):
        """FP4: GRVW forced to 2 so the load width is 0.25 dw (1 byte)."""
        state = _build_state(
            data_type_a=DataType(DataTypeEnum.Float4),
            data_type_b=DataType(DataTypeEnum.Float4),
        )

        Solution.setGlobalReadVectorWidth(state, tc, 999, 8, False)

        assert state["GlobalReadVectorWidth%s" % tc] == 2

    @pytest.mark.parametrize("enum", [DataTypeEnum.Float6, DataTypeEnum.BFloat6])
    def test_fp6_bumped_to_4(self, enum):
        """FP6/BFloat6: GRVW forced to 4 so the load width is 0.75 dw (3 bytes)."""
        state = _build_state(
            data_type_a=DataType(enum),
            data_type_b=DataType(enum),
        )

        Solution.setGlobalReadVectorWidth(state, "A", 999, 8, False)

        assert state["GlobalReadVectorWidthA"] == 4

    def test_byte_or_larger_dtype_stays_1(self):
        """Types with bpe >= 1 (e.g. Half) keep the small GRVW=1 for TDM."""
        state = _build_state(
            data_type_a=DataType(DataTypeEnum.Half),
            data_type_b=DataType(DataTypeEnum.Half),
        )

        Solution.setGlobalReadVectorWidth(state, "A", 999, 8, False)

        assert state["GlobalReadVectorWidthA"] == 1

    @pytest.mark.parametrize("tc", ["MXSA", "MXSB"])
    def test_scale_tensor_stays_1(self, tc):
        """Scale tensors (MXSA/MXSB) are not sub-byte data — GRVW stays 1
        regardless of the A/B data type."""
        state = _build_state(
            data_type_a=DataType(DataTypeEnum.Float4),
            data_type_b=DataType(DataTypeEnum.Float4),
        )

        Solution.setGlobalReadVectorWidth(state, tc, 999, 8, False)

        assert state["GlobalReadVectorWidth%s" % tc] == 1

    @pytest.mark.parametrize("sparse,tc", [(1, "A"), (2, "B")])
    def test_sparse_tensor_bumped_to_4(self, sparse, tc):
        """The sparse tensor (Sparse==1 -> A, Sparse==2 -> B) loads 2:4 metadata
        that requires GRVW % 4 == 0, so it is bumped to 4 rather than forced to 1."""
        state = _build_state(
            data_type_a=DataType(DataTypeEnum.Half),
            data_type_b=DataType(DataTypeEnum.Half),
            sparse=sparse,
        )

        Solution.setGlobalReadVectorWidth(state, tc, 999, 8, False)

        assert state["GlobalReadVectorWidth%s" % tc] == 4

    def test_numloads_normalized_for_tdm(self):
        """TDM normalizes the per-thread tile-split params to 1."""
        state = _build_state(
            data_type_a=DataType(DataTypeEnum.Float4),
            data_type_b=DataType(DataTypeEnum.Float4),
        )

        Solution.setGlobalReadVectorWidth(state, "A", 999, 8, False)

        assert state["NumLoadsA"] == 1
        assert state["NumLoadsCoalescedA"] == 1
        assert state["NumLoadsPerpendicularA"] == 1

    def test_non_tdm_fp4_not_bumped(self):
        """Without TDM (TDMInst=0), FP4 follows the classic path and keeps the
        requested GRVW — the sub-byte bump is TDM-specific."""
        state = _build_state(
            data_type_a=DataType(DataTypeEnum.Float4),
            data_type_b=DataType(DataTypeEnum.Float4),
            tdm_inst=0,
        )

        # totalVectors divisible by NumThreads so the classic path does not reject.
        Solution.setGlobalReadVectorWidth(state, "A", 256, 8, False)

        assert state["GlobalReadVectorWidthA"] == 8
