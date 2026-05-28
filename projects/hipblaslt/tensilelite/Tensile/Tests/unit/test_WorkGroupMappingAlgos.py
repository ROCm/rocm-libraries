# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for Tensile/Components/WorkGroupMappingAlgos.py
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from types import SimpleNamespace
from conftest import create_mock_writer


@pytest.mark.unit
class TestScalarUInt24DivideAndRemainderPair:
    """Test scalarUInt24DivideAndRemainderPair function"""

    def test_generates_division_module(self):
        """Should generate module for pairwise division"""
        from Tensile.Components.WorkGroupMappingAlgos import scalarUInt24DivideAndRemainderPair

        qReg = [0, 1]
        dReg = [2, 3]
        divReg = [4, 5]
        rReg = [6, 7]
        tmpVgprRes = SimpleNamespace(idx=10)
        wavewidth = 64

        module = scalarUInt24DivideAndRemainderPair(
            qReg, dReg, divReg, rReg, tmpVgprRes, wavewidth,
            doRemainder=True, doQuotient=True
        )

        # Module should be created
        assert module is not None
        # Check that module has instructions
        assert len(module.items()) > 0

    def test_quotient_only_mode(self):
        """Should generate module for quotient-only division"""
        from Tensile.Components.WorkGroupMappingAlgos import scalarUInt24DivideAndRemainderPair

        qReg = [0, 1]
        dReg = [2, 3]
        divReg = [4, 5]
        rReg = [6, 7]
        tmpVgprRes = SimpleNamespace(idx=10)
        wavewidth = 64

        module = scalarUInt24DivideAndRemainderPair(
            qReg, dReg, divReg, rReg, tmpVgprRes, wavewidth,
            doRemainder=False, doQuotient=True
        )

        assert module is not None

    def test_remainder_only_mode(self):
        """Should generate module for remainder-only division"""
        from Tensile.Components.WorkGroupMappingAlgos import scalarUInt24DivideAndRemainderPair

        qReg = [0, 1]
        dReg = [2, 3]
        divReg = [4, 5]
        rReg = [6, 7]
        tmpVgprRes = SimpleNamespace(idx=10)
        wavewidth = 64

        module = scalarUInt24DivideAndRemainderPair(
            qReg, dReg, divReg, rReg, tmpVgprRes, wavewidth,
            doRemainder=True, doQuotient=False
        )

        assert module is not None

    def test_wavewidth_32(self):
        """Should handle wavewidth 32"""
        from Tensile.Components.WorkGroupMappingAlgos import scalarUInt24DivideAndRemainderPair

        qReg = [0, 1]
        dReg = [2, 3]
        divReg = [4, 5]
        rReg = [6, 7]
        tmpVgprRes = SimpleNamespace(idx=10)
        wavewidth = 32  # Different wavewidth

        module = scalarUInt24DivideAndRemainderPair(
            qReg, dReg, divReg, rReg, tmpVgprRes, wavewidth
        )

        assert module is not None



@pytest.mark.unit
class TestChipletTransformChunked:
    """Test chiplet_transform_chunked function"""

    # Removed: use shared create_mock_writer() from conftest.py

    def test_generates_transform_module(self):
        """Should generate chiplet transform module"""
        from Tensile.Components.WorkGroupMappingAlgos import chiplet_transform_chunked

        writer = create_mock_writer(add_labels=True, label_prefix="ChipletLabel")
        kernel = {"WavefrontSize": 64}
        sgprNumXCC = 5
        sgprIndex = 6
        sgprNumWG = 7
        sgprChunkSize = 8

        module = chiplet_transform_chunked(
            writer, kernel, sgprNumXCC, sgprIndex, sgprNumWG, sgprChunkSize
        )

        assert module is not None
        # Should allocate sgprs
        assert writer.sgprPool.checkOut.call_count >= 4



@pytest.mark.unit
class TestChipletTransform:
    """Test chiplet_transform function"""

    # Removed: use shared create_mock_writer() from conftest.py

    def test_generates_transform_module(self, basic_kernel, sgpr_alloc):
        """Should generate 1D workgroup ID remap module"""
        from Tensile.Components.WorkGroupMappingAlgos import chiplet_transform

        writer = create_mock_writer()
        sgprIndex, sgprNumTilesM, sgprNumTilesN = sgpr_alloc.alloc_many(3)

        module = chiplet_transform(writer, basic_kernel, sgprIndex, sgprNumTilesM, sgprNumTilesN)

        assert module is not None
        # Should allocate multiple sgprs
        assert writer.sgprPool.checkOut.call_count >= 3


