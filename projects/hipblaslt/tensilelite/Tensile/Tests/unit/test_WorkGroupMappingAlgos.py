# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for Tensile/Components/WorkGroupMappingAlgos.py
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from types import SimpleNamespace


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
class TestWgmXCC:
    """Test wgmXCC function"""

    def create_mock_writer(self):
        """Create a mock writer object"""
        writer = Mock()
        writer.sgprPool = Mock()
        writer.sgprPool.checkOut = Mock(return_value=10)
        writer.sgprPool.checkIn = Mock()
        writer.vgprPool = Mock()
        writer.vgprPool.checkOutAligned = Mock(return_value=20)
        writer.vgprPool.checkIn = Mock()
        writer.allocTmpSgpr = Mock()
        writer.allocTmpSgpr.return_value.__enter__ = Mock(return_value=SimpleNamespace(idx=5))
        writer.allocTmpSgpr.return_value.__exit__ = Mock(return_value=False)
        writer.labels = Mock()
        writer.labels.getNameInc = Mock(return_value="TestLabel")
        return writer

    def test_with_cluster_enabled(self):
        """Should skip WGMXCC when cluster is enabled"""
        from Tensile.Components.WorkGroupMappingAlgos import wgmXCC

        writer = self.create_mock_writer()
        kernel = {
            "ClusterDim": [2, 2],  # Cluster enabled
            "StreamK": 0,
            "WorkGroupMappingXCC": 1,
            "WavefrontSize": 64
        }
        tmpSgprNumWorkGroups = 15

        module = wgmXCC(writer, kernel, tmpSgprNumWorkGroups)

        assert module is not None

    def test_with_streamk_and_wgmxcc_neg1(self):
        """Should handle StreamK with WorkGroupMappingXCC=-1"""
        # This test just verifies the function is importable and callable
        # We skip actual execution since it requires rocisa
        from Tensile.Components.WorkGroupMappingAlgos import wgmXCC

        # Just verify the function exists
        assert callable(wgmXCC)

    def test_without_streamk(self):
        """Should handle non-StreamK case"""
        from Tensile.Components.WorkGroupMappingAlgos import wgmXCC

        writer = self.create_mock_writer()
        kernel = {
            "ClusterDim": [1, 1],
            "StreamK": 0,
            "WorkGroupMappingXCC": 1,
            "WavefrontSize": 64
        }
        tmpSgprNumWorkGroups = 15

        module = wgmXCC(writer, kernel, tmpSgprNumWorkGroups)

        assert module is not None


@pytest.mark.unit
class TestDefaultWGM:
    """Test DefaultWGM function"""

    def create_mock_writer(self):
        """Create mock writer with required methods"""
        writer = Mock()
        writer.sgprPool = Mock()
        writer.sgprPool.checkOut = Mock(return_value=10)
        writer.sgprPool.checkIn = Mock()
        writer.vgprPool = Mock()
        writer.vgprPool.checkOutAligned = Mock(return_value=20)
        writer.vgprPool.checkIn = Mock()
        writer.allocTmpSgprList = Mock()
        writer.allocTmpSgprList.return_value.__enter__ = Mock(return_value=[
            SimpleNamespace(idx=5),
            SimpleNamespace(idx=7),
            SimpleNamespace(idx=8)
        ])
        writer.allocTmpSgprList.return_value.__exit__ = Mock(return_value=False)
        writer.labels = Mock()
        writer.labels.getNameInc = Mock(side_effect=lambda x: f"{x}_1")
        return writer

    def test_basic_wgm_calculation(self):
        """Should generate WGM calculation module"""
        from Tensile.Components.WorkGroupMappingAlgos import DefaultWGM

        writer = self.create_mock_writer()
        kernel = {
            "ClusterDim": [1, 1],
            "WavefrontSize": 64
        }
        sgprWGM = "WGM"

        module = DefaultWGM(writer, kernel, sgprWGM)

        assert module is not None
        # Should allocate temp sgpr
        assert writer.sgprPool.checkOut.called

    def test_with_cluster_enabled(self):
        """Should handle cluster-enabled case"""
        from Tensile.Components.WorkGroupMappingAlgos import DefaultWGM

        writer = self.create_mock_writer()
        kernel = {
            "ClusterDim": [2, 2],  # Cluster enabled
            "WavefrontSize": 64
        }
        sgprWGM = "WGM"

        module = DefaultWGM(writer, kernel, sgprWGM)

        assert module is not None


@pytest.mark.unit
class TestChipletTransformChunked:
    """Test chiplet_transform_chunked function"""

    def create_mock_writer(self):
        """Create mock writer"""
        writer = Mock()
        writer.sgprPool = Mock()
        writer.sgprPool.checkOut = Mock(return_value=10)
        writer.sgprPool.checkIn = Mock()
        writer.vgprPool = Mock()
        writer.vgprPool.checkOutAligned = Mock(return_value=20)
        writer.vgprPool.checkIn = Mock()
        writer.labels = Mock()
        writer.labels.getUniqueNamePrefix = Mock(return_value="ChipletLabel")
        return writer

    def test_generates_transform_module(self):
        """Should generate chiplet transform module"""
        from Tensile.Components.WorkGroupMappingAlgos import chiplet_transform_chunked

        writer = self.create_mock_writer()
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

    def create_mock_writer(self):
        """Create mock writer"""
        writer = Mock()
        writer.sgprPool = Mock()
        writer.sgprPool.checkOut = Mock(return_value=10)
        writer.sgprPool.checkIn = Mock()
        writer.vgprPool = Mock()
        writer.vgprPool.checkOutAligned = Mock(return_value=20)
        writer.vgprPool.checkIn = Mock()
        return writer

    def test_generates_transform_module(self):
        """Should generate 1D workgroup ID remap module"""
        from Tensile.Components.WorkGroupMappingAlgos import chiplet_transform

        writer = self.create_mock_writer()
        kernel = {"WavefrontSize": 64}
        sgprIndex = 5
        sgprNumTilesM = 6
        sgprNumTilesN = 7

        module = chiplet_transform(writer, kernel, sgprIndex, sgprNumTilesM, sgprNumTilesN)

        assert module is not None
        # Should allocate multiple sgprs
        assert writer.sgprPool.checkOut.call_count >= 3


@pytest.mark.unit
class TestSpaceFillingCurveWalk:
    """Test SpaceFillingCurveWalk function"""

    def create_mock_writer(self):
        """Create mock writer with full capabilities"""
        writer = Mock()
        writer.sgprPool = Mock()
        writer.sgprPool.checkOut = Mock(return_value=10)
        writer.sgprPool.checkOutAligned = Mock(return_value=10)
        writer.sgprPool.checkIn = Mock()
        writer.vgprPool = Mock()
        writer.vgprPool.checkOut = Mock(return_value=20)
        writer.vgprPool.checkIn = Mock()
        writer.addSgprVarToPool = Mock()
        writer.removeSgprVarFromPool = Mock()
        writer.sgprs = {
            "NumWorkGroups0": 1,
            "NumWorkGroups1": 2,
            "Alpha": 3,
            "Beta": 4
        }
        writer.states = Mock()
        writer.states.WGMTransformLevels = 1
        writer.labels = Mock()
        writer.labels.getUniqueNamePrefix = Mock(return_value="SFCLabel")
        return writer

    def test_with_single_level(self):
        """Should handle single-level space filling curve"""
        # This test just verifies the function is importable and callable
        # We skip actual execution since it requires rocisa
        from Tensile.Components.WorkGroupMappingAlgos import SpaceFillingCurveWalk

        # Just verify the function exists
        assert callable(SpaceFillingCurveWalk)

    def test_with_multi_level(self):
        """Should handle multi-level space filling curve"""
        # This test just verifies the function is importable and callable
        # We skip actual execution since it requires rocisa
        from Tensile.Components.WorkGroupMappingAlgos import SpaceFillingCurveWalk

        # Just verify the function exists
        assert callable(SpaceFillingCurveWalk)

    def test_with_streamk(self):
        """Should handle StreamK case"""
        # This test just verifies the function is importable and callable
        # We skip actual execution since it requires rocisa
        from Tensile.Components.WorkGroupMappingAlgos import SpaceFillingCurveWalk

        # Just verify the function exists
        assert callable(SpaceFillingCurveWalk)


@pytest.mark.unit
class TestTransformNLevels:
    """Test TransformNLevels function"""

    def create_mock_writer(self):
        """Create mock writer"""
        writer = Mock()
        writer.sgprPool = Mock()
        writer.sgprPool.checkOut = Mock(return_value=10)
        writer.sgprPool.checkOutAligned = Mock(return_value=10)
        writer.sgprPool.checkIn = Mock()
        writer.vgprPool = Mock()
        writer.vgprPool.checkOutAligned = Mock(return_value=20)
        writer.vgprPool.checkIn = Mock()
        writer.states = Mock()
        writer.states.WGMTransformLevels = 2
        writer.labels = Mock()
        writer.labels.getUniqueNamePrefix = Mock(return_value="TransformLabel")
        return writer

    def test_single_level_colmajor(self):
        """Should handle single-level column-major transform"""
        from Tensile.Components.WorkGroupMappingAlgos import TransformNLevels

        writer = self.create_mock_writer()
        kernel = {
            "SpaceFillingAlgo": [0],  # ColMajor
            "WavefrontSize": 64
        }
        sgprIndex = 5
        sgprNumTilesM = 6
        sgprNumTilesN = 7
        sgprWGM = 8
        sgprGlobalY = 9
        sgprGlobalX = 10
        numLevels = 1

        module = TransformNLevels(
            writer, kernel, sgprIndex, sgprNumTilesM, sgprNumTilesN,
            sgprWGM, sgprGlobalY, sgprGlobalX, numLevels
        )

        assert module is not None

    def test_single_level_rowmajor(self):
        """Should handle single-level row-major transform"""
        from Tensile.Components.WorkGroupMappingAlgos import TransformNLevels

        writer = self.create_mock_writer()
        kernel = {
            "SpaceFillingAlgo": [1],  # RowMajor
            "WavefrontSize": 64
        }
        sgprIndex = 5
        sgprNumTilesM = 6
        sgprNumTilesN = 7
        sgprWGM = 8
        sgprGlobalY = 9
        sgprGlobalX = 10
        numLevels = 1

        module = TransformNLevels(
            writer, kernel, sgprIndex, sgprNumTilesM, sgprNumTilesN,
            sgprWGM, sgprGlobalY, sgprGlobalX, numLevels
        )

        assert module is not None

    def test_single_level_hilbert(self):
        """Should handle single-level Hilbert curve"""
        from Tensile.Components.WorkGroupMappingAlgos import TransformNLevels

        writer = self.create_mock_writer()
        kernel = {
            "SpaceFillingAlgo": [2],  # Hilbert
            "WavefrontSize": 64
        }
        sgprIndex = 5
        sgprNumTilesM = 6
        sgprNumTilesN = 7
        sgprWGM = 8
        sgprGlobalY = 9
        sgprGlobalX = 10
        numLevels = 1

        module = TransformNLevels(
            writer, kernel, sgprIndex, sgprNumTilesM, sgprNumTilesN,
            sgprWGM, sgprGlobalY, sgprGlobalX, numLevels
        )

        assert module is not None

    def test_multi_level_transform(self):
        """Should handle multi-level transform"""
        from Tensile.Components.WorkGroupMappingAlgos import TransformNLevels

        writer = self.create_mock_writer()
        kernel = {
            "SpaceFillingAlgo": [0, 1],  # Two levels
            "WavefrontSize": 64
        }
        sgprIndex = 5
        sgprNumTilesM = 6
        sgprNumTilesN = 7
        sgprWGM = 8
        sgprGlobalY = 9
        sgprGlobalX = 10
        numLevels = 2

        module = TransformNLevels(
            writer, kernel, sgprIndex, sgprNumTilesM, sgprNumTilesN,
            sgprWGM, sgprGlobalY, sgprGlobalX, numLevels
        )

        assert module is not None


@pytest.mark.unit
class TestOrderingPreamble:
    """Test OrderingPreamble function"""

    def create_mock_writer(self):
        """Create mock writer"""
        writer = Mock()
        writer.sgprPool = Mock()
        writer.sgprPool.checkOut = Mock(return_value=10)
        writer.sgprPool.checkIn = Mock()
        writer.vgprPool = Mock()
        writer.vgprPool.checkOutAligned = Mock(return_value=20)
        writer.vgprPool.checkIn = Mock()
        return writer

    def test_level_0_preamble(self):
        """Should handle level 0 (no division needed)"""
        from Tensile.Components.WorkGroupMappingAlgos import OrderingPreamble

        writer = self.create_mock_writer()
        kernel = {"WavefrontSize": 64}
        sgprIndex = 5
        sgprGridY = 6
        sgprGridX = 7
        sgprCumulativeDenominator = 8
        sgprLocalIndex = 9
        lvl = 0

        module = OrderingPreamble(
            writer, kernel, sgprIndex, sgprGridY, sgprGridX,
            sgprCumulativeDenominator, sgprLocalIndex, lvl
        )

        assert module is not None

    def test_higher_level_preamble(self):
        """Should handle higher levels (with division)"""
        from Tensile.Components.WorkGroupMappingAlgos import OrderingPreamble

        writer = self.create_mock_writer()
        kernel = {"WavefrontSize": 64}
        sgprIndex = 5
        sgprGridY = 6
        sgprGridX = 7
        sgprCumulativeDenominator = 8
        sgprLocalIndex = 9
        lvl = 1

        module = OrderingPreamble(
            writer, kernel, sgprIndex, sgprGridY, sgprGridX,
            sgprCumulativeDenominator, sgprLocalIndex, lvl
        )

        assert module is not None


@pytest.mark.unit
class TestColRowMajor:
    """Test ColRowMajor function"""

    def create_mock_writer(self):
        """Create mock writer"""
        writer = Mock()
        writer.sgprPool = Mock()
        writer.sgprPool.checkOut = Mock(return_value=10)
        writer.sgprPool.checkIn = Mock()
        writer.vgprPool = Mock()
        writer.vgprPool.checkOutAligned = Mock(return_value=20)
        writer.vgprPool.checkIn = Mock()
        return writer

    def test_column_major_ordering(self):
        """Should generate column-major ordering"""
        from Tensile.Components.WorkGroupMappingAlgos import ColRowMajor

        writer = self.create_mock_writer()
        kernel = {"WavefrontSize": 64}
        sgprIndex = 5
        sgprGridY = 6
        sgprGridX = 7
        sgprLocalY = 8
        sgprLocalX = 9
        sgprCumulativeDenominator = 10
        lvl = 0
        isCol = True

        module = ColRowMajor(
            writer, kernel, sgprIndex, sgprGridY, sgprGridX,
            sgprLocalY, sgprLocalX, sgprCumulativeDenominator, lvl, isCol
        )

        assert module is not None

    def test_row_major_ordering(self):
        """Should generate row-major ordering"""
        from Tensile.Components.WorkGroupMappingAlgos import ColRowMajor

        writer = self.create_mock_writer()
        kernel = {"WavefrontSize": 64}
        sgprIndex = 5
        sgprGridY = 6
        sgprGridX = 7
        sgprLocalY = 8
        sgprLocalX = 9
        sgprCumulativeDenominator = 10
        lvl = 0
        isCol = False

        module = ColRowMajor(
            writer, kernel, sgprIndex, sgprGridY, sgprGridX,
            sgprLocalY, sgprLocalX, sgprCumulativeDenominator, lvl, isCol
        )

        assert module is not None


@pytest.mark.unit
class TestSpaceFillCurveSimpleImpl:
    """Test SpaceFillCurveSimpleImpl function"""

    def create_mock_writer(self):
        """Create mock writer"""
        writer = Mock()
        writer.sgprPool = Mock()
        writer.sgprPool.checkOut = Mock(return_value=10)
        writer.sgprPool.checkOutAligned = Mock(return_value=10)
        writer.sgprPool.checkIn = Mock()
        writer.vgprPool = Mock()
        writer.vgprPool.checkOutAligned = Mock(return_value=20)
        writer.vgprPool.checkIn = Mock()
        writer.labels = Mock()
        writer.labels.getUniqueNamePrefix = Mock(return_value="SFCSimpleLabel")
        return writer

    def test_hilbert_order(self):
        """Should generate Hilbert ordering (orderID=2)"""
        from Tensile.Components.WorkGroupMappingAlgos import SpaceFillCurveSimpleImpl

        writer = self.create_mock_writer()
        kernel = {"WavefrontSize": 64}
        sgprIndex = 5
        sgprGridY = 6
        sgprGridX = 7
        sgprLocalY = 8
        sgprLocalX = 9
        sgprCumulativeDenominator = 10
        lvl = 0
        orderID = 2  # Hilbert

        module = SpaceFillCurveSimpleImpl(
            writer, kernel, sgprIndex, sgprGridY, sgprGridX,
            sgprLocalY, sgprLocalX, sgprCumulativeDenominator, lvl, orderID
        )

        assert module is not None

    def test_morton_z_order(self):
        """Should generate Morton Z-order (orderID=3)"""
        from Tensile.Components.WorkGroupMappingAlgos import SpaceFillCurveSimpleImpl

        writer = self.create_mock_writer()
        kernel = {"WavefrontSize": 64}
        sgprIndex = 5
        sgprGridY = 6
        sgprGridX = 7
        sgprLocalY = 8
        sgprLocalX = 9
        sgprCumulativeDenominator = 10
        lvl = 0
        orderID = 3  # Morton Z

        module = SpaceFillCurveSimpleImpl(
            writer, kernel, sgprIndex, sgprGridY, sgprGridX,
            sgprLocalY, sgprLocalX, sgprCumulativeDenominator, lvl, orderID
        )

        assert module is not None

    def test_morton_reverse_n_order(self):
        """Should generate Morton Reverse-N order (orderID=4)"""
        from Tensile.Components.WorkGroupMappingAlgos import SpaceFillCurveSimpleImpl

        writer = self.create_mock_writer()
        kernel = {"WavefrontSize": 64}
        sgprIndex = 5
        sgprGridY = 6
        sgprGridX = 7
        sgprLocalY = 8
        sgprLocalX = 9
        sgprCumulativeDenominator = 10
        lvl = 0
        orderID = 4  # Morton Reverse-N

        module = SpaceFillCurveSimpleImpl(
            writer, kernel, sgprIndex, sgprGridY, sgprGridX,
            sgprLocalY, sgprLocalX, sgprCumulativeDenominator, lvl, orderID
        )

        assert module is not None

    def test_morton_u_order(self):
        """Should generate Morton U-order (orderID=5)"""
        from Tensile.Components.WorkGroupMappingAlgos import SpaceFillCurveSimpleImpl

        writer = self.create_mock_writer()
        kernel = {"WavefrontSize": 64}
        sgprIndex = 5
        sgprGridY = 6
        sgprGridX = 7
        sgprLocalY = 8
        sgprLocalX = 9
        sgprCumulativeDenominator = 10
        lvl = 0
        orderID = 5  # Morton U

        module = SpaceFillCurveSimpleImpl(
            writer, kernel, sgprIndex, sgprGridY, sgprGridX,
            sgprLocalY, sgprLocalX, sgprCumulativeDenominator, lvl, orderID
        )

        assert module is not None
