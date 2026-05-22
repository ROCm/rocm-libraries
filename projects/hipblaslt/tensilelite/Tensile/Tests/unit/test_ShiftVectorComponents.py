################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

import pytest
from unittest.mock import Mock, patch
from types import SimpleNamespace

# Import real rocisa objects to use in mocks
from rocisa.instruction import SMovB32, SWaitCnt
from rocisa.container import vgpr, sgpr
from rocisa.code import TextBlock

from Tensile.Components.ShiftVectorComponents import (
    ShiftVectorComponentsVALU,
    ShiftVectorComponentsMFMA
)


# Create mock functions that return REAL rocisa Item objects (not MagicMocks)
def mock_vectorStaticRemainder_returns_real_item(*args, **kwargs):
    """Return a real rocisa Item instead of MagicMock"""
    return SMovB32(dst=vgpr(0), src=sgpr(0))

def mock_vectorStaticDivide_returns_real_item(*args, **kwargs):
    """Return a real rocisa Item instead of MagicMock"""
    return SMovB32(dst=vgpr(1), src=sgpr(1))

def mock_vectorStaticMultiply_returns_real_item(*args, **kwargs):
    """Return a real rocisa Item instead of MagicMock"""
    return SMovB32(dst=vgpr(2), src=sgpr(2))


@pytest.mark.unit
class TestShiftVectorComponentsVALU:
    """Tests for ShiftVectorComponentsVALU class"""

    def create_mock_writer(self):
        """Create a mock writer with all necessary attributes"""
        writer = Mock()
        writer.states = Mock()
        writer.states.kernel = {"WavefrontSize": 64}
        writer.states.laneSGPRCount = 2
        writer.states.bpeCinternal = 4
        writer.states.bpr = 4
        writer.states.c = Mock()
        writer.states.c.startVgprValu = 100

        # Mock vgpr pool
        writer.vgprPool = Mock()
        writer.vgprPool.checkOut = Mock(side_effect=lambda count=1, name="": 10)
        writer.vgprPool.checkIn = Mock()

        # Mock sgpr pool context manager
        tmpSgprInfo = Mock()
        tmpSgprInfo.idx = 5
        tmpSgprInfo.__enter__ = Mock(return_value=tmpSgprInfo)
        tmpSgprInfo.__exit__ = Mock(return_value=False)
        writer.allocTmpSgpr = Mock(return_value=tmpSgprInfo)

        # Mock labels
        writer.labels = Mock()
        writer.labels.getName = Mock(return_value="test_label")

        return writer

    def create_basic_kernel(self):
        """Create basic kernel configuration"""
        return {
            "WavefrontSize": 64,
            "VectorWidthA": 2,
            "VectorWidthB": 2,
            "ThreadTile0": 4,
            "ThreadTile1": 4,
            "SubGroup0": 16,
            "SubGroup1": 16,
            "MacroTile0": 64,
            "MacroTile1": 64,
            8: 16,  # For kernel[tP["tt"]]
        }

    def create_tP_A(self):
        """Create tensor parameters for A"""
        return {
            "idx": 0,
            "tensorChar": "A",
            "glvw": 2,
            "tt": 8,
            "wg": "WorkGroup0",
            "mt": "MacroTile0",
            "sg": "SubGroup0",
            "isA": True,
            "isB": False,
        }

    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticRemainder', side_effect=mock_vectorStaticRemainder_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticDivide', side_effect=mock_vectorStaticDivide_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticMultiply', side_effect=mock_vectorStaticMultiply_returns_real_item)
    def test_valu_basic_execution_tensor_a(self, mock_mult, mock_div, mock_rem):
        """Test VALU executes for tensor A"""
        shifter = ShiftVectorComponentsVALU()
        writer = self.create_mock_writer()
        kernel = self.create_basic_kernel()
        tP = self.create_tP_A()

        result = shifter(writer, kernel, tP)

        # Verify vgpr allocations happened
        assert writer.vgprPool.checkOut.call_count > 0
        assert writer.vgprPool.checkIn.call_count > 0
        # Verify sgpr allocation happened
        assert writer.allocTmpSgpr.called
        # Verify result is a Module
        assert result is not None

    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticRemainder', side_effect=mock_vectorStaticRemainder_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticDivide', side_effect=mock_vectorStaticDivide_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticMultiply', side_effect=mock_vectorStaticMultiply_returns_real_item)
    def test_valu_glvw_1(self, mock_mult, mock_div, mock_rem):
        """Test VALU with glvw=1"""
        shifter = ShiftVectorComponentsVALU()
        writer = self.create_mock_writer()
        kernel = self.create_basic_kernel()
        tP = self.create_tP_A()
        tP["glvw"] = 1

        result = shifter(writer, kernel, tP)
        assert result is not None

    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticRemainder', side_effect=mock_vectorStaticRemainder_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticDivide', side_effect=mock_vectorStaticDivide_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticMultiply', side_effect=mock_vectorStaticMultiply_returns_real_item)
    def test_valu_glvw_4(self, mock_mult, mock_div, mock_rem):
        """Test VALU with glvw=4"""
        shifter = ShiftVectorComponentsVALU()
        writer = self.create_mock_writer()
        kernel = self.create_basic_kernel()
        kernel["VectorWidthA"] = 4
        tP = self.create_tP_A()
        tP["glvw"] = 4

        result = shifter(writer, kernel, tP)
        assert result is not None

    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticRemainder', side_effect=mock_vectorStaticRemainder_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticDivide', side_effect=mock_vectorStaticDivide_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticMultiply', side_effect=mock_vectorStaticMultiply_returns_real_item)
    def test_valu_tensor_b(self, mock_mult, mock_div, mock_rem):
        """Test VALU for tensor B"""
        shifter = ShiftVectorComponentsVALU()
        writer = self.create_mock_writer()
        kernel = self.create_basic_kernel()
        tP = {
            "idx": 1,
            "tensorChar": "B",
            "glvw": 2,
            "tt": 8,
            "wg": "WorkGroup1",
            "mt": "MacroTile1",
            "sg": "SubGroup1",
            "isA": False,
            "isB": True,
        }

        result = shifter(writer, kernel, tP)
        assert result is not None


@pytest.mark.unit
class TestShiftVectorComponentsMFMA:
    """Tests for ShiftVectorComponentsMFMA class"""

    def create_mock_writer(self):
        """Create a mock writer with MFMA support"""
        writer = Mock()
        writer.states = Mock()
        writer.states.kernel = {"WavefrontSize": 64}
        writer.states.laneSGPRCount = 2
        writer.states.bpeCinternal = 4
        writer.states.bpr = 4

        # Mock vgpr pool
        writer.vgprPool = Mock()
        writer.vgprPool.checkOut = Mock(side_effect=lambda count=1, name="": 10)
        writer.vgprPool.checkOutAligned = Mock(side_effect=lambda count, align: 10)
        writer.vgprPool.checkIn = Mock()

        # Mock sgpr pool
        tmpSgprInfo = Mock()
        tmpSgprInfo.idx = 5
        tmpSgprInfo.__enter__ = Mock(return_value=tmpSgprInfo)
        tmpSgprInfo.__exit__ = Mock(return_value=False)
        writer.allocTmpSgpr = Mock(return_value=tmpSgprInfo)

        # Mock labels
        writer.labels = Mock()
        writer.labels.getName = Mock(return_value="test_label")

        # Mock acc functions - return real rocisa Items
        writer.accVgprReadWriteIndex = Mock(return_value="acc[0]")
        # accVgprReadWriteFunction should return a real Item, not a Mock
        def mock_acc_function(*args, **kwargs):
            return SMovB32(dst=vgpr(0), src=sgpr(0))
        writer.accVgprReadWriteFunction = Mock(return_value=mock_acc_function)
        writer.updateBranchPlaceHolder = Mock()

        return writer

    def create_mfma_kernel(self):
        """Create MFMA kernel configuration"""
        return {
            "WavefrontSize": 64,
            "VectorWidthA": 1,
            "VectorWidthB": 1,
            "MatrixInstM": 16,
            "MatrixInstN": 16,
            "MatrixInstBM": 1,
            "MatrixInstBN": 1,
            "MIWaveGroup": [1, 1],
            "MIWaveTile": [16, 16],
            "MIOutputVectorWidth": 4,
            "SourceSwap": False,
            "MIRegPerOut": 1,
            "MIArchVgpr": True,
            "MacroTile0": 64,
            "MacroTile1": 64,
            16: 64,  # For kernel[tP["tt"]]
            "ProblemType": {
                "MacDataTypeA": SimpleNamespace(isComplex=lambda: False),
                "MacDataTypeB": SimpleNamespace(isComplex=lambda: False),
            }
        }

    def create_tP_A_mfma(self):
        """Create MFMA tensor parameters for A"""
        return {
            "idx": 0,
            "tensorChar": "A",
            "glvw": 2,
            "tt": 16,
            "wg": "WorkGroup0",
            "mt": "MacroTile0",
            "sg": "SubGroup0",
            "isA": True,
            "isB": False,
        }

    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticRemainder', side_effect=mock_vectorStaticRemainder_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticDivide', side_effect=mock_vectorStaticDivide_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticMultiply', side_effect=mock_vectorStaticMultiply_returns_real_item)
    def test_mfma_partial_thread_path(self, mock_mult, mock_div, mock_rem):
        """Test MFMA partial thread path"""
        shifter = ShiftVectorComponentsMFMA()
        writer = self.create_mock_writer()
        kernel = self.create_mfma_kernel()
        tP = self.create_tP_A_mfma()
        tP["glvw"] = 2  # Small glvw for partial thread

        result = shifter(writer, kernel, tP)

        # Verify vgpr allocations
        assert writer.vgprPool.checkOut.call_count > 0
        assert result is not None

    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticRemainder', side_effect=mock_vectorStaticRemainder_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticDivide', side_effect=mock_vectorStaticDivide_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticMultiply', side_effect=mock_vectorStaticMultiply_returns_real_item)
    def test_mfma_all_thread_path(self, mock_mult, mock_div, mock_rem):
        """Test MFMA all thread path"""
        shifter = ShiftVectorComponentsMFMA()
        writer = self.create_mock_writer()
        kernel = self.create_mfma_kernel()
        tP = self.create_tP_A_mfma()
        tP["glvw"] = 128  # Large glvw for all thread

        result = shifter(writer, kernel, tP)
        assert result is not None

    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticRemainder', side_effect=mock_vectorStaticRemainder_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticDivide', side_effect=mock_vectorStaticDivide_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticMultiply', side_effect=mock_vectorStaticMultiply_returns_real_item)
    def test_mfma_tensor_b(self, mock_mult, mock_div, mock_rem):
        """Test MFMA for tensor B"""
        shifter = ShiftVectorComponentsMFMA()
        writer = self.create_mock_writer()
        kernel = self.create_mfma_kernel()
        tP = {
            "idx": 1,
            "tensorChar": "B",
            "glvw": 2,
            "tt": 16,
            "wg": "WorkGroup1",
            "mt": "MacroTile1",
            "sg": "SubGroup1",
            "isA": False,
            "isB": True,
        }

        result = shifter(writer, kernel, tP)
        assert result is not None

    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticRemainder', side_effect=mock_vectorStaticRemainder_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticDivide', side_effect=mock_vectorStaticDivide_returns_real_item)
    @patch('Tensile.Components.ShiftVectorComponents.vectorStaticMultiply', side_effect=mock_vectorStaticMultiply_returns_real_item)
    def test_mfma_source_swap(self, mock_mult, mock_div, mock_rem):
        """Test MFMA with source swap"""
        shifter = ShiftVectorComponentsMFMA()
        writer = self.create_mock_writer()
        kernel = self.create_mfma_kernel()
        kernel["SourceSwap"] = True
        tP = self.create_tP_A_mfma()

        result = shifter(writer, kernel, tP)
        assert result is not None
