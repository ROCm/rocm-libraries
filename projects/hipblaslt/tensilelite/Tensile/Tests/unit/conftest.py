"""Shared pytest fixtures for unit tests"""
import pytest
from unittest.mock import Mock
from types import SimpleNamespace
from Tensile.Common.DataType import DataType


class SgprAllocator:
    """Helper to allocate sequential sgpr register numbers for test setup.

    Usage:
        sgprs = SgprAllocator()
        sgprIndex = sgprs.alloc()  # 5
        sgprGridY = sgprs.alloc()  # 6
        sgprGridX = sgprs.alloc()  # 7
    """
    def __init__(self, start=5):
        self._current = start

    def alloc(self):
        """Allocate and return next sequential sgpr number"""
        val = self._current
        self._current += 1
        return val

    def alloc_many(self, count):
        """Allocate multiple sequential sgprs and return as list"""
        return [self.alloc() for _ in range(count)]


@pytest.fixture
def basic_kernel():
    """Basic kernel configuration for WorkGroupMappingAlgos tests"""
    return {"WavefrontSize": 64}


@pytest.fixture
def sgpr_alloc():
    """Fixture that provides an SgprAllocator for sequential sgpr allocation"""
    return SgprAllocator(start=5)


@pytest.fixture
def basic_state():
    """Create a basic state configuration shared across all KernelWriterBetaOnly tests"""
    return {
        "ProblemType": {
            "ComputeDataType": DataType('s'),
            "DestDataType": DataType('s'),
            "DataType": DataType('s'),  # Needed for some tests
            "Index0": 0,
            "Index1": 1,
            "NumIndicesC": 2,
            "StridedBatched": True,
            "GroupedGemm": False,
            "BetaOnlyUseBias": False,
            "UseInitialStridesCD": False,
            "HighPrecisionAccumulate": False,  # Needed for some tests
        },
        "_GlobalAccumulation": False,
    }


def create_mock_writer(add_alloc_tmp_sgpr=False, add_alloc_tmp_sgpr_list=False,
                       add_labels=False, label_prefix="TestPrefix", label_name_inc=None,
                       add_full_writer=False):
    """Create a mock writer object with configurable components.

    This is a shared helper to avoid duplicating mock writer creation across test classes.

    Args:
        add_alloc_tmp_sgpr: Add allocTmpSgpr context manager mock
        add_alloc_tmp_sgpr_list: Add allocTmpSgprList context manager mock
        add_labels: Add labels mock with getUniqueNamePrefix and getNameInc
        label_prefix: Return value for getUniqueNamePrefix (default: "TestPrefix")
        label_name_inc: Return value for getNameInc, can be a callable (default: "TestLabel")
        add_full_writer: Add full writer capabilities (sgprs dict, states, addSgprVarToPool, etc.)

    Returns:
        Mock writer object with requested components
    """
    writer = Mock()

    # Base pools - always present
    writer.sgprPool = Mock()
    writer.sgprPool.checkOut = Mock(return_value=10)
    writer.sgprPool.checkOutAligned = Mock(return_value=10)
    writer.sgprPool.checkIn = Mock()
    writer.vgprPool = Mock()
    writer.vgprPool.checkOut = Mock(return_value=20)
    writer.vgprPool.checkOutAligned = Mock(return_value=20)
    writer.vgprPool.checkIn = Mock()

    # Optional: allocTmpSgpr context manager
    if add_alloc_tmp_sgpr:
        writer.allocTmpSgpr = Mock()
        writer.allocTmpSgpr.return_value.__enter__ = Mock(return_value=SimpleNamespace(idx=5))
        writer.allocTmpSgpr.return_value.__exit__ = Mock(return_value=False)

    # Optional: allocTmpSgprList context manager
    if add_alloc_tmp_sgpr_list:
        writer.allocTmpSgprList = Mock()
        writer.allocTmpSgprList.return_value.__enter__ = Mock(return_value=[
            SimpleNamespace(idx=5),
            SimpleNamespace(idx=7),
            SimpleNamespace(idx=8),
        ])
        writer.allocTmpSgprList.return_value.__exit__ = Mock(return_value=False)

    # Optional: labels
    if add_labels:
        writer.labels = Mock()
        writer.labels.getUniqueNamePrefix = Mock(return_value=label_prefix)
        if label_name_inc is None:
            writer.labels.getNameInc = Mock(return_value="TestLabel")
        else:
            writer.labels.getNameInc = Mock(side_effect=label_name_inc)

    # Optional: full writer capabilities for complex tests
    if add_full_writer:
        writer.addSgprVarToPool = Mock()
        writer.removeSgprVarFromPool = Mock()
        writer.sgprs = {
            "NumWorkGroups0": 1,
            "NumWorkGroups1": 2,
            "Alpha": 3,
            "Beta": 4,
            "WorkGroup2": 5,
            "StreamKLocalStart": 6,
            "StreamKIterEnd": 7,
            "LoopCounterL": 8,
            "OrigLoopCounterL": 9,
            "SizesSum": 10
        }
        writer.states = Mock()
        writer.states.WGMTransformLevels = 1

    return writer
