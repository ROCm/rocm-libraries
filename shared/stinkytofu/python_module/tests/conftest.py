"""
Shared pytest fixtures and configuration for StinkyTofu tests.
"""

import sys
import os
import pytest

# Add the build directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python_module'))

from stinkytofu import StinkyAsmIR, vgpr, sgpr, acc


@pytest.fixture
def gfx942_builder():
    """Fixture providing a StinkyAsmIR builder for gfx942 architecture (MI300A/MI300X)."""
    return StinkyAsmIR([9, 4, 2])


@pytest.fixture
def gfx950_builder():
    """Fixture providing a StinkyAsmIR builder for gfx950 architecture."""
    return StinkyAsmIR([9, 5, 0])


@pytest.fixture
def gfx1250_builder():
    """Fixture providing a StinkyAsmIR builder for gfx1250 architecture."""
    return StinkyAsmIR([12, 5, 0])


@pytest.fixture(params=[[9, 4, 2], [9, 5, 0], [12, 5, 0]])
def any_builder(request):
    """Parametrized fixture that tests across multiple architectures."""
    return StinkyAsmIR(request.param)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "architecture: tests for specific GPU architectures"
    )
    config.addinivalue_line(
        "markers", "composite: tests for composite/lowered instructions"
    )
    config.addinivalue_line(
        "markers", "mfma: tests for MFMA/WMMA instructions"
    )
    config.addinivalue_line(
        "markers", "sparse: tests for sparse matrix instructions"
    )

