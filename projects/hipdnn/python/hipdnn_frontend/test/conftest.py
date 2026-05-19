# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Shared pytest fixtures for hipDNN Python binding tests."""

import pytest

import hipdnn_frontend as hipdnn


@pytest.fixture()
def handle():
    """Create a hipDNN handle for GPU operations."""
    return hipdnn.create_handle()


@pytest.fixture()
def graph():
    """Create a hipDNN Graph configured with FLOAT data types."""
    g = hipdnn.Graph()
    g.set_io_data_type(hipdnn.DataType.FLOAT)
    g.set_intermediate_data_type(hipdnn.DataType.FLOAT)
    g.set_compute_data_type(hipdnn.DataType.FLOAT)
    return g
