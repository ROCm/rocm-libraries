# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import pytest
import hipblaslt
c = hipblaslt._core

requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


@requires_gpu
def test_handle_create_and_close():
    h = c.Handle()
    assert h.ptr != 0
    h.close()


@requires_gpu
def test_handle_context_manager():
    with c.Handle() as h:
        assert h.ptr != 0
