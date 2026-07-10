# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


@requires_gpu
def test_desc_create_and_epilogue():
    d = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
    d.set_attribute_int(c.MatmulDescAttr.EPILOGUE, int(c.Epilogue.RELU))
    assert d.get_attribute_int(c.MatmulDescAttr.EPILOGUE) == int(c.Epilogue.RELU)


@requires_gpu
def test_desc_scale_mode():
    d = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
    d.set_attribute_int(c.MatmulDescAttr.A_SCALE_MODE, int(c.ScaleMode.VEC32_UE8M0))


@requires_gpu
def test_preference_workspace():
    p = c.Preference()
    p.set_max_workspace(32 * 1024 * 1024)
    assert p.ptr != 0
