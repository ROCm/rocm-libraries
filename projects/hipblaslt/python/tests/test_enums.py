# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import hipblaslt
c = hipblaslt._core


def test_datatype_has_r_32f():
    assert hasattr(c.DataType, "R_32F")


def test_epilogue_bias_value():
    # HIPBLASLT_EPILOGUE_BIAS == 4
    assert int(c.Epilogue.BIAS) == 4


def test_scalemode_vec32_ue8m0_value():
    # HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 == 2
    assert int(c.ScaleMode.VEC32_UE8M0) == 2


def test_enum_members_roundtrip():
    members = c.enum_members("Epilogue")
    assert members["BIAS"] == 4
    assert members["DEFAULT"] == 1
