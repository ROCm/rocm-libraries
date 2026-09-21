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
from Tensile.SolutionStructs.Utilities import getMiInputType


def _build_kernel(*, enable_f32_xdl_math_op=False, use_f32x_emulation=False,
                  data_type=None, f32_xdl_math_op=None):
    """Build a minimal kernel dict for getMiInputType tests."""
    return {
        "EnableF32XdlMathOp": enable_f32_xdl_math_op,
        "UseF32XEmulation": use_f32x_emulation,
        "ProblemType": {
            "DataType": data_type or DataType(DataTypeEnum.Float),
            "F32XdlMathOp": f32_xdl_math_op or DataType(DataTypeEnum.XFloat32),
        },
    }


class TestGetMiInputType:
    """Verify getMiInputType selects the correct MFMA operand type.

    EnableF32XdlMathOp=False                         → ProblemType["DataType"]
    EnableF32XdlMathOp=True,  UseF32XEmulation=False → ProblemType["F32XdlMathOp"]
    EnableF32XdlMathOp=True,  UseF32XEmulation=True  → DataType(BFloat16)
    """

    def test_plain_datatype(self):
        """Neither flag set → returns ProblemType.DataType."""
        kernel = _build_kernel()

        result = getMiInputType(kernel)

        assert result == DataType(DataTypeEnum.Float)

    def test_native_xf32(self):
        """EnableF32XdlMathOp without emulation → returns F32XdlMathOp."""
        kernel = _build_kernel(enable_f32_xdl_math_op=True)

        result = getMiInputType(kernel)

        assert result == DataType(DataTypeEnum.XFloat32)

    def test_use_f32x_emulation(self):
        """UseF32XEmulation + EnableF32XdlMathOp → returns DataType(BFloat16)."""
        kernel = _build_kernel(
            enable_f32_xdl_math_op=True,
            use_f32x_emulation=True,
        )

        result = getMiInputType(kernel)

        assert result == DataType(DataTypeEnum.BFloat16)

    def test_missing_enable_flag_raises(self):
        """EnableF32XdlMathOp must exist; absent key means broken caller."""
        kernel = _build_kernel()
        del kernel["EnableF32XdlMathOp"]

        with pytest.raises(KeyError):
            getMiInputType(kernel)

    def test_missing_emulation_flag_raises(self):
        """UseF32XEmulation must exist; absent key means broken caller."""
        kernel = _build_kernel(enable_f32_xdl_math_op=True)
        del kernel["UseF32XEmulation"]

        with pytest.raises(KeyError):
            getMiInputType(kernel)
