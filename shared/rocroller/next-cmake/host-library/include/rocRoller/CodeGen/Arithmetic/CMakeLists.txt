################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the Software), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################


set(_CMAKE_CURRENT_SOURCE_DIR "${ROCROLLER_LIB_DIR}/include/rocRoller/CodeGen/Arithmetic")
target_sources(rocroller
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/Add.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddShiftL.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ArithmeticGenerator.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ArithmeticShiftR.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/BitFieldExtract.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/BitwiseAnd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/BitwiseNegate.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/BitwiseOr.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/BitwiseXor.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Conditional.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Convert.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Divide.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Equal.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Exponential2.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/GreaterThanEqual.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/GreaterThan.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LessThanEqual.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LessThan.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LogicalAnd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LogicalNot.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LogicalOr.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LogicalShiftR.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MatrixMultiply_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MatrixMultiply.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Modulo.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MultiplyAdd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MultiplyHigh.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Multiply.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Negate.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/NotEqual.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/RandomNumber.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ScaledMatrixMultiply_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ScaledMatrixMultiply.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ShiftLAdd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ShiftL.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Subtract.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Utility.hpp"
)
