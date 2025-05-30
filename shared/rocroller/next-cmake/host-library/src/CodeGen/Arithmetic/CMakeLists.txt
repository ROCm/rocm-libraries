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


set(_CMAKE_CURRENT_SOURCE_DIR "${ROCROLLER_LIB_DIR}/source/CodeGen/Arithmetic")
target_sources(rocroller
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/Add.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddShiftL.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ArithmeticGenerator.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ArithmeticShiftR.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/BitFieldExtract.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/BitwiseAnd.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/BitwiseNegate.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/BitwiseOr.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/BitwiseXor.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Conditional.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Convert.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Divide.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Equal.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Exponential2.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/GreaterThan.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/GreaterThanEqual.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LessThan.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LessThanEqual.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LogicalAnd.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LogicalNot.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LogicalOr.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LogicalShiftR.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MatrixMultiply.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Modulo.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MultiplyAdd.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Multiply.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MultiplyHigh.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Negate.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/NotEqual.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/RandomNumber.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ScaledMatrixMultiply.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ShiftLAdd.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ShiftL.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Subtract.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Utility.cpp"
)
