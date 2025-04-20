# ########################################################################
# Copyright (C) 2022 Advanced Micro Devices, Inc.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# Need to support Release/Debug

string(REGEX MATCHALL "gfx[a-z0-9]+" archs "${Tensile_ARCHITECTURE}")
list(REMOVE_DUPLICATES archs)
list(JOIN archs "," archs)

set(output_library "${PROJECT_BINARY_DIR}/Tensile/library/hipblasltTransform.hsaco")
if(WIN32)
    cmake_path(CONVERT "${output_library}" TO_NATIVE_PATH_LIST output_library)
endif()

add_custom_command(
    COMMAND ${CMAKE_CXX_COMPILER} -x hip ${CMAKE_CURRENT_SOURCE_DIR}/matrix_transform.cpp --offload-arch="${archs}" -c --offload-device-only -Xoffload-linker --build-id=${Tensile_BUILD_ID} -O3 -o ${output_library}
    OUTPUT ${PROJECT_BINARY_DIR}/Tensile/library/hipblasltTransform.hsaco
    COMMENT "Compiling matrix transform kernels"
)

add_custom_target(MatrixTransformKernels ALL
    DEPENDS ${PROJECT_BINARY_DIR}/Tensile/library/hipblasltTransform.hsaco
)

add_dependencies(MatrixTransformKernels TENSILE_LIBRARY_TARGET)

unset(output_path)
