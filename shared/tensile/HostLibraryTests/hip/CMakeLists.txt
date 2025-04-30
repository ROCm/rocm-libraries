################################################################################
#
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

get_filename_component(COMPILER ${CMAKE_CXX_COMPILER} NAME)
set(CODE_OBJECT_VERSION "default")

TensileCreateLibraryFiles(
    "${CMAKE_CURRENT_SOURCE_DIR}/../configs/lite_configs"
    "${TEST_DATA_DIR}/test_kernels_lite"
    TENSILE_ROOT "${TENSILE_SCRIPT_ROOT}"
    EMBED_LIBRARY test_kernels_lite
    EMBED_KEY     kernels_lite
    VAR_PREFIX LITE
    NO_MERGE_FILES
    COMPILER ${COMPILER}
    COMPILER_PATH ${CMAKE_CXX_COMPILER}
    CODE_OBJECT_VERSION ${CODE_OBJECT_VERSION}
    )

TensileCreateLibraryFiles(
    "${CMAKE_CURRENT_SOURCE_DIR}/../configs/lite_configs_mixed"
    "${TEST_DATA_DIR}/test_kernels_lite_mixed"
    TENSILE_ROOT "${TENSILE_SCRIPT_ROOT}"
    EMBED_LIBRARY test_kernels_lite_mixed
    EMBED_KEY     kernels_lite_mixed
    VAR_PREFIX LITE_MIXED
    NO_MERGE_FILES
    COMPILER ${COMPILER}
    COMPILER_PATH ${CMAKE_CXX_COMPILER}
    CODE_OBJECT_VERSION ${CODE_OBJECT_VERSION}
    )

TensileCreateLibraryFiles(
    "${CMAKE_CURRENT_SOURCE_DIR}/../configs/lite_configs"
    "${TEST_DATA_DIR}/test_kernels_lite_2"
    TENSILE_ROOT "${TENSILE_SCRIPT_ROOT}"
    EMBED_LIBRARY test_kernels_lite_2
    EMBED_KEY     kernels_lite_2
    VAR_PREFIX LITE_2
    NO_MERGE_FILES
    COMPILER ${COMPILER}
    COMPILER_PATH ${CMAKE_CXX_COMPILER}
    CODE_OBJECT_VERSION ${CODE_OBJECT_VERSION}
    )

TensileCreateLibraryFiles(
    "${CMAKE_CURRENT_SOURCE_DIR}/../configs/tile_aware_selection"
    "${TEST_DATA_DIR}/test_tile_aware_selection"
    TENSILE_ROOT "${TENSILE_SCRIPT_ROOT}"
    EMBED_LIBRARY test_tile_aware_selection
    EMBED_KEY     tile_aware_selection
    VAR_PREFIX TILE_AWARE
    NO_MERGE_FILES
    COMPILER ${COMPILER}
    COMPILER_PATH ${CMAKE_CXX_COMPILER}
    CODE_OBJECT_VERSION ${CODE_OBJECT_VERSION}
    )

set(test_yaml rocblas_sgemm_asm_single_kernel.yaml)
file(COPY ${test_yaml} DESTINATION .)
add_custom_target(test_code_object
    COMMAND python ${TENSILE_ROOT}/Tensile/Tensile.py ${test_yaml} code_object
    DEPENDS ${test_yaml})

set(test_sources ${test_sources}
    hip/HipSolutionAdapter_test.cpp
    hip/CopyTensor_test.cpp
    hip/RunGEMMKernel_test.cpp PARENT_SCOPE)

set(HIP_TEST_LIBRARIES
    hip::host
    test_kernels_lite
    test_kernels_lite_2
    test_kernels_lite_mixed
    test_tile_aware_selection
    TensileClient PARENT_SCOPE)
