# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.25.0)
project(clients LANGUAGES CXX)

set(TEMP_CLIENTS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../clients")
set(TEMP_TENSILE_CLIENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite/Tensile/Source/client")
set(TEMP_TENSILELITE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite")

add_executable(hipblaslt-bench)
add_library(hipblaslt-clients-common STATIC)

target_include_directories(hipblaslt-clients-common
    PUBLIC
        $<BUILD_INTERFACE:${TEMP_CLIENTS_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${TEMP_CLIENTS_SOURCE_DIR}/common>
        $<BUILD_INTERFACE:${TEMP_TENSILE_CLIENT_SOURCE_DIR}/include>
    PRIVATE
        ${TEMP_TENSILELITE_DIR}
)

if(HIPBLASLT_ENABLE_BLIS)
    target_link_libraries(hipblaslt-clients-common PRIVATE BLIS::BLIS)
endif()

target_link_libraries(hipblaslt-clients-common
    PUBLIC
        roc::hipblaslt
        ${LAPACK_LIBRARIES} 
        ${BLAS_LIBRARIES}
        ${CBLAS_LIBRARIES}
    PRIVATE
        hip::device
)
target_compile_definitions(hipblaslt-clients-common PRIVATE CLIENTS_NO_FORTRAN) # necessary?

if(HIPBLASLT_ENABLE_ASAN)
    hipblaslt_target_configure_sanitizers(hipblaslt-clients-common PUBLIC)
endif()

target_link_libraries(hipblaslt-bench
    PRIVATE
        hip::device
        hipblaslt-clients-common
)

if(HIPBLASLT_ENABLE_OPENMP)
    target_link_libraries(hipblaslt-bench PRIVATE OpenMP::OpenMP_CXX)
endif()

add_subdirectory(common)
add_subdirectory(bench)

add_executable(hipblaslt-bench-groupedgemm-fixed-mk "${TEMP_CLIENTS_SOURCE_DIR}/benchmarks/client_groupedgemm_fixed_mk.cpp")
target_link_libraries(hipblaslt-bench-groupedgemm-fixed-mk PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-groupedgemm-fixed-mk PRIVATE ROCBLASLT_INTERNAL_API )

add_executable(hipblaslt-bench-extop-layernorm "${TEMP_CLIENTS_SOURCE_DIR}/benchmarks/client_extop_layernorm.cpp")
target_link_libraries(hipblaslt-bench-extop-layernorm PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-layernorm PRIVATE ROCBLASLT_INTERNAL_API)

add_executable(hipblaslt-bench-extop-matrixtransform "${TEMP_CLIENTS_SOURCE_DIR}/benchmarks/client_extop_matrixtransform.cpp")
target_link_libraries(hipblaslt-bench-extop-matrixtransform PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-matrixtransform PRIVATE ROCBLASLT_INTERNAL_API )

add_executable(hipblaslt-bench-extop-softmax "${TEMP_CLIENTS_SOURCE_DIR}/benchmarks/client_extop_softmax.cpp")
target_link_libraries(hipblaslt-bench-extop-softmax PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-softmax PRIVATE ROCBLASLT_INTERNAL_API )

add_executable(hipblaslt-bench-extop-amax "${TEMP_CLIENTS_SOURCE_DIR}/benchmarks/client_extop_amax.cpp")
target_link_libraries(hipblaslt-bench-extop-amax PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-amax PRIVATE ROCBLASLT_INTERNAL_API )

find_package(LLVM REQUIRED)
add_executable(hipblaslt-sequence "${TEMP_CLIENTS_SOURCE_DIR}/benchmarks/client_sequence.cpp")
target_link_libraries(hipblaslt-sequence PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-sequence PRIVATE ROCBLASLT_INTERNAL_API)
target_link_libraries(hipblaslt-sequence PRIVATE LLVMObjectYAML)
target_include_directories(hipblaslt-sequence PRIVATE ${LLVM_INCLUDE_DIRS})

if(BUILD_TESTING OR HIPBLASLT_BUILD_TESTING)
    find_package(GTest REQUIRED)
    add_executable(hipblaslt-test)
    if(rocm_smi_FOUND)
        target_link_libraries(hipblaslt-test PRIVATE rocm_smi64)
    endif()
    target_link_libraries(hipblaslt-test 
        PRIVATE
            hip::device
            hipblaslt-clients-common
            ${LAPACK_LIBRARIES}
            ${BLAS_LIBRARIES}
            GTest::gtest
    )
    if(HIPBLASLT_ENABLE_BLIS)
        target_link_libraries(hipblaslt-test PRIVATE BLIS::BLIS) # necessary?
    endif()
    target_compile_definitions(hipblaslt-test PRIVATE GOOGLE_TEST HIPBLASLT_TEST)
    add_subdirectory(test)
endif()

if(HIPBLASLT_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()
