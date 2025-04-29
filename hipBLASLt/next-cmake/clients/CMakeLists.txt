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

target_link_libraries(hipblaslt-clients-common PUBLIC
    roc::hipblaslt
    ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
)
target_compile_definitions(hipblaslt-clients-common PRIVATE CLIENTS_NO_FORTRAN) # necessary?

target_link_libraries(hipblaslt-bench PRIVATE hipblaslt-clients-common)

# TODO: Still haven't added hipblaslt-bench-groupedgemm-fixed-mk and related targets

add_subdirectory(common)
add_subdirectory(bench)

if(BUILD_TESTING OR HIPBLSALT_BUILD_TESTING)
    find_package(GTest REQUIRED)
    add_executable(hipblaslt-test)
    if(rocm_smi_FOUND)
        target_link_libraries(hipblaslt-test PRIVATE rocm_smi64)
    endif()
    target_link_libraries(hipblaslt-test PRIVATE hipblaslt-clients-common
        ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
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
