# ####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ####################################################################################

cmake_policy(SET CMP0057 NEW)

find_package(Threads REQUIRED)
rocm_test_link_libraries(Threads::Threads migraphx migraphx_onnx migraphx_tf)
rocm_test_include_directories(include)

set(MIGRAPHX_DISABLE_LARGE_BUFFER_TESTS Off CACHE BOOL "")
if(MIGRAPHX_DISABLE_LARGE_BUFFER_TESTS)
    add_compile_definitions(MIGRAPHX_DISABLE_LARGE_BUFFER_TESTS)
endif()

add_library(register_targets STATIC register_target.cpp)
target_link_libraries(register_targets PRIVATE migraphx migraphx_all_targets)

file(GLOB TESTS CONFIGURE_DEPENDS *.cpp)
list(REMOVE_ITEM TESTS ${CMAKE_CURRENT_SOURCE_DIR}/register_target.cpp)

foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    rocm_add_test_executable(test_${BASE_NAME} ${TEST})
    rocm_clang_tidy_check(test_${BASE_NAME})
endforeach()

if(MIGRAPHX_ENABLE_GPU)
    # gpu tests
    file(GLOB GPU_TESTS CONFIGURE_DEPENDS gpu/*.cpp)

    foreach(TEST ${GPU_TESTS})
        get_filename_component(BASE_NAME ${TEST} NAME_WE)
        rocm_add_test_executable(test_gpu_${BASE_NAME} ${TEST})
        rocm_clang_tidy_check(test_gpu_${BASE_NAME})
        set_tests_properties(test_gpu_${BASE_NAME} PROPERTIES
            COST 10
            RESOURCE_LOCK gpu
        )
        if(MIGRAPHX_USE_HIPRTC)
        target_compile_definitions(test_gpu_${BASE_NAME} PUBLIC -DMIGRAPHX_USE_HIPRTC)
        endif()
        target_link_libraries(test_gpu_${BASE_NAME} migraphx_gpu migraphx_kernels register_targets)
    endforeach()
endif()

if(MIGRAPHX_ENABLE_FPGA)
    # fpga tests
    file(GLOB FPGA_TESTS CONFIGURE_DEPENDS fpga/*.cpp)

    foreach(TEST ${FPGA_TESTS})
        get_filename_component(BASE_NAME ${TEST} NAME_WE)
        rocm_add_test_executable(test_fpga_${BASE_NAME} ${TEST})
        rocm_clang_tidy_check(test_fpga_${BASE_NAME})
        set_tests_properties(test_fpga_${BASE_NAME} PROPERTIES
            COST 10
            RESOURCE_LOCK fpga
        )
        target_link_libraries(test_fpga_${BASE_NAME} migraphx_fpga)
    endforeach()
endif()

# Onnx test
set(TEST_ONNX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnx)
add_subdirectory(onnx)

# tf test
set(TEST_TF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tf)
add_subdirectory(tf)

add_subdirectory(api)
add_subdirectory(verify)
add_subdirectory(ref)

if(MIGRAPHX_ENABLE_PYTHON)
    add_subdirectory(py)
endif()

# multitarget test
if(MIGRAPHX_ENABLE_GPU AND MIGRAPHX_ENABLE_CPU AND MIGRAPHX_ENABLE_FPGA)
    set(TEST_MULTI_TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/multi_target)
    file(GLOB MULTI_TARGET_TESTS CONFIGURE_DEPENDS ${TEST_MULTI_TARGET_DIR}/*.cpp)

    foreach(MULTI_TARGET_TEST ${MULTI_TARGET_TESTS})
        get_filename_component(BASE_NAME ${MULTI_TARGET_TEST} NAME_WE)
        set(TEST_NAME test_${BASE_NAME})
        add_executable(${TEST_NAME} ${MULTI_TARGET_TEST})
        rocm_clang_tidy_check(${TEST_NAME})
        target_link_libraries(${TEST_NAME} migraphx migraphx_onnx migraphx_tf migraphx_all_targets register_targets)
        target_include_directories(${TEST_NAME} PUBLIC include)
        add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}> WORKING_DIRECTORY ${TEST_MULTI_TARGET_DIR})
        rocm_mark_as_test(${TEST_NAME})
    endforeach()
endif()


function(test_header NAME HEADER)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/header-main-include-${NAME}.cpp "
#include <${HEADER}>
int main() {}\n"
    )
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/header-static-include-${NAME}.cpp "
#include <${HEADER}>
#if defined(min) || defined(max) || defined(near) || defined(far)
#error \"Do not include windows.h in header files\"
#endif
\n"
    )
    rocm_add_test_executable(${NAME}
        ${CMAKE_CURRENT_BINARY_DIR}/header-main-include-${NAME}.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/header-static-include-${NAME}.cpp
    )
endfunction()

function(test_headers PREFIX)
    file(GLOB HEADERS CONFIGURE_DEPENDS ${ARGN})
    if(NOT MIGRAPHX_USE_COMPOSABLEKERNEL)
        list(REMOVE_ITEM HEADERS
              ${CMAKE_SOURCE_DIR}/src/targets/gpu/include/migraphx/gpu/ck.hpp)
    endif()
    list(REMOVE_ITEM HEADERS ${CMAKE_SOURCE_DIR}/src/include/migraphx/float8_impl.hpp)
    foreach(HEADER ${HEADERS})
        file(RELATIVE_PATH HEADER_REL ${CMAKE_SOURCE_DIR} ${HEADER})
        string(MAKE_C_IDENTIFIER ${HEADER_REL} TEST_NAME)
        get_filename_component(BASE_NAME ${HEADER} NAME_WE)
        test_header(header_${TEST_NAME} ${PREFIX}/${BASE_NAME}.hpp)
        target_link_libraries(header_${TEST_NAME} migraphx migraphx_onnx migraphx_tf migraphx_all_targets register_targets)
    endforeach()
endfunction()

test_headers(migraphx ${CMAKE_SOURCE_DIR}/src/include/migraphx/*.hpp)
test_headers(migraphx/ref ${CMAKE_SOURCE_DIR}/src/targets/ref/include/migraphx/ref/*.hpp)

if(MIGRAPHX_ENABLE_GPU)
    test_headers(migraphx/gpu HEADERS ${CMAKE_SOURCE_DIR}/src/targets/gpu/include/migraphx/gpu/*.hpp DEPENDS migraphx_gpu)
endif()
if(MIGRAPHX_ENABLE_CPU)
    test_headers(migraphx/cpu HEADERS ${CMAKE_SOURCE_DIR}/src/targets/cpu/include/migraphx/cpu/*.hpp migraphx_cpu)
endif()
if(MIGRAPHX_ENABLE_FPGA)
    test_headers(migraphx/fpga HEADERS ${CMAKE_SOURCE_DIR}/src/targets/fpga/include/migraphx/fpga/*.hpp migraphx_fpga)
endif()


