###############################################################################
 #
 # MIT License
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
 ###############################################################################

# Compile-time checks that must fail with a specific diagnostic.
#
# Each source becomes an object library that is not part of the default build.
# Its test builds it and passes only if the build output contains the expected
# message. These tests run from the build tree only and are not installed.

function(add_rocwmma_compile_fail_test TEST_NAME SOURCE EXPECTED_REGEX)
    add_library(${TEST_NAME} OBJECT EXCLUDE_FROM_ALL ${SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE rocwmma)
    target_include_directories(${TEST_NAME} PRIVATE ${ROCWMMA_TEST_INCLUDE_DIRS})
    add_test(NAME ${TEST_NAME}
             COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${TEST_NAME} --config $<CONFIG>)
    set_tests_properties(${TEST_NAME} PROPERTIES PASS_REGULAR_EXPRESSION "${EXPECTED_REGEX}")
endfunction()

set(ROCWMMA_AMBIGUOUS_WMMA_OPS_MESSAGE "rocWMMA: the WMMA capability list of this target has two operations")
add_rocwmma_compile_fail_test(wmma_ops_duplicate_fails
                              ${CMAKE_CURRENT_SOURCE_DIR}/wmma_ops_duplicate.cpp
                              "${ROCWMMA_AMBIGUOUS_WMMA_OPS_MESSAGE}")
add_rocwmma_compile_fail_test(wmma_ops_overlap_fails
                              ${CMAKE_CURRENT_SOURCE_DIR}/wmma_ops_overlap.cpp
                              "${ROCWMMA_AMBIGUOUS_WMMA_OPS_MESSAGE}")

# A missing builtin is diagnosed in the device pass of a target that uses it.
if(GPU_TARGETS MATCHES "gfx11|gfx12")
    add_rocwmma_compile_fail_test(wmma_builtin_unavailable_fails
                                  ${CMAKE_CURRENT_SOURCE_DIR}/wmma_builtin_unavailable.cpp
                                  "rocWMMA: the compiler does not provide __builtin_amdgcn_wmma_f32_16x16x")
endif()
