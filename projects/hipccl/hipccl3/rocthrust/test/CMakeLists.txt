# MIT License
#
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

function(add_hipthrust_test TEST_NAME TEST_SOURCES)
    list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
    get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_include_directories(${TEST_TARGET} SYSTEM BEFORE
        PUBLIC
            ${GTEST_INCLUDE_DIRS}
    )
    target_link_libraries(${TEST_TARGET}
        PRIVATE
            hipthrust
            roc::rocprim_hip
            ${GTEST_BOTH_LIBRARIES}
    )
    foreach(amdgpu_target ${AMDGPU_TARGETS})
        target_link_libraries(${TEST_TARGET}
            PRIVATE
                --amdgpu-target=${amdgpu_target}
        )
    endforeach()
    # Set HIP as device system for Thrust
    target_compile_definitions(${TEST_TARGET}
        PRIVATE
            THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP
    )
    set_target_properties(${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/"
    )
    add_test(${TEST_NAME} ${TEST_TARGET})
endfunction()

# ****************************************************************************
# Tests
# ****************************************************************************

add_hipthrust_test("thrust.hip.copy" test_copy.cpp)
add_hipthrust_test("thrust.hip.for_each" test_for_each.cpp)
add_hipthrust_test("thrust.hip.memory" test_memory.cpp)
add_hipthrust_test("thrust.hip.transform" test_transform.cpp)
add_hipthrust_test("thrust.hip.sort" test_sort.cpp)
add_hipthrust_test("thrust.hip.device_ptr" test_device_ptr.cpp)
