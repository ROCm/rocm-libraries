# MIT License
#
# Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

# Tests which check if there are no linkage problems when there
# are multiple sources (for example multiple definition errors).

# Add rocRAND linkage test
function(add_rocrand_link_test TEST_SOURCES)
    list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
    get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    # nvcc/CUDA
    if(HIP_COMPILER STREQUAL "nvcc")
        set_source_files_properties(${TEST_SOURCES}
          PROPERTIES
            LANGUAGE CUDA
            CUDA_STANDARD 14
        )
    endif()

    if(TARGET GTest::GTest)
        target_link_libraries(${TEST_TARGET}
            GTest::GTest
            GTest::Main
        )
    else()
        target_link_libraries(${TEST_TARGET}
            GTest::gtest
            GTest::gtest_main
        )
    endif()

    if(USE_SYSTEM_LIB)
        target_link_libraries(${TEST_TARGET} roc::rocrand)
    else()
        target_link_libraries(${TEST_TARGET} rocrand)
    endif()

    if(HIP_COMPILER STREQUAL "clang")
        target_link_libraries(${TEST_TARGET} hip::device)
    endif()
    set_target_properties(
        ${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
    add_relative_test(${TEST_TARGET} ${TEST_TARGET})
    rocm_install(TARGETS ${TEST_TARGET} COMPONENT tests)
endfunction()

add_rocrand_link_test("test_rocrand_linkage.cpp;get_rocrand_version.cpp")
