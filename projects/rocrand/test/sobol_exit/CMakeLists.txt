# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Process-exit coverage for Sobol constant teardown after hipDeviceReset().
# This is intentionally not a GoogleTest binary: death tests _exit() the child
# and skip the function-local static destructor this test exists to run.
# hipDeviceReset() is process-wide, so each case is its own CTest process.

if(HIP_COMPILER STREQUAL "nvcc")
    return()
endif()

add_executable(test_rocrand_sobol_exit test_rocrand_sobol_exit.cpp)

if(USE_SYSTEM_LIB)
    target_link_libraries(test_rocrand_sobol_exit roc::rocrand)
else()
    target_link_libraries(test_rocrand_sobol_exit rocrand)
endif()

if(HIP_COMPILER STREQUAL "clang")
    target_link_libraries(test_rocrand_sobol_exit hip::device)
endif()

set_target_properties(test_rocrand_sobol_exit
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
)

foreach(generator sobol32 scrambled_sobol32)
    foreach(mode reset noreset)
        set(test_name "test_rocrand_sobol_exit_${generator}_${mode}")
        add_test(
            NAME ${test_name}
            COMMAND test_rocrand_sobol_exit ${generator} ${mode}
        )
        # Not registered through add_relative_test: that path appends a gtest
        # filter, and this binary is not a GoogleTest executable. Label it so
        # the documented ctest -L quick / -L standard selections still run it.
        set_tests_properties(${test_name} PROPERTIES LABELS "quick;standard")
        file(APPEND "${INSTALL_TEST_FILE}"
            "add_test(${test_name} \"../test_rocrand_sobol_exit\" ${generator} ${mode})\n")
    endforeach()
endforeach()

rocm_install(TARGETS test_rocrand_sobol_exit COMPONENT tests)
