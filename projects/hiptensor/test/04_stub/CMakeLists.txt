###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 ###############################################################################

# Tests for the host-only hipTensor stub.
#
# hiptensor_stub_test links the stub source directly (not the installed
# libhiptensor, which in a device build is the real CK-backed library) and
# asserts the stub's contract: every operational entry point returns
# HIPTENSOR_STATUS_NOT_SUPPORTED, while the informational helpers keep their
# normal semantics. It is host-only -- no device code, no Composable Kernel --
# so it builds and runs for any GPU_TARGETS.

set(_stub_dir ${PROJECT_SOURCE_DIR}/library/stub)

# The stub includes hiptensor.h, which pulls in the generated export header.
# Provide a minimal one so the test is self-contained regardless of whether the
# device or stub library produced the shared copy.
set(_stub_test_inc ${CMAKE_CURRENT_BINARY_DIR}/stub_test_include)
file(WRITE "${_stub_test_inc}/hiptensor/internal/hiptensor-export.h"
"#ifndef HIPTENSOR_EXPORT_H
#define HIPTENSOR_EXPORT_H
#define HIPTENSOR_EXPORT
#define HIPTENSOR_NO_EXPORT
#endif // HIPTENSOR_EXPORT_H
")

add_executable(hiptensor_stub_test
    ${CMAKE_CURRENT_SOURCE_DIR}/hiptensor_stub_test.cpp
    ${_stub_dir}/hiptensor_stub.cpp)
target_compile_options(hiptensor_stub_test PRIVATE ${CLANG_DRIVER_MODE})
target_link_options(hiptensor_stub_test PRIVATE ${CLANG_DRIVER_MODE})
target_compile_definitions(hiptensor_stub_test PRIVATE hiptensor_EXPORTS)
target_include_directories(hiptensor_stub_test PRIVATE
    ${PROJECT_SOURCE_DIR}/library/include
    ${_stub_test_inc}
    ${_stub_test_inc}/hiptensor)
target_link_libraries(hiptensor_stub_test PRIVATE hip::host GTest::gtest GTest::gtest_main)

add_dependencies(hiptensor_tests hiptensor_stub_test)

add_test(NAME hiptensor_stub_test COMMAND hiptensor_stub_test)
set_property(TEST hiptensor_stub_test PROPERTY WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

rocm_install_targets(TARGETS hiptensor_stub_test COMPONENT tests)

# Add to the install-tree CTest manifests so the test runs from the installed
# package the same way the other component tests do.
file(APPEND "${INSTALL_TEST_FILE}" "add_test(hiptensor_stub_test \"../hiptensor_stub_test\")\n")
file(APPEND "${INSTALL_SMOKE_TEST_FILE}" "add_test(\"hiptensor_stub_test smoke\" \"../../hiptensor_stub_test\")\n")
file(APPEND "${INSTALL_REGRESSION_TEST_FILE}" "add_test(\"hiptensor_stub_test regression\" \"../../hiptensor_stub_test\")\n")

# Python unit tests for the coverage-source generator. Registered as a CTest so
# `ctest` exercises the parser/render logic too. Build-tree only (the generator
# script is not shipped in the install package). Uses the stdlib unittest runner
# (always available) rather than pytest, which is not present on all CI images.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME hiptensor_generate_api_coverage_pytest
        COMMAND ${Python3_EXECUTABLE} -m unittest test_generate_api_coverage
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
