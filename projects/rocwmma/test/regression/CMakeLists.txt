###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# Regression test category - simplified test infrastructure for bug reproducers
#
# Key differences from other test categories:
# - Single executable containing all regression tests
# - Uses standard GTest::gtest_main (not rocwmma_gtest_main.cpp)
# - No emulation framework (no smoke/regression/extended variants)
# - Validation-only (no benchmark mode)
# - Minimal boilerplate per test
# - Flat directory structure (no subdirectories)
#
# Adding a new test: Add a new <name>.cpp file in this directory and rebuild (no CMake changes needed).
#                    Recommended naming convention: issue_<number>.cpp (for example, issue_4398.cpp).

# Collect all regression test source files
file(GLOB REGRESSION_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Only build if we have test files
if(REGRESSION_TEST_SOURCES)
    # Create single executable from all test sources
    set(TEST_TARGET rocwmma_regression_tests)

    add_executable(${TEST_TARGET}
        ${REGRESSION_TEST_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/../hip_device.cpp)

    # Link against rocWMMA and GTest
    target_link_libraries(${TEST_TARGET} rocwmma GTest::gtest GTest::gtest_main hiprtc::hiprtc)
    target_link_libraries(${TEST_TARGET} ${ROCWMMA_OPENMP_LIBRARY})
    if(UNIX)
        set_target_properties(${TEST_TARGET} PROPERTIES INSTALL_RPATH "\$ORIGIN/../llvm/lib")
        target_link_libraries(${TEST_TARGET} "-fno-rtlib-add-rpath")
    endif()

    # Include directories
    target_include_directories(${TEST_TARGET} PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR}
                               ${ROCWMMA_TEST_INCLUDE_DIRS})

    # Use offload compress flag if supported
    if(BUILD_OFFLOAD_COMPRESS AND CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
        target_compile_options(${TEST_TARGET} PRIVATE "--offload-compress")
    endif()

    # Add validation tests definition
    target_compile_definitions(${TEST_TARGET} PRIVATE ROCWMMA_VALIDATION_TESTS)

    # Add support to build the target's assembly files
    if(ROCWMMA_BUILD_ASSEMBLY)
        foreach(file_name ${REGRESSION_TEST_SOURCES})
            # Replicate the current source tree in the build output/assembly folder
            file(RELATIVE_PATH relative_file_path "${CMAKE_CURRENT_SOURCE_DIR}" "${file_name}")
            string(REPLACE "../" "__/" relative_file_string "${relative_file_path}")
            add_custom_command(TARGET ${TEST_TARGET}
                               POST_BUILD
                               COMMAND make ARGS ${relative_file_string}.s
                               COMMAND ${CMAKE_COMMAND} -E copy
                                 "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TEST_TARGET}.dir/${relative_file_string}.s"
                                 "${CMAKE_CURRENT_BINARY_DIR}/assembly/${relative_file_string}.s"
                               WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        endforeach()
    endif()

    # Register single CTest test
    add_test(NAME regression_tests COMMAND ${TEST_TARGET})
    set_property(TEST regression_tests PROPERTY SKIP_REGULAR_EXPRESSION "no ROCm-capable device" "unsupported host device")

    # Install the executable
    rocm_install_targets(
        TARGETS ${TEST_TARGET}
        COMPONENT tests
    )

    # Add to install test file
    get_target_property(EXE_NAME ${TEST_TARGET} RUNTIME_OUTPUT_NAME)
    if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
        get_target_property(EXE_NAME ${TEST_TARGET} OUTPUT_NAME)
        if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
            set(EXE_NAME "${TEST_TARGET}")
        endif()
    endif()

    file(APPEND "${INSTALL_TEST_FILE}" "add_test(regression_tests \"../${EXE_NAME}\")\n")
    file(APPEND "${INSTALL_TEST_FILE}" "set_tests_properties(regression_tests PROPERTIES SKIP_REGULAR_EXPRESSION \"no ROCm-capable device;unsupported host device\")\n")

  file(APPEND "${INSTALL_SMOKE_TEST_FILE}" "add_test(\"regression_tests\" \"../../${EXE_NAME}\")\n")
  file(APPEND "${INSTALL_SMOKE_TEST_FILE}" "set_tests_properties(\"regression_tests\" PROPERTIES SKIP_REGULAR_EXPRESSION \"no ROCm-capable device;unsupported host device\")\n")

  file(APPEND "${INSTALL_REGRESSION_TEST_FILE}" "add_test(\"regression_tests\" \"../../${EXE_NAME}\")\n")
  file(APPEND "${INSTALL_REGRESSION_TEST_FILE}" "set_tests_properties(\"regression_tests\" PROPERTIES SKIP_REGULAR_EXPRESSION \"no ROCm-capable device;unsupported host device\")\n")

    message(STATUS "Regression tests: Enabled (${TEST_TARGET})")
else()
    message(STATUS "Regression tests: No test files found in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()
