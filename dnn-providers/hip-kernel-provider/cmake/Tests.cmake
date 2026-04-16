# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(NOT HIPKERNELPROVIDER_ENABLE_TESTS)
    return()
endif()

include(GoogleTest)

set(CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated global dependencies for test name validation" FORCE)
set(CHECK_EXECUTABLE_PATHS_GLOBAL "" CACHE INTERNAL "Accumulated global check executable paths" FORCE)

enable_testing() # Cmake wont discover or run tests without this line

if(HIPKERNELPROVIDER_ENABLE_COVERAGE)
    # Ensure coverage report directory exists
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/coverage-report/profraw")

    # For code coverage builds, we want each profraw file to have a unique name. The %m in the
    # LLVM_PROFILE_FILE environment variable will auto generate a unique id.
    set(COVERAGE_ENVIRONMENT
        "LLVM_PROFILE_FILE=${CMAKE_BINARY_DIR}/coverage-report/profraw/%m.profraw"
    )
endif()

# ~~~
# Internal helper function to record, configure, and register a ctest test target. Assumes that the
# test target is a gtest executable, setting up:
# - Test name validation tracking (adds to global dependency and executable path lists)
# - RPATH settings for relocatable test executables
# - Installation rules for test binaries
# - CTest registration with appropriate labels (e.g. unit / integration test labels)
#
# Parameters:
#   APPEND_FUNCTION_SUFFIX - Primary label to apply to the test (e.g., "unit_test", "integration_test", "test")
#   TARGET - Name of the test executable target (must already exist)
#   WORKING_DIR - Working directory for test execution
#   EXTRA_LABELS - (Optional) Additional labels to apply to the test (semicolon-separated list)
# ~~~
function(_add_test_target_internal APPEND_FUNCTION_SUFFIX TARGET WORKING_DIR)
    # Parse optional extra labels from remaining arguments
    set(EXTRA_LABELS ${ARGN})
    set(TARGET_EXE ${TARGET})

    if(CMAKE_EXECUTABLE_SUFFIX)
        set(TARGET_EXE "${TARGET_EXE}${CMAKE_EXECUTABLE_SUFFIX}")
    endif()

    message(STATUS "Appending ${APPEND_FUNCTION_SUFFIX} check target: ${TARGET} -> ${TARGET_EXE} in working directory: ${WORKING_DIR}")

    set(CHECK_DEPENDS_GLOBAL ${CHECK_DEPENDS_GLOBAL} ${TARGET}
        CACHE INTERNAL "Accumulated global dependencies for test name validation" FORCE
    )
    set(CHECK_EXECUTABLE_PATHS_GLOBAL ${CHECK_EXECUTABLE_PATHS_GLOBAL} "${CMAKE_INSTALL_BINDIR}/${TARGET_EXE}"
        CACHE INTERNAL "Accumulated global check executable paths" FORCE
    )

    set_property(GLOBAL APPEND PROPERTY HIP_KERNEL_PROVIDER_TEST_TARGETS ${TARGET})

    set_target_properties(
        ${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
    )

    set_target_properties(
        ${TARGET}
        PROPERTIES
            INSTALL_RPATH
            "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR};\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/hipdnn_plugins/engines"
            INSTALL_RPATH_USE_LINK_PATH TRUE
            BUILD_RPATH_USE_ORIGIN TRUE
    )

    install(TARGETS ${TARGET} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    # Combine primary label with any extra labels
    set(ALL_LABELS ${APPEND_FUNCTION_SUFFIX})
    if(EXTRA_LABELS)
        list(APPEND ALL_LABELS ${EXTRA_LABELS})
    endif()

    add_test(NAME ${TARGET} COMMAND ${TARGET} WORKING_DIRECTORY ${WORKING_DIR})
    set_tests_properties(${TARGET} PROPERTIES LABELS "${ALL_LABELS}")

    if(COVERAGE_ENVIRONMENT)
        set_tests_properties(${TARGET} PROPERTIES ENVIRONMENT "${COVERAGE_ENVIRONMENT}")
    endif()
endfunction() # _add_test_target_internal

# ~~~
# Adds a unit test target
#
# Usage:
#   add_unit_test_target(TARGET WORKING_DIR [LABELS label1 label2 ...])
# ~~~
function(add_unit_test_target TARGET WORKING_DIR)
    cmake_parse_arguments(ARG "" "" "LABELS" ${ARGN})
    _add_test_target_internal(unit_test ${TARGET} ${WORKING_DIR} ${ARG_LABELS})
endfunction() # add_unit_test_target

# ~~~
# Adds an integration test target
#
# Usage:
#   add_integration_test_target(TARGET WORKING_DIR [LABELS label1 label2 ...])
# ~~~
function(add_integration_test_target TARGET WORKING_DIR)
    cmake_parse_arguments(ARG "" "" "LABELS" ${ARGN})
    _add_test_target_internal(integration_test ${TARGET} ${WORKING_DIR} ${ARG_LABELS})
endfunction() # add_integration_test_target

# Finalizes and creates all of the test targets
function(finalize_test_targets)
    # cmake-format: off
    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR}
        COMMENT "Running all tests via ctest"
        USES_TERMINAL
    )

    add_custom_target(unit-check
        COMMAND ${CMAKE_CTEST_COMMAND} -L unit_test --output-on-failure -C ${CMAKE_CFG_INTDIR}
        COMMENT "Running unit tests via ctest"
        USES_TERMINAL
    )

    add_custom_target(integration-check
        COMMAND ${CMAKE_CTEST_COMMAND} -L integration_test --output-on-failure -C ${CMAKE_CFG_INTDIR}
        COMMENT "Running integration tests via ctest"
        USES_TERMINAL
    )
    # cmake-format: on
    message(STATUS "Created ctest targets: check, unit-check, integration-check")
endfunction()

# Install CTest configuration files for direct test execution
function(install_hip_kernel_provider_ctest_files)
    set(HIP_KERNEL_PROVIDER_CTEST_FILE_INSTALL_PATH "${CMAKE_INSTALL_BINDIR}/hip_kernel_provider")

    set(INSTALLED_CTEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake.install")

    file(WRITE "${INSTALLED_CTEST_FILE}"
         "# Autogenerated CTestTestfile for installed hip_kernel_provider tests\n"
    )
    file(APPEND "${INSTALLED_CTEST_FILE}" "# Generated by hip_kernel_provider build system\n\n")

    get_property(all_tests GLOBAL PROPERTY HIP_KERNEL_PROVIDER_TEST_TARGETS)

    foreach(test_target ${all_tests})
        file(APPEND "${INSTALLED_CTEST_FILE}" "add_test(${test_target} \"../${test_target}\")\n")
    endforeach()

    install(FILES "${INSTALLED_CTEST_FILE}"
            DESTINATION ${HIP_KERNEL_PROVIDER_CTEST_FILE_INSTALL_PATH} RENAME CTestTestfile.cmake
    )
endfunction()
