# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIPDNN_SKIP_TESTS)
    return()
endif()

hipdnn_add_dependency(GTest VERSION ${HIPDNN_GTEST_VERSION})
include(GoogleTest)
include(${CMAKE_CURRENT_LIST_DIR}/CheckToolVersion.cmake)

find_package(Python3 COMPONENTS Interpreter)

findandcheckllvmsymbolizer()

set(CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated global dependencies for test name validation" FORCE)
set(CHECK_EXECUTABLE_PATHS_GLOBAL "" CACHE INTERNAL "Accumulated global check executable paths" FORCE)

# Builds the test environment list with optional code coverage support
# ~~~
# Parameters:
#   OUT_VAR - The name of the variable to store the result in (will be set in PARENT_SCOPE)
# ~~~
function(_build_test_environment_list_internal OUT_VAR)
    set(ENVIRONMENT_LIST "")
    if(DEFINED TEST_ENVIRONMENT)
        set(ENVIRONMENT_LIST ${TEST_ENVIRONMENT})
    endif()

    if(HIPDNN_ENABLE_COVERAGE)
        # Ensure coverage report directory exists
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/coverage-report/profraw")

        # For code coverage builds, we want each profraw file to have a unique name.  The %m in the
        # LLVM_PROFILE_FILE environment variable will auto generate a unique id.
        list(APPEND ENVIRONMENT_LIST
             "LLVM_PROFILE_FILE=${CMAKE_BINARY_DIR}/coverage-report/profraw/%m.profraw"
        )
    endif()

    set(${OUT_VAR} ${ENVIRONMENT_LIST} PARENT_SCOPE)
endfunction() # _build_test_environment_list_internal

# Creates a custom target and ctest test to validate test names using a Python script
function(_create_test_name_validation_target_internal prefix_name)
    if(Python3_FOUND)
        # Write list of test executables with their paths to a file
        set(TEST_EXECUTABLES_FILE ${CMAKE_BINARY_DIR}/${prefix_name}_test_executables.txt)
        list(REMOVE_DUPLICATES CHECK_EXECUTABLE_PATHS_GLOBAL)
        file(WRITE ${TEST_EXECUTABLES_FILE} "")
        foreach(test_executable ${CHECK_EXECUTABLE_PATHS_GLOBAL})
            file(APPEND ${TEST_EXECUTABLES_FILE} "${test_executable}\n")
        endforeach()

        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/${prefix_name}_test_names_validated
            COMMAND
                ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/cmake/scripts/test_name_validator.py
                --test-executables ${TEST_EXECUTABLES_FILE} --build-dir ${CMAKE_BINARY_DIR} --strict
            COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/${prefix_name}_test_names_validated
            DEPENDS ${PROJECT_SOURCE_DIR}/cmake/scripts/test_name_validator.py ${CHECK_DEPENDS_GLOBAL}
            COMMENT "Validating test names with --gtest_list_tests test collection"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            VERBATIM
        )

        add_custom_target(
            ${prefix_name}-validate_test_names DEPENDS ${CMAKE_BINARY_DIR}/${prefix_name}_test_names_validated
            COMMENT "Validating test names"
        )

        # Also register as a ctest test so it runs with ctest and appears in test results
        add_test(
            NAME ${prefix_name}_test_name_validation
            COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/cmake/scripts/test_name_validator.py
                --test-executables ${TEST_EXECUTABLES_FILE}
                --build-dir ${CMAKE_BINARY_DIR}
                --strict
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
        set_tests_properties(${prefix_name}_test_name_validation PROPERTIES LABELS "unit_test;integration_test;quick")
    else()
        message(WARNING "Python3 not found. Test name validation will be skipped.")
        add_custom_target(
            ${prefix_name}-validate_test_names COMMAND ${CMAKE_COMMAND} -E echo
                                        "Test name validation skipped - Python3 not found"
            COMMENT "Skipping test name validation"
        )
    endif() # Python3_FOUND
endfunction() # _create_test_name_validation_target_internal

enable_testing() # Cmake wont discover or run tests without this line

# Internal helper function to create a ctest target
# ~~~
# Parameters:
#   PREFIX_NAME - Prefix for target names
#   TARGET_NAME - Name of the ctest target to create (will be prefixed)
#   LABEL - Optional label filter for ctest (empty string for no filter)
#   VERBOSE - Set to TRUE to add --verbose flag, FALSE otherwise
#   COMMENT - Comment describing the target
# ~~~
function(_add_ctest_target_internal PREFIX_NAME TARGET_NAME LABEL VERBOSE COMMENT)
    # Build the ctest command
    set(CTEST_CMD ${CMAKE_COMMAND} -E env ${CTEST_ENV} ${CMAKE_CTEST_COMMAND})

    # Add label filter if specified
    if(NOT "${LABEL}" STREQUAL "")
        list(APPEND CTEST_CMD -L "${LABEL}")
    endif()

    # Always add --output-on-failure
    list(APPEND CTEST_CMD --output-on-failure)

    # Add --verbose if requested
    if(VERBOSE)
        list(APPEND CTEST_CMD --verbose)
    endif()

    # Add configuration
    list(APPEND CTEST_CMD -C ${CMAKE_CFG_INTDIR})

    # Create the target with prefix
    set(FULL_TARGET_NAME "${PREFIX_NAME}-${TARGET_NAME}")
    add_custom_target(${FULL_TARGET_NAME} COMMAND ${CTEST_CMD} COMMENT "${COMMENT}" USES_TERMINAL)
    add_dependencies(${FULL_TARGET_NAME} ${PREFIX_NAME}-validate_test_names)
    message(VERBOSE "Created ${FULL_TARGET_NAME} target")
endfunction() # _add_ctest_target_internal

# Internal helper function to create the check targets for running tests via ctest
function(_create_ctest_targets_internal prefix_name)
    # cmake-format: off
    # Build test environment once for all ctest targets
    _build_test_environment_list_internal(CTEST_ENV)

    # Regular targets (without --verbose)
    _add_ctest_target_internal(${prefix_name} "check_ctest" "" FALSE "Running all tests via ctest")
    _add_ctest_target_internal(${prefix_name} "unit-check_ctest" "unit_test" FALSE "Running unit tests via ctest")
    _add_ctest_target_internal(${prefix_name} "integration-check_ctest" "integration_test" FALSE "Running integration tests via ctest")

    # Verbose targets (with --verbose)
    _add_ctest_target_internal(${prefix_name} "check_ctest-verbose" "" TRUE "Running all tests via ctest (verbose)")
    _add_ctest_target_internal(${prefix_name} "unit-check_ctest-verbose" "unit_test" TRUE "Running unit tests via ctest (verbose)")
    _add_ctest_target_internal(${prefix_name} "integration-check_ctest-verbose" "integration_test" TRUE "Running integration tests via ctest (verbose)")
    # cmake-format: on
endfunction() # create_ctest_targets

# Finalizes and creates all of the test targets
#
# Arguments:
#   prefix_name - Prefix to add to all target names (e.g., "hipdnn" creates "hipdnn-check")
#
# Creates prefixed targets (e.g., "hipdnn-check", "hipdnn-unit-check", etc.)
# In standalone builds (non-superbuild), also creates unprefixed aliases for backward compatibility
function(finalize_test_targets prefix_name)
    _create_test_name_validation_target_internal(${prefix_name})

    _create_ctest_targets_internal(${prefix_name})

    # cmake-format: off
    # Determine if we should create legacy aliases (only in standalone builds)
    set(CREATE_ALIASES FALSE)
    if(NOT ROCM_LIBS_SUPERBUILD)
        set(CREATE_ALIASES TRUE)
    endif()

    # Create prefixed test targets that depend on the prefixed _ctest targets
    # Regular targets (without --verbose)
    add_custom_target(${prefix_name}-check DEPENDS ${prefix_name}-check_ctest COMMENT "Running all tests via ctest")
    add_custom_target(${prefix_name}-unit-check DEPENDS ${prefix_name}-unit-check_ctest COMMENT "Running unit tests via ctest")
    add_custom_target(${prefix_name}-integration-check DEPENDS ${prefix_name}-integration-check_ctest COMMENT "Running integration tests via ctest")
    message(STATUS "Created ctest targets: ${prefix_name}-check, ${prefix_name}-unit-check, ${prefix_name}-integration-check")
    # Verbose targets (with --verbose)
    add_custom_target(${prefix_name}-check-verbose DEPENDS ${prefix_name}-check_ctest-verbose COMMENT "Running all tests via ctest (verbose)")
    add_custom_target(${prefix_name}-unit-check-verbose DEPENDS ${prefix_name}-unit-check_ctest-verbose COMMENT "Running unit tests via ctest (verbose)")
    add_custom_target(${prefix_name}-integration-check-verbose DEPENDS ${prefix_name}-integration-check_ctest-verbose COMMENT "Running integration tests via ctest (verbose)")
    message(STATUS "Created ctest verbose targets: ${prefix_name}-check-verbose, ${prefix_name}-unit-check-verbose, ${prefix_name}-integration-check-verbose")

    # Create legacy unprefixed aliases for backward compatibility (standalone builds only)
    if(CREATE_ALIASES)
        add_custom_target(check DEPENDS ${prefix_name}-check COMMENT "Alias for ${prefix_name}-check")
        add_custom_target(unit-check DEPENDS ${prefix_name}-unit-check COMMENT "Alias for ${prefix_name}-unit-check")
        add_custom_target(integration-check DEPENDS ${prefix_name}-integration-check COMMENT "Alias for ${prefix_name}-integration-check")
        add_custom_target(check-verbose DEPENDS ${prefix_name}-check-verbose COMMENT "Alias for ${prefix_name}-check-verbose")
        add_custom_target(unit-check-verbose DEPENDS ${prefix_name}-unit-check-verbose COMMENT "Alias for ${prefix_name}-unit-check-verbose")
        add_custom_target(integration-check-verbose DEPENDS ${prefix_name}-integration-check-verbose COMMENT "Alias for ${prefix_name}-integration-check-verbose")
        message(STATUS "Created legacy alias targets for backward compatibility")
    endif()
    # cmake-format: on
endfunction() # finalize_test_targets

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

    # Add executable suffix if needed (e.g., .exe on Windows)
    if(CMAKE_EXECUTABLE_SUFFIX)
        set(TARGET_EXE "${TARGET_EXE}${CMAKE_EXECUTABLE_SUFFIX}")
    endif()

    message(STATUS "Appending ${APPEND_FUNCTION_SUFFIX} check target: ${TARGET} -> ${TARGET_EXE} in working directory: ${WORKING_DIR}")

    # Track the dependencies for test name validation
    set(CHECK_DEPENDS_GLOBAL ${CHECK_DEPENDS_GLOBAL} ${TARGET}
        CACHE INTERNAL "Accumulated global dependencies for test name validation" FORCE
    )
    # Track the binary paths for test name validation
    set(CHECK_EXECUTABLE_PATHS_GLOBAL ${CHECK_EXECUTABLE_PATHS_GLOBAL} "${CMAKE_INSTALL_BINDIR}/${TARGET_EXE}"
        CACHE INTERNAL "Accumulated global check executable paths" FORCE
    )

    # Track this test target for later use in generating installed CTestTestfile.cmake
    set_property(GLOBAL APPEND PROPERTY HIPDNN_TEST_TARGETS ${TARGET})

    set_target_properties(
        ${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
    )

    # Make test executables relocatable so they can find libraries when build directory is moved
    # Include both the main lib directory and the engine plugin directories
    set_target_properties(
        ${TARGET}
        PROPERTIES
            INSTALL_RPATH
            "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR};\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/hipdnn_plugins/engines"
            INSTALL_RPATH_USE_LINK_PATH TRUE
            BUILD_RPATH_USE_ORIGIN TRUE
    )

    # Install test executables to bin directory
    install(TARGETS ${TARGET} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    # Combine primary label with any extra labels
    set(ALL_LABELS ${APPEND_FUNCTION_SUFFIX})
    if(EXTRA_LABELS)
        list(APPEND ALL_LABELS ${EXTRA_LABELS})
    endif()

    add_test(NAME ${TARGET} COMMAND ${TARGET} WORKING_DIRECTORY ${WORKING_DIR})
    set_tests_properties(${TARGET} PROPERTIES LABELS "${ALL_LABELS}")
    if(DEFINED TEST_ENVIRONMENT)
        set_tests_properties(${TARGET} PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}")
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

# Install CTest configuration files for direct test execution This should be called once at the end
# of the main CMakeLists.txt after all tests are registered
function(install_hipdnn_ctest_files)
    # Define the CTest installation directory
    set(HIPDNN_CTEST_FILE_INSTALL_PATH "${CMAKE_INSTALL_BINDIR}/hipdnn")

    # Generate a new CTestTestfile.cmake that references installed test executables
    set(INSTALLED_CTEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake.install")

    file(WRITE "${INSTALLED_CTEST_FILE}"
         "# Autogenerated CTestTestfile for installed hipDNN tests\n"
    )
    file(APPEND "${INSTALLED_CTEST_FILE}" "# Generated by hipDNN build system\n\n")

    # Get all test targets that were registered
    get_property(all_tests GLOBAL PROPERTY HIPDNN_TEST_TARGETS)

    foreach(test_target ${all_tests})
        file(APPEND "${INSTALLED_CTEST_FILE}" "add_test(${test_target} \"../${test_target}\")\n")
    endforeach()

    # Install the generated CTestTestfile.cmake to HIPDNN_CTEST_FILE_INSTALL_PATH
    install(FILES "${INSTALLED_CTEST_FILE}" DESTINATION ${HIPDNN_CTEST_FILE_INSTALL_PATH}
            RENAME CTestTestfile.cmake
    )

endfunction() # install_hipdnn_ctest_files
