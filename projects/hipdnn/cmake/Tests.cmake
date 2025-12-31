# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIP_DNN_SKIP_TESTS)
    return()
endif()

hipdnn_add_dependency(GTest VERSION ${HIPDNN_GTEST_VERSION})
include(GoogleTest)
include(${CMAKE_CURRENT_LIST_DIR}/CheckToolVersion.cmake)

find_package(Python3 COMPONENTS Interpreter)

findandcheckllvmsymbolizer()

# Set executable prefix based on platform
if(WIN32)
    set(EXEC_PREFIX "")
else()
    set(EXEC_PREFIX "./")
endif()

set(CHECK_COMMAND_GLOBAL "" CACHE INTERNAL "Accumulated check commands" FORCE)
set(CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated check depends" FORCE)
set(CHECK_EXECUTABLE_PATHS_GLOBAL "" CACHE INTERNAL "Accumulated check executable paths" FORCE)

# Global collections for unit tests
set(UNIT_CHECK_COMMAND_GLOBAL "" CACHE INTERNAL "Accumulated unit check commands" FORCE)
set(UNIT_CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated unit check depends" FORCE)

# Global collections for integration tests
set(INTEGRATION_CHECK_COMMAND_GLOBAL "" CACHE INTERNAL "Accumulated integration check commands"
                                              FORCE
)
set(INTEGRATION_CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated integration check depends"
                                              FORCE
)

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
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/coverage_report/profraw")

        # For code coverage builds, we want each profraw file to have a unique name.  The %m in the
        # LLVM_PROFILE_FILE environment variable will auto generate a unique id.
        list(APPEND ENVIRONMENT_LIST
             "LLVM_PROFILE_FILE=${CMAKE_BINARY_DIR}/coverage_report/profraw/%m.profraw"
        )
    endif()

    set(${OUT_VAR} ${ENVIRONMENT_LIST} PARENT_SCOPE)
endfunction() # _build_test_environment_list_internal

# Creates a custom target to validate test names using a Python script
function(_create_test_name_validation_target_internal)
    if(Python3_FOUND)
        # Write list of test executables with their paths to a file
        set(TEST_EXECUTABLES_FILE ${CMAKE_BINARY_DIR}/test_executables.txt)
        list(REMOVE_DUPLICATES CHECK_EXECUTABLE_PATHS_GLOBAL)
        file(WRITE ${TEST_EXECUTABLES_FILE} "")
        foreach(test_executable ${CHECK_EXECUTABLE_PATHS_GLOBAL})
            file(APPEND ${TEST_EXECUTABLES_FILE} "${test_executable}\n")
        endforeach()

        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/test_names_validated
            COMMAND
                ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/scripts/test_name_validator.py
                --test-executables ${TEST_EXECUTABLES_FILE} --build-dir ${CMAKE_BINARY_DIR} --strict
            COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/test_names_validated
            DEPENDS ${CMAKE_SOURCE_DIR}/cmake/scripts/test_name_validator.py ${CHECK_DEPENDS_GLOBAL}
            COMMENT "Validating test names with --gtest_list_tests test collection"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            VERBATIM
        )

        add_custom_target(
            validate_test_names DEPENDS ${CMAKE_BINARY_DIR}/test_names_validated
            COMMENT "Validating test names"
        )
    else()
        message(WARNING "Python3 not found. Test name validation will be skipped.")
        add_custom_target(
            validate_test_names COMMAND ${CMAKE_COMMAND} -E echo
                                        "Test name validation skipped - Python3 not found"
            COMMENT "Skipping test name validation"
        )
    endif() # Python3_FOUND
endfunction() # _create_test_name_validation_target_internal

# Generic internal function to append tests to check targets
function(_append_test_to_check_target_internal TARGET WORKING_DIR TEST_TYPE STATUS_MESSAGE)
    if(STATUS_MESSAGE)
        message(STATUS "${STATUS_MESSAGE}: ${TARGET} in working directory: ${WORKING_DIR}")
    endif()

    if("${TEST_TYPE}" STREQUAL "UNIT")
        set(COMMAND_VAR "UNIT_CHECK_COMMAND_GLOBAL")
        set(DEPENDS_VAR "UNIT_CHECK_DEPENDS_GLOBAL")
        set(CACHE_DESC "Accumulated unit check targets")
    elseif("${TEST_TYPE}" STREQUAL "INTEGRATION")
        set(COMMAND_VAR "INTEGRATION_CHECK_COMMAND_GLOBAL")
        set(DEPENDS_VAR "INTEGRATION_CHECK_DEPENDS_GLOBAL")
        set(CACHE_DESC "Accumulated integration check targets")
    else()
        set(COMMAND_VAR "CHECK_COMMAND_GLOBAL")
        set(DEPENDS_VAR "CHECK_DEPENDS_GLOBAL")
        set(CACHE_DESC "Accumulated check targets")
    endif()

    # Build environment list properly
    _build_test_environment_list_internal(ENVIRONMENT_LIST)

    set(NEW_COMMAND "")
    if("${${COMMAND_VAR}}" STREQUAL "")
        set(NEW_COMMAND ${CMAKE_COMMAND} -E env ${ENVIRONMENT_LIST}
                        ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${TARGET}
        )
    else()
        set(NEW_COMMAND && ${CMAKE_COMMAND} -E env ${ENVIRONMENT_LIST}
                        ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${TARGET}
        )
    endif()

    set(${COMMAND_VAR} ${${COMMAND_VAR}} ${NEW_COMMAND} CACHE INTERNAL "${CACHE_DESC}" FORCE)
    set(${DEPENDS_VAR} ${${DEPENDS_VAR}} ${TARGET}
        CACHE INTERNAL "Accumulated ${TEST_TYPE} check depends" FORCE
    )

    # Track the binary paths for test name validation
    set(EXECUTABLE_PATH "${CMAKE_INSTALL_BINDIR}/${TARGET}")
    set(CHECK_EXECUTABLE_PATHS_GLOBAL ${CHECK_EXECUTABLE_PATHS_GLOBAL} ${EXECUTABLE_PATH}
        CACHE INTERNAL "Accumulated check executable paths" FORCE
    )
endfunction() # _append_test_to_check_target_internal

# Generic internal function to finalize DIY check targets
function(_finalize_check_target_internal TARGET_NAME COMMAND_VAR DEPENDS_VAR)
    add_custom_target(
        ${TARGET_NAME}
        COMMAND ${${COMMAND_VAR}}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS ${${DEPENDS_VAR}}
        VERBATIM
        COMMENT "Running ${TARGET_NAME}"
        USES_TERMINAL
    )
    message(VERBOSE "Created ${TARGET_NAME} target")
endfunction() # _finalize_check_target_internal

# Internal function to finalize the DIY unclassified check-old target
function(_finalize_unclassified_check_target_internal)
    _finalize_check_target_internal("check-old" "CHECK_COMMAND_GLOBAL" "CHECK_DEPENDS_GLOBAL")
endfunction() # _finalize_unclassified_check_target_internal

# Internal function to finalize the DIY unit-check-old target
function(_finalize_unit_check_target_internal)
    _finalize_check_target_internal(
        "unit-check-old" "UNIT_CHECK_COMMAND_GLOBAL" "UNIT_CHECK_DEPENDS_GLOBAL"
    )
endfunction() # _finalize_unit_check_target_internal

# Internal function to finalize the DIY integration-check-old target
function(_finalize_integration_check_target_internal)
    _finalize_check_target_internal(
        "integration-check-old" "INTEGRATION_CHECK_COMMAND_GLOBAL"
        "INTEGRATION_CHECK_DEPENDS_GLOBAL"
    )
endfunction() # _finalize_integration_check_target_internal

enable_testing() # Cmake wont discover or run tests without this line

# Internal helper function to create a ctest target
# ~~~
# Parameters:
#   TARGET_NAME - Name of the ctest target to create
#   LABEL - Optional label filter for ctest (empty string for no filter)
#   VERBOSE - Set to TRUE to add --verbose flag, FALSE otherwise
#   COMMENT - Comment describing the target
# ~~~
function(_add_ctest_target_internal TARGET_NAME LABEL VERBOSE COMMENT)
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

    # Create the target
    add_custom_target(${TARGET_NAME} COMMAND ${CTEST_CMD} COMMENT "${COMMENT}" USES_TERMINAL)
    add_dependencies(${TARGET_NAME} validate_test_names)
    message(VERBOSE "Created ${TARGET_NAME} target")
endfunction() # _add_ctest_target_internal

# Internal helper function to create the (new) check targets for running tests via ctest
function(_create_ctest_targets_internal)
    # cmake-format: off
    # Build test environment once for all ctest targets
    _build_test_environment_list_internal(CTEST_ENV)

    # Regular targets (without --verbose)
    _add_ctest_target_internal(check_ctest "" FALSE "Running all tests via ctest")
    _add_ctest_target_internal(unit-check_ctest "unit_test" FALSE "Running unit tests via ctest")
    _add_ctest_target_internal(integration-check_ctest "integration_test" FALSE "Running integration tests via ctest")

    # Verbose targets (with --verbose)
    _add_ctest_target_internal(check_ctest-verbose "" TRUE "Running all tests via ctest (verbose)")
    _add_ctest_target_internal(unit-check_ctest-verbose "unit_test" TRUE "Running unit tests via ctest (verbose)")
    _add_ctest_target_internal(integration-check_ctest-verbose "integration_test" TRUE "Running integration tests via ctest (verbose)")
    # cmake-format: on
endfunction() # create_ctest_targets

# Finalizes and creates all of the test targets
function(finalize_test_targets)
    _create_test_name_validation_target_internal()

    _create_ctest_targets_internal()

    _finalize_unclassified_check_target_internal()
    _finalize_unit_check_target_internal()
    _finalize_integration_check_target_internal()

    add_dependencies(check-old validate_test_names)

    # cmake-format: off
    # Create alias test targets without '_ctest' in the name
    # Regular targets (without --verbose)
    add_custom_target(check DEPENDS check_ctest COMMENT "Running all tests via ctest")
    add_custom_target(unit-check DEPENDS unit-check_ctest COMMENT "Running unit tests via ctest")
    add_custom_target(integration-check DEPENDS integration-check_ctest COMMENT "Running integration tests via ctest")
    message(STATUS "Created ctest targets: check, unit-check, integration-check")
    # Verbose targets (with --verbose)
    add_custom_target(check-verbose DEPENDS check_ctest-verbose COMMENT "Running all tests via ctest (verbose)")
    add_custom_target(unit-check-verbose DEPENDS unit-check_ctest-verbose COMMENT "Running unit tests via ctest (verbose)")
    add_custom_target(integration-check-verbose DEPENDS integration-check_ctest-verbose COMMENT "Running integration tests via ctest (verbose)")
    message(STATUS "Created ctest verbose targets: check-verbose, unit-check-verbose, integration-check-verbose")
    # cmake-format: on
endfunction() # finalize_test_targets

# Internal (old) DIY function to add a test target (assumes the test executable is a gtest
# executable) Still needed for collecting list of test executables in HIPDNN_TEST_TARGETS and
# setting test-type labels. TODO: Deprecate this funcion and remove manual test collection once new
# ctest-based method is proven.
function(_add_gtest_target_internal APPEND_FUNCTION_SUFFIX TARGET WORKING_DIR)
    if("${APPEND_FUNCTION_SUFFIX}" STREQUAL "test")
        _append_test_to_check_target_internal(
            ${TARGET} ${WORKING_DIR} "" "Appending unclassified check target"
        )
    elseif("${APPEND_FUNCTION_SUFFIX}" STREQUAL "unit_test")
        _append_test_to_check_target_internal(
            ${TARGET} ${WORKING_DIR} "UNIT" "Appending unit check target"
        )
        _append_test_to_check_target_internal(${TARGET} ${WORKING_DIR} "" "")
    elseif("${APPEND_FUNCTION_SUFFIX}" STREQUAL "integration_test")
        _append_test_to_check_target_internal(
            ${TARGET} ${WORKING_DIR} "INTEGRATION" "Appending integration check target"
        )
        _append_test_to_check_target_internal(${TARGET} ${WORKING_DIR} "" "")
    else()
        message(FATAL_ERROR "Unknown test type suffix: ${APPEND_FUNCTION_SUFFIX}")
    endif()

    set_target_properties(
        ${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
    )

    # Track this test target for later use in generating installed CTestTestfile.cmake
    set_property(GLOBAL APPEND PROPERTY HIPDNN_TEST_TARGETS ${TARGET})

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

    add_test(NAME ${TARGET} COMMAND ${TARGET} WORKING_DIRECTORY ${WORKING_DIR})
    set_tests_properties(${TARGET} PROPERTIES LABELS ${APPEND_FUNCTION_SUFFIX})
endfunction() # _add_gtest_target_internal

# Adds a generic test target
function(add_unclassified_test_target TARGET WORKING_DIR)
    _add_gtest_target_internal(test ${TARGET} ${WORKING_DIR})
endfunction() # add_unclassified_test_target

# Adds a unit test target
function(add_unit_test_target TARGET WORKING_DIR)
    _add_gtest_target_internal(unit_test ${TARGET} ${WORKING_DIR})
endfunction() # add_unit_test_target

# Adds an integration test target
function(add_integration_test_target TARGET WORKING_DIR)
    _add_gtest_target_internal(integration_test ${TARGET} ${WORKING_DIR})
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
