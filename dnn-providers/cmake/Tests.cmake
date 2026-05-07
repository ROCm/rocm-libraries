# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)
include(CheckToolVersion)

find_package(Python3 COMPONENTS Interpreter)

findandcheckllvmsymbolizer()

set(CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated global dependencies for test name validation" FORCE)
set(CHECK_EXECUTABLE_PATHS_GLOBAL "" CACHE INTERNAL "Accumulated global check executable paths" FORCE)

# Resolve the path to the test name validator script once at include time
set(_TEST_NAME_VALIDATOR_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/scripts/test_name_validator.py"
    CACHE INTERNAL "Path to the common test name validator script"
)

# Builds the test environment list with optional code coverage support
# ~~~
# Parameters:
#   OUT_VAR - The name of the variable to store the result in (will be set in PARENT_SCOPE)
#   COVERAGE - If TRUE, add LLVM_PROFILE_FILE to the environment
# ~~~
function(_build_test_environment_list_internal OUT_VAR COVERAGE)
    set(ENVIRONMENT_LIST "")
    if(DEFINED TEST_ENVIRONMENT)
        set(ENVIRONMENT_LIST ${TEST_ENVIRONMENT})
    endif()

    if(COVERAGE)
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/coverage-report/profraw")

        list(APPEND ENVIRONMENT_LIST
             "LLVM_PROFILE_FILE=${CMAKE_BINARY_DIR}/coverage-report/profraw/%m.profraw"
        )
    endif()

    set(${OUT_VAR} ${ENVIRONMENT_LIST} PARENT_SCOPE)
endfunction()

# Creates a custom target to validate test names using a Python script.
# Validation runs when the script exists at the common cmake/scripts/ location and
# SKIP_TEST_NAME_VALIDATION is not set; otherwise a dummy target is created.
function(_create_test_name_validation_target_internal prefix_name)
    if(Python3_FOUND AND EXISTS "${_TEST_NAME_VALIDATOR_SCRIPT}" AND NOT SKIP_TEST_NAME_VALIDATION)
        set(TEST_EXECUTABLES_FILE ${CMAKE_BINARY_DIR}/${prefix_name}_test_executables.txt)
        list(REMOVE_DUPLICATES CHECK_EXECUTABLE_PATHS_GLOBAL)
        file(WRITE ${TEST_EXECUTABLES_FILE} "")
        foreach(test_executable ${CHECK_EXECUTABLE_PATHS_GLOBAL})
            file(APPEND ${TEST_EXECUTABLES_FILE} "${test_executable}\n")
        endforeach()

        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/${prefix_name}_test_names_validated
            COMMAND
                ${Python3_EXECUTABLE} ${_TEST_NAME_VALIDATOR_SCRIPT}
                --test-executables ${TEST_EXECUTABLES_FILE} --build-dir ${CMAKE_BINARY_DIR} --strict
            COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/${prefix_name}_test_names_validated
            DEPENDS ${_TEST_NAME_VALIDATOR_SCRIPT} ${CHECK_DEPENDS_GLOBAL}
            COMMENT "Validating test names with --gtest_list_tests test collection"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            VERBATIM
        )

        add_custom_target(
            ${prefix_name}-validate_test_names DEPENDS ${CMAKE_BINARY_DIR}/${prefix_name}_test_names_validated
            COMMENT "Validating test names"
        )
    else()
        add_custom_target(
            ${prefix_name}-validate_test_names COMMAND ${CMAKE_COMMAND} -E echo
                                        "Test name validation skipped"
            COMMENT "Skipping test name validation"
        )
    endif()
endfunction()

enable_testing()

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
    set(CTEST_CMD ${CMAKE_COMMAND} -E env ${CTEST_ENV} ${CMAKE_CTEST_COMMAND})

    if(NOT "${LABEL}" STREQUAL "")
        list(APPEND CTEST_CMD -L "${LABEL}")
    endif()

    list(APPEND CTEST_CMD --output-on-failure)

    if(VERBOSE)
        list(APPEND CTEST_CMD --verbose)
    endif()

    list(APPEND CTEST_CMD -C ${CMAKE_CFG_INTDIR})

    set(FULL_TARGET_NAME "${PREFIX_NAME}-${TARGET_NAME}")
    add_custom_target(${FULL_TARGET_NAME} COMMAND ${CTEST_CMD} COMMENT "${COMMENT}" USES_TERMINAL)
    add_dependencies(${FULL_TARGET_NAME} ${PREFIX_NAME}-validate_test_names)
    message(VERBOSE "Created ${FULL_TARGET_NAME} target")
endfunction()

# Internal helper function to create the check targets for running tests via ctest
function(_create_ctest_targets_internal prefix_name coverage)
    # cmake-format: off
    _build_test_environment_list_internal(CTEST_ENV ${coverage})

    # Regular targets (without --verbose)
    _add_ctest_target_internal(${prefix_name} "check_ctest" "" FALSE "Running all tests via ctest")
    _add_ctest_target_internal(${prefix_name} "unit-check_ctest" "unit_test" FALSE "Running unit tests via ctest")
    _add_ctest_target_internal(${prefix_name} "integration-check_ctest" "integration_test" FALSE "Running integration tests via ctest")

    # Verbose targets (with --verbose)
    _add_ctest_target_internal(${prefix_name} "check_ctest-verbose" "" TRUE "Running all tests via ctest (verbose)")
    _add_ctest_target_internal(${prefix_name} "unit-check_ctest-verbose" "unit_test" TRUE "Running unit tests via ctest (verbose)")
    _add_ctest_target_internal(${prefix_name} "integration-check_ctest-verbose" "integration_test" TRUE "Running integration tests via ctest (verbose)")
    # cmake-format: on
endfunction()

# Finalizes and creates all of the test targets
#
# Usage:
#   finalize_test_targets(<prefix_name> [ENABLE_COVERAGE])
#
# Arguments:
#   prefix_name      - Prefix for all target names (e.g., "miopen-provider" creates "miopen-provider-check")
#   ENABLE_COVERAGE  - Optional flag to enable code coverage profraw collection in the test environment
function(finalize_test_targets prefix_name)
    cmake_parse_arguments(ARG "ENABLE_COVERAGE" "" "" ${ARGN})

    _create_test_name_validation_target_internal(${prefix_name})

    _create_ctest_targets_internal(${prefix_name} ${ARG_ENABLE_COVERAGE})

    # cmake-format: off
    set(CREATE_ALIASES FALSE)
    if(NOT ROCM_LIBS_SUPERBUILD)
        set(CREATE_ALIASES TRUE)
    endif()

    add_custom_target(${prefix_name}-check DEPENDS ${prefix_name}-check_ctest COMMENT "Running all tests via ctest")
    add_custom_target(${prefix_name}-unit-check DEPENDS ${prefix_name}-unit-check_ctest COMMENT "Running unit tests via ctest")
    add_custom_target(${prefix_name}-integration-check DEPENDS ${prefix_name}-integration-check_ctest COMMENT "Running integration tests via ctest")
    message(STATUS "Created ctest targets: ${prefix_name}-check, ${prefix_name}-unit-check, ${prefix_name}-integration-check")

    add_custom_target(${prefix_name}-check-verbose DEPENDS ${prefix_name}-check_ctest-verbose COMMENT "Running all tests via ctest (verbose)")
    add_custom_target(${prefix_name}-unit-check-verbose DEPENDS ${prefix_name}-unit-check_ctest-verbose COMMENT "Running unit tests via ctest (verbose)")
    add_custom_target(${prefix_name}-integration-check-verbose DEPENDS ${prefix_name}-integration-check_ctest-verbose COMMENT "Running integration tests via ctest (verbose)")
    message(STATUS "Created ctest verbose targets: ${prefix_name}-check-verbose, ${prefix_name}-unit-check-verbose, ${prefix_name}-integration-check-verbose")

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
endfunction()

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

    set_target_properties(
        ${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
    )

    set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_TEST_TARGETS ${TARGET})

    set_target_properties(
        ${TARGET}
        PROPERTIES
            INSTALL_RPATH
            "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR};\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/hipdnn_plugins/engines"
            INSTALL_RPATH_USE_LINK_PATH TRUE
            BUILD_RPATH_USE_ORIGIN TRUE
    )

    install(TARGETS ${TARGET} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    set(ALL_LABELS ${APPEND_FUNCTION_SUFFIX})
    if(EXTRA_LABELS)
        list(APPEND ALL_LABELS ${EXTRA_LABELS})
    endif()

    add_test(NAME ${TARGET} COMMAND ${TARGET} WORKING_DIRECTORY ${WORKING_DIR})
    set_tests_properties(${TARGET} PROPERTIES LABELS "${ALL_LABELS}")
endfunction()

# ~~~
# Adds a unit test target
#
# Usage:
#   add_unit_test_target(TARGET WORKING_DIR [LABELS label1 label2 ...])
# ~~~
function(add_unit_test_target TARGET WORKING_DIR)
    cmake_parse_arguments(ARG "" "" "LABELS" ${ARGN})
    _add_test_target_internal(unit_test ${TARGET} ${WORKING_DIR} ${ARG_LABELS})
endfunction()

# ~~~
# Adds an integration test target
#
# Usage:
#   add_integration_test_target(TARGET WORKING_DIR [LABELS label1 label2 ...])
# ~~~
function(add_integration_test_target TARGET WORKING_DIR)
    cmake_parse_arguments(ARG "" "" "LABELS" ${ARGN})
    _add_test_target_internal(integration_test ${TARGET} ${WORKING_DIR} ${ARG_LABELS})
endfunction()

# ~~~
# Adds a tiered test target with Smoke/Standard/Comprehensive/Full ctest entries.
#
# Use this instead of add_unit_test_target() for test binaries that use GTest
# prefix-based tier filtering (INSTANTIATE_TEST_SUITE_P with Smoke/Standard/
# Comprehensive/Full prefixes).  Creates four ctest entries with appropriate
# exclusion/inclusion filters, cumulative labels, and per-tier timeouts.
# The smoke-only entry is accumulated for install staging so TheRock CI
# (which runs bare ctest with no -L filter) only executes quick tests.
#
# Usage:
#   add_tiered_test_target(TARGET WORKING_DIR
#       [SMOKE_TIMEOUT seconds]          # default 600
#       [STANDARD_TIMEOUT seconds]       # default 1800
#       [COMPREHENSIVE_TIMEOUT seconds]  # default 3600
#       [FULL_TIMEOUT seconds])          # default 7200
# ~~~
function(add_tiered_test_target TARGET WORKING_DIR)
    cmake_parse_arguments(ARG ""
        "SMOKE_TIMEOUT;STANDARD_TIMEOUT;COMPREHENSIVE_TIMEOUT;FULL_TIMEOUT" "" ${ARGN})

    # Default timeouts
    if(NOT ARG_SMOKE_TIMEOUT)
        set(ARG_SMOKE_TIMEOUT 600)
    endif()
    if(NOT ARG_STANDARD_TIMEOUT)
        set(ARG_STANDARD_TIMEOUT 1800)
    endif()
    if(NOT ARG_COMPREHENSIVE_TIMEOUT)
        set(ARG_COMPREHENSIVE_TIMEOUT 3600)
    endif()
    if(NOT ARG_FULL_TIMEOUT)
        set(ARG_FULL_TIMEOUT 7200)
    endif()

    set(TARGET_EXE "${TARGET}${CMAKE_EXECUTABLE_SUFFIX}")

    message(STATUS "Adding tiered test target: ${TARGET} -> ${TARGET_EXE}")

    # -- Infra setup (same as _add_test_target_internal, without the unfiltered add_test) --
    set(CHECK_DEPENDS_GLOBAL ${CHECK_DEPENDS_GLOBAL} ${TARGET}
        CACHE INTERNAL "Accumulated global dependencies for test name validation" FORCE)
    set(CHECK_EXECUTABLE_PATHS_GLOBAL ${CHECK_EXECUTABLE_PATHS_GLOBAL}
        "${CMAKE_INSTALL_BINDIR}/${TARGET_EXE}"
        CACHE INTERNAL "Accumulated global check executable paths" FORCE)

    set_target_properties(${TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
        INSTALL_RPATH
            "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR};\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/hipdnn_plugins/engines"
        INSTALL_RPATH_USE_LINK_PATH TRUE
        BUILD_RPATH_USE_ORIGIN TRUE)
    install(TARGETS ${TARGET} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    # -- Four ctest entries with cumulative labels --
    # Each tier gets a FAIL_REGULAR_EXPRESSION guard.  GTest prints "Running 0
    # tests from 0 test suites" and exits 0 when no tests match a filter — the
    # guard turns that silent pass into a ctest failure so accidentally empty
    # tiers are caught early.  If a tier is intentionally empty, add a single
    # INSTANTIATE_TEST_SUITE_P with a minimal case rather than removing the guard.
    set(_no_tests_re "Running 0 tests from 0 test suites")

    # Smoke: catch-all exclusion (everything not Standard/Comprehensive/Full).
    # The "unit_test" label is intentional — smoke tests are quick enough to
    # run in the unit-check target alongside real unit tests.
    add_test(NAME ${TARGET}_quick
        COMMAND ${TARGET} --gtest_filter=-Standard*:Comprehensive*:Full*
        WORKING_DIRECTORY ${WORKING_DIR})
    set_tests_properties(${TARGET}_quick PROPERTIES
        LABELS "quick;standard;comprehensive;full;unit_test" TIMEOUT ${ARG_SMOKE_TIMEOUT}
        FAIL_REGULAR_EXPRESSION "${_no_tests_re}")

    add_test(NAME ${TARGET}_standard
        COMMAND ${TARGET} --gtest_filter=Standard*
        WORKING_DIRECTORY ${WORKING_DIR})
    set_tests_properties(${TARGET}_standard PROPERTIES
        LABELS "standard;comprehensive;full" TIMEOUT ${ARG_STANDARD_TIMEOUT}
        FAIL_REGULAR_EXPRESSION "${_no_tests_re}")

    add_test(NAME ${TARGET}_comprehensive
        COMMAND ${TARGET} --gtest_filter=Comprehensive*
        WORKING_DIRECTORY ${WORKING_DIR})
    set_tests_properties(${TARGET}_comprehensive PROPERTIES
        LABELS "comprehensive;full" TIMEOUT ${ARG_COMPREHENSIVE_TIMEOUT}
        FAIL_REGULAR_EXPRESSION "${_no_tests_re}")

    add_test(NAME ${TARGET}_full
        COMMAND ${TARGET} --gtest_filter=Full*
        WORKING_DIRECTORY ${WORKING_DIR})
    set_tests_properties(${TARGET}_full PROPERTIES
        LABELS "full" TIMEOUT ${ARG_FULL_TIMEOUT}
        FAIL_REGULAR_EXPRESSION "${_no_tests_re}")

    # -- Install staging: smoke only --
    # Accumulated in a global property so install_integration_tests_ctest_files()
    # can emit all tiered entries automatically.
    set_property(GLOBAL APPEND_STRING PROPERTY TIERED_TEST_INSTALL_STAGING
        "add_test(${TARGET}_quick \"../${TARGET_EXE}\" --gtest_filter=-Standard*:Comprehensive*:Full*)\nset_tests_properties(${TARGET}_quick PROPERTIES LABELS \"quick\" TIMEOUT ${ARG_SMOKE_TIMEOUT})\n")
endfunction() # add_tiered_test_target

# Install CTest configuration files for direct test execution. This should be called once at the end
# of the main CMakeLists.txt after all tests are registered.
#
# Usage:
#   install_provider_ctest_files(<install_subdir>)
#
# Parameters:
#   INSTALL_SUBDIR - Subdirectory under CMAKE_INSTALL_BINDIR for the CTestTestfile.cmake
function(install_provider_ctest_files INSTALL_SUBDIR)
    set(CTEST_INSTALL_PATH "${CMAKE_INSTALL_BINDIR}/${INSTALL_SUBDIR}")

    set(INSTALLED_CTEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake.install")

    file(WRITE "${INSTALLED_CTEST_FILE}"
         "# Autogenerated CTestTestfile for installed ${INSTALL_SUBDIR} tests\n"
    )
    file(APPEND "${INSTALLED_CTEST_FILE}" "# Generated by ${PROJECT_NAME} build system\n\n")

    get_property(all_tests GLOBAL PROPERTY ${PROJECT_NAME}_TEST_TARGETS)

    foreach(test_target ${all_tests})
        file(APPEND "${INSTALLED_CTEST_FILE}" "add_test(${test_target} \"../${test_target}\")\n")
    endforeach()

    # Append tiered test entries (smoke tier only for CI).
    # These are accumulated by add_tiered_test_target() calls.
    get_property(_tiered_staging GLOBAL PROPERTY TIERED_TEST_INSTALL_STAGING)
    if(_tiered_staging)
        file(APPEND "${INSTALLED_CTEST_FILE}" "\n# Tiered test entries (smoke tier only for CI)\n")
        file(APPEND "${INSTALLED_CTEST_FILE}" "${_tiered_staging}")
    endif()

    install(FILES "${INSTALLED_CTEST_FILE}"
            DESTINATION ${CTEST_INSTALL_PATH} RENAME CTestTestfile.cmake
    )
endfunction()
