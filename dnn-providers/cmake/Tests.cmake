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

# Applies YAML-defined CTest category labels to explicit test names when a
# provider supplies DNN_PROVIDER_CTEST_CATEGORIES_YAML.
#
# Arguments:
#   ARGN - One or more CTest test names to label
function(_apply_provider_ctest_category_labels)
    if(COMMAND apply_ctest_category_labels AND DNN_PROVIDER_CTEST_CATEGORIES_YAML)
        apply_ctest_category_labels(
            "${DNN_PROVIDER_CTEST_CATEGORIES_YAML}"
            EXPLICIT_TESTS ${ARGN}
        )
    endif()
endfunction()

# Reads all YAML category names that should become provider check targets.
#
# Arguments:
#   out_var - Variable to receive the de-duplicated category list
function(_get_provider_ctest_category_names out_var)
    set(_categories "")
    foreach(_yaml IN LISTS DNN_PROVIDER_TEST_CATEGORY_YAMLS)
        if(COMMAND get_ctest_category_names)
            get_ctest_category_names("${_yaml}" _yaml_categories)
            list(APPEND _categories ${_yaml_categories})
        endif()
    endforeach()
    if(_categories)
        list(REMOVE_DUPLICATES _categories)
    endif()
    set(${out_var} "${_categories}" PARENT_SCOPE)
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

        # Also register as a ctest test so it runs with ctest and appears in test results
        add_test(
            NAME ${prefix_name}_test_name_validation
            COMMAND ${Python3_EXECUTABLE}
                ${_TEST_NAME_VALIDATOR_SCRIPT}
                --test-executables ${TEST_EXECUTABLES_FILE}
                --build-dir ${CMAKE_BINARY_DIR}
                --strict
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
        if(DNN_PROVIDER_CTEST_CATEGORIES_YAML)
            _apply_provider_ctest_category_labels(${prefix_name}_test_name_validation)
        else()
            set_tests_properties(${prefix_name}_test_name_validation PROPERTIES LABELS "unit_test;integration_test;quick")
        endif()
    else()
        add_custom_target(
            ${prefix_name}-validate_test_names COMMAND ${CMAKE_COMMAND} -E echo
                                        "Test name validation skipped"
            COMMENT "Skipping test name validation"
        )
    endif()
endfunction()

enable_testing()

# On ASAN builds (HIPDNN_TEST_MIOPEN_CACHE_DIR set by Sanitizers.cmake), a fixture wipes the build-local
# MIOpen cache once per ctest run; tests opt in via FIXTURES_REQUIRED (see the registration helpers
# below). The GLOBAL-property guard defines it once across the add_subdirectory'd build tree; CTest
# matches the setup to requiring tests in any directory.
# A scope block keeps the intermediate path variables out of the including project's scope; the
# fixture registration (add_test / set_property GLOBAL) is not variable-scoped, so it escapes.
block(SCOPE_FOR VARIABLES)
    get_property(_fixture_added GLOBAL PROPERTY _hipdnn_clear_miopen_test_cache_fixture_added)
    if(DEFINED HIPDNN_TEST_MIOPEN_CACHE_DIR AND NOT _fixture_added)
        # Safety: this path is baked into a generated `rm -rf`, so require it to be strictly under the build tree.
        get_filename_component(_cache_dir_abs "${HIPDNN_TEST_MIOPEN_CACHE_DIR}" ABSOLUTE)
        get_filename_component(_binary_dir_abs "${CMAKE_BINARY_DIR}" ABSOLUTE)
        string(FIND "${_cache_dir_abs}" "${_binary_dir_abs}/" _binary_dir_prefix_pos)
        if(NOT _binary_dir_prefix_pos EQUAL 0)
            message(FATAL_ERROR
                "HIPDNN_TEST_MIOPEN_CACHE_DIR ('${HIPDNN_TEST_MIOPEN_CACHE_DIR}') must be a subdirectory "
                "of the build tree ('${CMAKE_BINARY_DIR}'); abort during hipdnn miopen cache clear fixture registration.")
        endif()
        set_property(GLOBAL PROPERTY _hipdnn_clear_miopen_test_cache_fixture_added TRUE)
        add_test(NAME hipdnn_clear_miopen_test_cache
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${HIPDNN_TEST_MIOPEN_CACHE_DIR}")
        set_tests_properties(hipdnn_clear_miopen_test_cache
            PROPERTIES FIXTURES_SETUP hipdnn_clear_miopen_test_cache)
    endif()
endblock()

# Internal helper function to create a ctest target
# ~~~
# Parameters:
#   PREFIX_NAME - Prefix for target names
#   TARGET_NAME - Name of the ctest target to create (will be prefixed)
#   LABEL - Optional label filter for ctest (empty string for no filter)
#   VERBOSE - Set to TRUE to add --verbose flag, FALSE otherwise
#   COMMENT - Comment describing the target
# ~~~
# Adds dependencies that must be built before invoking ctest check targets.
#   DNN_PROVIDER_CHECK_EXCLUDE_LABELS - Optional provider labels to exclude from
#                                      generated check targets
#
#
# Providers can set DNN_PROVIDER_TEST_RUN_DEPENDS to executables/plugins that
# CTest entries execute, including generated YAML category suites.
function(_add_ctest_target_internal PREFIX_NAME TARGET_NAME LABEL VERBOSE COMMENT)
    set(CTEST_CMD ${CMAKE_COMMAND} -E env ${CTEST_ENV} ${CMAKE_CTEST_COMMAND})
    if(NOT "${LABEL}" STREQUAL "")
        list(APPEND CTEST_CMD -L "^${LABEL}$")
    endif()

    foreach(exclude_label IN LISTS DNN_PROVIDER_CHECK_EXCLUDE_LABELS)
        list(APPEND CTEST_CMD -LE "^${exclude_label}$")
    endforeach()

    list(APPEND CTEST_CMD --output-on-failure)

    if(VERBOSE)
        list(APPEND CTEST_CMD --verbose)
    endif()

    list(APPEND CTEST_CMD -C ${CMAKE_CFG_INTDIR})

    set(FULL_TARGET_NAME "${PREFIX_NAME}-${TARGET_NAME}")
    add_custom_target(${FULL_TARGET_NAME} COMMAND ${CTEST_CMD} COMMENT "${COMMENT}" USES_TERMINAL)
    add_dependencies(${FULL_TARGET_NAME} ${PREFIX_NAME}-validate_test_names)
    if(ROCM_LIBS_SUPERBUILD AND DNN_PROVIDER_TEST_RUN_DEPENDS)
        add_dependencies(${FULL_TARGET_NAME} ${DNN_PROVIDER_TEST_RUN_DEPENDS})
    endif()
    message(VERBOSE "Created ${FULL_TARGET_NAME} target")
endfunction()

# Internal helper function to create the check targets for running tests via ctest
function(_create_ctest_targets_internal prefix_name coverage)
    # cmake-format: off
    _build_test_environment_list_internal(CTEST_ENV ${coverage})

    _add_ctest_target_internal(${prefix_name} "check_ctest" "" FALSE "Running all tests via ctest")
    _add_ctest_target_internal(${prefix_name} "check_ctest-verbose" "" TRUE "Running all tests via ctest (verbose)")

    if(DNN_PROVIDER_TEST_CATEGORY_YAMLS)
        _get_provider_ctest_category_names(DNN_PROVIDER_TEST_CATEGORIES)
    else()
        set(DNN_PROVIDER_TEST_CATEGORIES unit integration)
        _add_ctest_target_internal(${prefix_name} "unit-check_ctest" "unit_test" FALSE "Running unit tests via ctest")
        _add_ctest_target_internal(${prefix_name} "integration-check_ctest" "integration_test" FALSE "Running integration tests via ctest")
        _add_ctest_target_internal(${prefix_name} "unit-check_ctest-verbose" "unit_test" TRUE "Running unit tests via ctest (verbose)")
        _add_ctest_target_internal(${prefix_name} "integration-check_ctest-verbose" "integration_test" TRUE "Running integration tests via ctest (verbose)")
    endif()

    foreach(_category IN LISTS DNN_PROVIDER_TEST_CATEGORIES)
        if(NOT _category MATCHES "^[A-Za-z0-9_.+-]+$")
            message(FATAL_ERROR "Invalid ${prefix_name} test category '${_category}'. Category names must be valid CMake target-name fragments.")
        endif()
        if(_category MATCHES "^check(-verbose)?$")
            message(FATAL_ERROR "Invalid ${prefix_name} test category '${_category}'. Category name is reserved.")
        endif()
        if(DNN_PROVIDER_TEST_CATEGORY_YAMLS)
            _add_ctest_target_internal(${prefix_name} "${_category}-check_ctest" "${_category}" FALSE "Running ${_category} tests via ctest")
            _add_ctest_target_internal(${prefix_name} "${_category}-check_ctest-verbose" "${_category}" TRUE "Running ${_category} tests via ctest (verbose)")
        endif()
    endforeach()

    set(DNN_PROVIDER_TEST_CATEGORIES "${DNN_PROVIDER_TEST_CATEGORIES}" PARENT_SCOPE)
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
    add_custom_target(${prefix_name}-check-verbose DEPENDS ${prefix_name}-check_ctest-verbose COMMENT "Running all tests via ctest (verbose)")

    if(CREATE_ALIASES)
        add_custom_target(check DEPENDS ${prefix_name}-check COMMENT "Alias for ${prefix_name}-check")
        add_custom_target(check-verbose DEPENDS ${prefix_name}-check-verbose COMMENT "Alias for ${prefix_name}-check-verbose")
    endif()

    foreach(_category IN LISTS DNN_PROVIDER_TEST_CATEGORIES)
        add_custom_target(${prefix_name}-${_category}-check DEPENDS ${prefix_name}-${_category}-check_ctest COMMENT "Running ${_category} tests via ctest")
        add_custom_target(${prefix_name}-${_category}-check-verbose DEPENDS ${prefix_name}-${_category}-check_ctest-verbose COMMENT "Running ${_category} tests via ctest (verbose)")

        if(CREATE_ALIASES)
            add_custom_target(${_category}-check DEPENDS ${prefix_name}-${_category}-check COMMENT "Alias for ${prefix_name}-${_category}-check")
            add_custom_target(${_category}-check-verbose DEPENDS ${prefix_name}-${_category}-check-verbose COMMENT "Alias for ${prefix_name}-${_category}-check-verbose")
        endif()
    endforeach()
    # cmake-format: on
endfunction()

# ~~~
# Internal helper function to record, configure, and register a ctest test target. Assumes that the
# test target is a gtest executable, setting up:
# - Test name validation tracking (adds to global dependency and executable path lists)
# - RPATH settings for relocatable test executables
# - Installation rules for test binaries
# - CTest registration
#   YAML-driven category labels when DNN_PROVIDER_TEST_CATEGORY_YAMLS is set,
#   otherwise legacy labels such as unit_test/integration_test
#
#   APPEND_FUNCTION_SUFFIX - Legacy grouping name retained by add_unit_test_target/add_integration_test_target
#   TARGET - Name of the test executable target (must already exist)
#   WORKING_DIR - Working directory for test execution
#   LABELS - (Optional) Additional labels to apply in legacy label mode
#   ENVIRONMENT - (Optional) Extra ENVIRONMENT entries (KEY=VALUE) for this
#       target, merged with the ambient TEST_ENVIRONMENT. Applied directly
#       via set_tests_properties() in legacy label mode. When YAML-driven
#       categorization is active, this function never reaches add_test()
#       (see below), so the merged result is instead published as
#       <TARGET>_TEST_ENVIRONMENT (PARENT_SCOPE) for the caller to forward
#       into whichever suites actually get registered later.
# ~~~
function(_add_test_target_internal APPEND_FUNCTION_SUFFIX TARGET WORKING_DIR)
    cmake_parse_arguments(ARG "" "" "LABELS;ENVIRONMENT" ${ARGN})
    set(EXTRA_LABELS ${ARG_LABELS})
    set(TARGET_EXE ${TARGET})

    if(CMAKE_EXECUTABLE_SUFFIX)
        set(TARGET_EXE "${TARGET_EXE}${CMAKE_EXECUTABLE_SUFFIX}")
    endif()

    message(STATUS "Registering ${APPEND_FUNCTION_SUFFIX} test target: ${TARGET} -> ${TARGET_EXE} in working directory: ${WORKING_DIR}")

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

    # On Windows, stage the shadowed ROCm DLLs (amd_comgr.dll) before this test binary
    # is built, so a partial build (`cmake --build --target ${TARGET}`) + manual ctest doesn't load
    # the stale System32 copy. Placed before the YAML early-return so it also covers the
    # apply_test_category_labels() suites, which invoke this same executable target.
    if(TARGET stage_shadowed_rocm_dlls)
        add_dependencies(${TARGET} stage_shadowed_rocm_dlls)
    endif()
    set(_MERGED_TEST_ENVIRONMENT ${TEST_ENVIRONMENT} ${ARG_ENVIRONMENT})

    # YAML-driven categorization (apply_test_category_labels(), keyed off
    # DNN_PROVIDER_TEST_CATEGORY_YAMLS) generates its own tiered suites after this
    # function returns; registering the raw, unfiltered ${TARGET} test here would
    # just duplicate the *_full_suite entry with zero labels (never selectable via
    # `ctest -L`, always run by a bare `ctest`).
    #
    # Providers with a pre-registered external CTest-name suite also set
    # DNN_PROVIDER_CTEST_CATEGORIES_YAML for apply_ctest_category_labels(), but every
    # existing caller (miopen-provider, hipblaslt-provider, hip-kernel-provider) folds
    # that same YAML path into DNN_PROVIDER_TEST_CATEGORY_YAMLS too, so checking only
    # the latter covers both GTest-filter-only projects (integration-tests) and
    # providers with an external CTest-name YAML.
    #
    # Callers cannot set properties on ${TARGET} below since it was never
    # registered as a CTest test in this mode -- publish the merged
    # environment instead so the caller can forward it explicitly to
    # whichever suites apply_test_category_labels()/apply_ctest_category_labels()
    # actually creates.
    if(DNN_PROVIDER_TEST_CATEGORY_YAMLS)
        set(${TARGET}_TEST_ENVIRONMENT "${_MERGED_TEST_ENVIRONMENT}" PARENT_SCOPE)
        return()
    endif()

    add_test(NAME ${TARGET} COMMAND ${TARGET} WORKING_DIRECTORY ${WORKING_DIR})
    set(ALL_LABELS ${APPEND_FUNCTION_SUFFIX})
    if(EXTRA_LABELS)
        list(APPEND ALL_LABELS ${EXTRA_LABELS})
    endif()
    set_tests_properties(${TARGET} PROPERTIES LABELS "${ALL_LABELS}")

    if(_MERGED_TEST_ENVIRONMENT)
        set_tests_properties(${TARGET} PROPERTIES ENVIRONMENT "${_MERGED_TEST_ENVIRONMENT}")
    endif()
    # PATH prepends (e.g. the Windows ASAN runtime / ROCm / build DLL dirs) go through
    # ENVIRONMENT_MODIFICATION so the runtime PATH is extended, not replaced.
    if(TEST_ENVIRONMENT_MODIFICATION)
        set_tests_properties(${TARGET} PROPERTIES
            ENVIRONMENT_MODIFICATION "${TEST_ENVIRONMENT_MODIFICATION}")
    endif()
    # Clear the build-local MIOpen cache before this test runs (ASAN builds only). See the
    # hipdnn_clear_miopen_test_cache fixture above.
    if(HIPDNN_TEST_MIOPEN_CACHE_DIR)
        set_tests_properties(${TARGET} PROPERTIES FIXTURES_REQUIRED hipdnn_clear_miopen_test_cache)
    endif()
endfunction()

# ~~~
# Adds a unit test target
#
# Usage:
#   add_unit_test_target(TARGET WORKING_DIR [LABELS label1 label2 ...]
#                         [ENVIRONMENT KEY=VALUE ...])
#
# ENVIRONMENT is forwarded to _add_test_target_internal(); see its
# ENVIRONMENT parameter doc for how it is applied and, in YAML-categorized
# builds, published back as <TARGET>_TEST_ENVIRONMENT.
# ~~~
function(add_unit_test_target TARGET WORKING_DIR)
    cmake_parse_arguments(ARG "" "" "LABELS;ENVIRONMENT" ${ARGN})
    _add_test_target_internal(unit_test ${TARGET} ${WORKING_DIR} LABELS ${ARG_LABELS} ENVIRONMENT ${ARG_ENVIRONMENT})
    if(DEFINED ${TARGET}_TEST_ENVIRONMENT)
        set(${TARGET}_TEST_ENVIRONMENT "${${TARGET}_TEST_ENVIRONMENT}" PARENT_SCOPE)
    endif()
endfunction()

# ~~~
# Adds an integration test target
#
# Usage:
#   add_integration_test_target(TARGET WORKING_DIR [LABELS label1 label2 ...]
#                                [ENVIRONMENT KEY=VALUE ...])
#
# ENVIRONMENT is forwarded to _add_test_target_internal(); see its
# ENVIRONMENT parameter doc for how it is applied and, in YAML-categorized
# builds, published back as <TARGET>_TEST_ENVIRONMENT.
# ~~~
function(add_integration_test_target TARGET WORKING_DIR)
    cmake_parse_arguments(ARG "" "" "LABELS;ENVIRONMENT" ${ARGN})
    _add_test_target_internal(integration_test ${TARGET} ${WORKING_DIR} LABELS ${ARG_LABELS} ENVIRONMENT ${ARG_ENVIRONMENT})
    if(DEFINED ${TARGET}_TEST_ENVIRONMENT)
        set(${TARGET}_TEST_ENVIRONMENT "${${TARGET}_TEST_ENVIRONMENT}" PARENT_SCOPE)
    endif()
endfunction()

# Install CTest configuration files for direct test execution. This should be called once at the end
# of the main CMakeLists.txt after all tests are registered.
#
# Usage:
#   install_provider_ctest_files(<install_subdir> [TEST_CATEGORIES_YAML <yaml>])
#
# Parameters:
#   INSTALL_SUBDIR - Subdirectory under CMAKE_INSTALL_BINDIR for the CTestTestfile.cmake
#   TEST_CATEGORIES_YAML - Optional path to a test_categories.yaml. When given
#       (and apply_ctest_category_labels is available), tiered category labels
#       are applied to the generated install file so `ctest -L <tier>` works
#       from the installed tree.
function(install_provider_ctest_files INSTALL_SUBDIR)
    cmake_parse_arguments(ARG "" "TEST_CATEGORIES_YAML" "" ${ARGN})
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

    # Append external integration test entries (cross-provider suite).
    # These are accumulated by add_external_integration_test_target() calls
    # that pass INSTALL_SUBDIR matching the value passed here.
    get_property(_external_staging GLOBAL
        PROPERTY "EXTERNAL_TEST_INSTALL_STAGING_${INSTALL_SUBDIR}"
    )
    if(_external_staging)
        file(APPEND "${INSTALLED_CTEST_FILE}" "\n# External integration test entries (cross-provider suite)\n")
        file(APPEND "${INSTALLED_CTEST_FILE}" "${_external_staging}")
    endif()

    # Apply YAML-driven category labels to the generated install file (scans
    # its add_test() names). Must run before install() so the labeled file is
    # the one staged. No-op when no YAML is passed or the shared helper isn't
    # available (standalone / sparse checkout).
    if(ARG_TEST_CATEGORIES_YAML AND COMMAND apply_ctest_category_labels)
        apply_ctest_category_labels("${ARG_TEST_CATEGORIES_YAML}" "${INSTALLED_CTEST_FILE}")
    endif()

    install(FILES "${INSTALLED_CTEST_FILE}"
            DESTINATION ${CTEST_INSTALL_PATH} RENAME CTestTestfile.cmake
    )
endfunction()
