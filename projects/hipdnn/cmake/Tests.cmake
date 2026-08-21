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

# YAML-driven CTest categorisation. Build-tree tests get labels as they are
# registered; install_hipdnn_ctest_files() bakes the same labels into the
# installed CTestTestfile.cmake.
set(_HIPDNN_TEST_CATEGORIES_YAML "${PROJECT_SOURCE_DIR}/test_categories.yaml")
set(_HIPDNN_SHARED_CTEST "${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
if(EXISTS "${_HIPDNN_SHARED_CTEST}" AND EXISTS "${_HIPDNN_TEST_CATEGORIES_YAML}")
    include("${_HIPDNN_SHARED_CTEST}")
    message(STATUS "hipDNN: YAML-based CTest categorization enabled")
else()
    if(NOT EXISTS "${_HIPDNN_SHARED_CTEST}")
        message(STATUS
            "hipDNN: shared/ctest not found at ${_HIPDNN_SHARED_CTEST}; skipping CTest categories"
        )
    endif()
    if(NOT EXISTS "${_HIPDNN_TEST_CATEGORIES_YAML}")
        message(STATUS
            "hipDNN: ${_HIPDNN_TEST_CATEGORIES_YAML} not found; skipping CTest categories"
        )
    endif()
endif()

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

# Applies YAML-defined CTest category labels to explicit test names.
#
# Arguments:
#   ARGN - One or more CTest test names to label using _HIPDNN_TEST_CATEGORIES_YAML
function(_apply_hipdnn_test_category_labels)
    if(COMMAND apply_ctest_category_labels)
        apply_ctest_category_labels(
            "${_HIPDNN_TEST_CATEGORIES_YAML}"
            EXPLICIT_TESTS ${ARGN}
        )
    endif()
endfunction() # _apply_hipdnn_test_category_labels


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
        _apply_hipdnn_test_category_labels(${prefix_name}_test_name_validation)
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

# On ASAN builds (HIPDNN_TEST_MIOPEN_CACHE_DIR set by Sanitizers.cmake), a fixture wipes the build-local
# MIOpen cache once per ctest run; tests opt in via FIXTURES_REQUIRED (see add_hipdnn_test). The
# GLOBAL-property guard defines it once across the add_subdirectory'd build tree; CTest matches the
# setup to requiring tests in any directory.
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
function(_add_check_target_internal PREFIX_NAME TARGET_NAME LABEL VERBOSE COMMENT)
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
endfunction() # _add_check_target_internal

# Internal helper function to create the ninja-check targets for running tests via ctest
function(_create_check_targets_internal prefix_name)
    # cmake-format: off
    # Build test environment once for all ctest targets
    _build_test_environment_list_internal(CTEST_ENV)

    # Regular all-test targets (without --verbose)
    _add_check_target_internal(${prefix_name} "check_ctest" "" FALSE "Running all tests via ctest")

    # Verbose all-test targets
    _add_check_target_internal(${prefix_name} "check_ctest-verbose" "" TRUE "Running all tests via ctest (verbose)")

    if(COMMAND get_ctest_category_names)
        get_ctest_category_names("${_HIPDNN_TEST_CATEGORIES_YAML}" HIPDNN_TEST_CATEGORIES)
    else()
        set(HIPDNN_TEST_CATEGORIES "")
    endif()

    set(HIPDNN_TEST_CATEGORIES "${HIPDNN_TEST_CATEGORIES}" PARENT_SCOPE)

    foreach(_category IN LISTS HIPDNN_TEST_CATEGORIES)
        if(NOT _category MATCHES "^[A-Za-z0-9_.+-]+$")
            message(FATAL_ERROR "Invalid hipDNN test category '${_category}'. Category names must be valid CMake target-name fragments.")
        endif()
        if(_category MATCHES "^check(-verbose)?$")
            message(FATAL_ERROR "Invalid hipDNN test category '${_category}'. Category name is reserved.")
        endif()

        _add_check_target_internal(${prefix_name} "${_category}-check_ctest" "${_category}" FALSE "Running ${_category} tests via ctest")
        _add_check_target_internal(${prefix_name} "${_category}-check_ctest-verbose" "${_category}" TRUE "Running ${_category} tests via ctest (verbose)")
    endforeach()
    # cmake-format: on
endfunction() # _create_check_targets_internal



# Registers the cache-key generator's own unit tests as a ctest test. The generated
# header's runtime behaviour is covered by the C++ suites; this covers the generator's
# field policy, so a change to it fails here rather than silently reshaping the key.
#
# The policy under test belongs to the schemas, so this runs regardless of
# HIPDNN_ENABLE_KERNEL_INGESTOR.
function(_create_cache_key_codegen_test_internal prefix_name)
    if(Python3_FOUND)
        add_test(
            NAME ${prefix_name}_cache_key_codegen_tests
            COMMAND ${Python3_EXECUTABLE} -m unittest discover -s
                    ${PROJECT_SOURCE_DIR}/scripts -p "test_gen_cache_key.py" -v
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/scripts
        )
        _apply_hipdnn_test_category_labels(${prefix_name}_cache_key_codegen_tests)
    endif() # Python3_FOUND
endfunction() # _create_cache_key_codegen_test_internal

# Finalizes and creates all of the test targets
#
# Arguments:
#   prefix_name - Prefix to add to all target names (e.g., "hipdnn" creates "hipdnn-check")
#
# Creates prefixed targets from the YAML categories (e.g., "hipdnn-quick-check").
# In standalone builds (non-superbuild), also creates unprefixed aliases for backward compatibility.
function(finalize_test_targets prefix_name)
    _create_test_name_validation_target_internal(${prefix_name})
    _create_cache_key_codegen_test_internal(${prefix_name})

    _create_check_targets_internal(${prefix_name})

    # cmake-format: off
    # Determine if we should create legacy aliases (only in standalone builds)
    set(CREATE_ALIASES FALSE)
    if(NOT ROCM_LIBS_SUPERBUILD)
        set(CREATE_ALIASES TRUE)
    endif()

    # Create all-tests targets.
    add_custom_target(${prefix_name}-check DEPENDS ${prefix_name}-check_ctest COMMENT "Running all tests via ctest")
    add_custom_target(${prefix_name}-check-verbose DEPENDS ${prefix_name}-check_ctest-verbose COMMENT "Running all tests via ctest (verbose)")

    if(CREATE_ALIASES)
        add_custom_target(check DEPENDS ${prefix_name}-check COMMENT "Alias for ${prefix_name}-check")
        add_custom_target(check-verbose DEPENDS ${prefix_name}-check-verbose COMMENT "Alias for ${prefix_name}-check-verbose")
    endif()

    foreach(_category IN LISTS HIPDNN_TEST_CATEGORIES)
        add_custom_target(${prefix_name}-${_category}-check DEPENDS ${prefix_name}-${_category}-check_ctest COMMENT "Running ${_category} tests via ctest")
        add_custom_target(${prefix_name}-${_category}-check-verbose DEPENDS ${prefix_name}-${_category}-check_ctest-verbose COMMENT "Running ${_category} tests via ctest (verbose)")

        if(CREATE_ALIASES)
            add_custom_target(${_category}-check DEPENDS ${prefix_name}-${_category}-check COMMENT "Alias for ${prefix_name}-${_category}-check")
            add_custom_target(${_category}-check-verbose DEPENDS ${prefix_name}-${_category}-check-verbose COMMENT "Alias for ${prefix_name}-${_category}-check-verbose")
        endif()
    endforeach()
    # cmake-format: on
endfunction() # finalize_test_targets

# ~~~
# Records, configures, and registers a hipDNN gtest-based CTest test target. Assumes that the
# test target is a gtest executable, setting up:
# - Test name validation tracking (adds to global dependency and executable path lists)
# - RPATH settings for relocatable test executables
# - Installation rules for test binaries
# - CTest registration
# - YAML-driven category labels from projects/hipdnn/test_categories.yaml
#
# Parameters:
#   TARGET - Name of the test executable target (must already exist)
#   WORKING_DIR - Working directory for test execution
# ~~~
function(add_hipdnn_test TARGET WORKING_DIR)
    set(TARGET_EXE ${TARGET})

    # Add executable suffix if needed (e.g., .exe on Windows)
    if(CMAKE_EXECUTABLE_SUFFIX)
        set(TARGET_EXE "${TARGET_EXE}${CMAKE_EXECUTABLE_SUFFIX}")
    endif()

    message(STATUS "Registering test target: ${TARGET} -> ${TARGET_EXE} in working directory: ${WORKING_DIR}")

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

    # Make test executables prefer the build/install tree next to the executable before any
    # toolchain paths. This prevents stale installed hipDNN libraries in /opt/rocm from shadowing
    # the freshly built test dependencies.
    set_target_properties(
        ${TARGET}
        PROPERTIES
            INSTALL_RPATH
            "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR};\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/hipdnn_plugins/engines"
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH_USE_LINK_PATH TRUE
    )

    # Install test executables to bin directory
    install(TARGETS ${TARGET} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    # On Windows, stage the shadowed ROCm DLLs before this test binary is built so a
    # partial build + manual ctest doesn't load the stale System32 amd_comgr.dll.
    if(TARGET stage_shadowed_rocm_dlls)
        add_dependencies(${TARGET} stage_shadowed_rocm_dlls)
    endif()

    add_test(NAME ${TARGET} COMMAND ${TARGET} WORKING_DIRECTORY ${WORKING_DIR})
    _apply_hipdnn_test_category_labels(${TARGET})
    if(DEFINED TEST_ENVIRONMENT)
        set_tests_properties(${TARGET} PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}")
    endif()
    # PATH prepends (e.g. the Windows ASAN runtime dir) are applied via ENVIRONMENT_MODIFICATION
    # rather than ENVIRONMENT so the runtime PATH is extended, not replaced.
    if(DEFINED TEST_ENVIRONMENT_MODIFICATION)
        set_tests_properties(${TARGET} PROPERTIES
            ENVIRONMENT_MODIFICATION "${TEST_ENVIRONMENT_MODIFICATION}")
    endif()
    # Require the build-local MIOpen cache to be cleared before this test runs (ASAN builds only;
    # HIPDNN_TEST_MIOPEN_CACHE_DIR is unset otherwise). See the hipdnn_clear_miopen_test_cache fixture.
    if(DEFINED HIPDNN_TEST_MIOPEN_CACHE_DIR)
        set_tests_properties(${TARGET} PROPERTIES FIXTURES_REQUIRED hipdnn_clear_miopen_test_cache)
    endif()
endfunction() # add_hipdnn_test

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

    # Bake the YAML-driven category labels into the installed
    # CTestTestfile.cmake so `ctest --test-dir $THEROCK_BIN_DIR/hipdnn -L
    # <tier>` works against the install tree.
    #
    # Passing INSTALLED_CTEST_FILE as the 2nd argument signals the shared
    # helper that the snippet will be evaluated by ctest's own script
    # interpreter (which does not implement
    # get_property(DIRECTORY ... PROPERTY TESTS)), so it emits explicit
    # per-test set_property() lines after auto-discovering the test
    # names from the add_test() lines we just wrote above.
    if(COMMAND apply_ctest_category_labels AND all_tests)
        apply_ctest_category_labels(
            "${_HIPDNN_TEST_CATEGORIES_YAML}"
            "${INSTALLED_CTEST_FILE}"
        )
    endif()

    # Install the generated CTestTestfile.cmake to HIPDNN_CTEST_FILE_INSTALL_PATH
    install(FILES "${INSTALLED_CTEST_FILE}" DESTINATION ${HIPDNN_CTEST_FILE_INSTALL_PATH}
            RENAME CTestTestfile.cmake
    )

endfunction() # install_hipdnn_ctest_files
