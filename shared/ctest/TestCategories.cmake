# This script reads the test categories YAML file and applies labels to CTest

# Find Python3 for running the parser script
find_package(Python3 COMPONENTS Interpreter)

# Parses optional arguments for apply_test_category_labels into the caller's scope.
#
# Arguments:
#   ARGN - Named or legacy positional optional arguments
# ~~~
function(_parse_test_category_optional_args)
    cmake_parse_arguments(
        ARG
        "USE_RTEST_DRIVER"
        "INSTALL_TEST_FILE;RESOURCE_GROUP;TEST_NAME_PREFIX;INSTALL_EXECUTABLE"
        "COMMAND_ARGS;INSTALL_COMMAND_ARGS;ADDITIONAL_LABELS;ENVIRONMENT;ENVIRONMENT_MODIFICATION;FIXTURES_REQUIRED"
        ${ARGN}
    )

    set(_install_test_file "${ARG_INSTALL_TEST_FILE}")
    set(_resource_group "${ARG_RESOURCE_GROUP}")
    set(_use_rtest_driver "${ARG_USE_RTEST_DRIVER}")
    if(ARG_UNPARSED_ARGUMENTS)
        list(LENGTH ARG_UNPARSED_ARGUMENTS _arg_count)
        if(NOT _install_test_file AND _arg_count GREATER 0)
            list(GET ARG_UNPARSED_ARGUMENTS 0 _install_test_file)
        endif()
        if(NOT _resource_group AND _arg_count GREATER 1)
            list(GET ARG_UNPARSED_ARGUMENTS 1 _resource_group)
        endif()
        if(NOT _use_rtest_driver)
            list(FIND ARG_UNPARSED_ARGUMENTS "--use-rtest-driver" _rtest_idx)
            if(NOT _rtest_idx EQUAL -1)
                set(_use_rtest_driver TRUE)
            endif()
        endif()
    endif()

    set(_TEST_CATEGORY_INSTALL_FILE "${_install_test_file}" PARENT_SCOPE)
    set(_TEST_CATEGORY_RESOURCE_GROUP "${_resource_group}" PARENT_SCOPE)
    set(_TEST_CATEGORY_USE_RTEST_DRIVER "${_use_rtest_driver}" PARENT_SCOPE)
    set(_TEST_CATEGORY_NAME_PREFIX "${ARG_TEST_NAME_PREFIX}" PARENT_SCOPE)
    set(_TEST_CATEGORY_INSTALL_EXECUTABLE "${ARG_INSTALL_EXECUTABLE}" PARENT_SCOPE)
    set(_TEST_CATEGORY_COMMAND_ARGS "${ARG_COMMAND_ARGS}" PARENT_SCOPE)
    set(_TEST_CATEGORY_INSTALL_COMMAND_ARGS "${ARG_INSTALL_COMMAND_ARGS}" PARENT_SCOPE)
    set(_TEST_CATEGORY_ADDITIONAL_LABELS "${ARG_ADDITIONAL_LABELS}" PARENT_SCOPE)
    set(_TEST_CATEGORY_ENVIRONMENT "${ARG_ENVIRONMENT}" PARENT_SCOPE)
    set(_TEST_CATEGORY_ENVIRONMENT_MODIFICATION "${ARG_ENVIRONMENT_MODIFICATION}" PARENT_SCOPE)
    set(_TEST_CATEGORY_FIXTURES_REQUIRED "${ARG_FIXTURES_REQUIRED}" PARENT_SCOPE)
endfunction()

# Appends parser args for generated GTest category suites.
#
# Arguments:
#   out_var - Variable to receive parser args
# ~~~
function(_build_test_category_parser_args out_var)
    set(extra_args "")
    if(_TEST_CATEGORY_RESOURCE_GROUP)
        list(APPEND extra_args "--resource-group" "${_TEST_CATEGORY_RESOURCE_GROUP}")
    endif()
    if(_TEST_CATEGORY_NAME_PREFIX)
        list(APPEND extra_args "--test-name-prefix" "${_TEST_CATEGORY_NAME_PREFIX}")
    endif()
    if(_TEST_CATEGORY_INSTALL_EXECUTABLE)
        list(APPEND extra_args "--install-executable" "${_TEST_CATEGORY_INSTALL_EXECUTABLE}")
    endif()
    foreach(command_arg IN LISTS _TEST_CATEGORY_COMMAND_ARGS)
        list(APPEND extra_args "--command-arg=${command_arg}")
    endforeach()
    foreach(install_command_arg IN LISTS _TEST_CATEGORY_INSTALL_COMMAND_ARGS)
        list(APPEND extra_args "--install-command-arg=${install_command_arg}")
    endforeach()
    foreach(additional_label IN LISTS _TEST_CATEGORY_ADDITIONAL_LABELS)
        list(APPEND extra_args "--additional-label" "${additional_label}")
    endforeach()
    foreach(extra_env_kv IN LISTS _TEST_CATEGORY_ENVIRONMENT)
        list(APPEND extra_args "--environment" "${extra_env_kv}")
    endforeach()
    foreach(extra_env_mod IN LISTS _TEST_CATEGORY_ENVIRONMENT_MODIFICATION)
        list(APPEND extra_args "--environment-modification" "${extra_env_mod}")
    endforeach()
    foreach(fixture IN LISTS _TEST_CATEGORY_FIXTURES_REQUIRED)
        list(APPEND extra_args "--fixtures-required" "${fixture}")
    endforeach()
    if(_TEST_CATEGORY_USE_RTEST_DRIVER)
        list(APPEND extra_args "--use-rtest-driver")
    endif()
    set(${out_var} "${extra_args}" PARENT_SCOPE)
endfunction()

# Validates common inputs for generated GTest category suites.
#
# Arguments:
#   target_name - GTest executable target name
#   yaml_file - Path to test_categories.yaml
#   working_dir - Working directory for test execution
#   parse_script - Parser script path to validate
#   out_var - Boolean result variable
# ~~~
function(_validate_test_category_inputs target_name yaml_file working_dir parse_script out_var)
    set(valid TRUE)
    if("${target_name}" STREQUAL "")
        message(WARNING "target_name is empty, cannot generate test categories")
        set(valid FALSE)
    endif()
    if(NOT EXISTS "${yaml_file}")
        message(WARNING "Test categories YAML file not found: ${yaml_file}")
        set(valid FALSE)
    endif()
    if(NOT IS_DIRECTORY "${working_dir}")
        message(WARNING "Working directory does not exist: ${working_dir}")
        set(valid FALSE)
    endif()
    if(NOT EXISTS "${parse_script}")
        message(WARNING "Test category parser script not found: ${parse_script}")
        set(valid FALSE)
    endif()
    set(${out_var} "${valid}" PARENT_SCOPE)
endfunction()


# Function to apply category labels to discovered GTest tests
#
# Arguments:
#   target_name - GTest executable target name
#   yaml_file - Path to test_categories.yaml
#   working_dir - Working directory for test execution
#
#   install_test_file - Path to write install-time test definitions
#   resource_group - CTest RESOURCE_GROUPS token to apply to generated suites
#
# Optional named arguments:
#   INSTALL_TEST_FILE - Path to write install-time test definitions
#   RESOURCE_GROUP - CTest RESOURCE_GROUPS token to apply to generated suites
#   TEST_NAME_PREFIX - Prefix for generated CTest names
#   COMMAND_ARGS - Extra build-tree command args before --gtest_filter
#   INSTALL_COMMAND_ARGS - Extra install-tree command args before --gtest_filter
#   INSTALL_EXECUTABLE - Install-tree executable path; defaults to ../target_name
#   ADDITIONAL_LABELS - Labels appended to every generated suite
#   ENVIRONMENT - Extra ENVIRONMENT entries (KEY=VALUE) applied to every
#       generated suite, merged with execution_settings.environment from the
#       YAML (this list wins on key conflicts). Use to forward CMake-side
#       TEST_ENVIRONMENT (ASAN symbolizer path, coverage LLVM_PROFILE_FILE).
#   USE_RTEST_DRIVER - Run the project's <stem>_rtest.py driver with
#       -t ctest_<category> instead of invoking the gtest binary directly.
# ~~~
function(apply_test_category_labels target_name yaml_file working_dir)
    _parse_test_category_optional_args(${ARGN})

    if(NOT Python3_FOUND)
        message(WARNING "Python3 not found, cannot parse test categories YAML")
        return()
    endif()

    set(PARSE_SCRIPT "${ROCM_LIBRARIES_ROOT}/shared/ctest/parse_test_categories.py")
    _validate_test_category_inputs(
        "${target_name}"
        "${yaml_file}"
        "${working_dir}"
        "${PARSE_SCRIPT}"
        inputs_valid
    )
    if(NOT inputs_valid)
        return()
    endif()

    _build_test_category_parser_args(extra_args)
    if(_TEST_CATEGORY_INSTALL_FILE)
        set(
            python_args
            ${extra_args}
            ${yaml_file}
            ${target_name}
            ${working_dir}
            ${_TEST_CATEGORY_INSTALL_FILE}
        )
    else()
        set(python_args ${extra_args} ${yaml_file} ${target_name} ${working_dir})
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${PARSE_SCRIPT} ${python_args}
        OUTPUT_VARIABLE CMAKE_CATEGORY_CODE
        ERROR_VARIABLE PARSE_ERROR
        RESULT_VARIABLE PARSE_RESULT
    )

    if(NOT PARSE_RESULT EQUAL 0)
        message(WARNING "Failed to parse test categories YAML: ${PARSE_ERROR}")
        return()
    endif()

    # Write the generated CMake code to a file and include it
    set(CATEGORY_CMAKE "${CMAKE_CURRENT_BINARY_DIR}/test_categories.cmake")
    file(WRITE "${CATEGORY_CMAKE}" "${CMAKE_CATEGORY_CODE}")

    message(STATUS "Generated test category configuration: ${CATEGORY_CMAKE}")

    # Verify the generated CMake file exists before including it
    if(NOT EXISTS "${CATEGORY_CMAKE}")
        message(WARNING "Generated test categories file not found: ${CATEGORY_CMAKE}")
        return()
    endif()

    # Include and execute the generated CMake code
    include("${CATEGORY_CMAKE}")
endfunction()

# Function to apply category labels to discovered Catch2 tests using tag-based filtering
# Optional 4th parameter: install_test_file - path to write install-time test definitions
function(apply_catch2_test_category_labels target_name yaml_file working_dir)
    if(NOT Python3_FOUND)
        message(WARNING "Python3 not found, cannot parse Catch2 test categories YAML")
        return()
    endif()

    # Validate inputs
    set(_validation_failed FALSE)
    if("${target_name}" STREQUAL "")
        message(WARNING "target_name is empty, cannot generate Catch2 test categories")
        set(_validation_failed TRUE)
    endif()
    if(NOT EXISTS "${yaml_file}")
        message(WARNING "Catch2 test categories YAML file not found: ${yaml_file}")
        set(_validation_failed TRUE)
    endif()
    if(NOT IS_DIRECTORY "${working_dir}")
        message(WARNING "Working directory does not exist: ${working_dir}")
        set(_validation_failed TRUE)
    endif()
    if(_validation_failed)
        return()
    endif()

    set(PARSE_SCRIPT "${ROCM_LIBRARIES_ROOT}/shared/ctest/parse_catch2_categories.py")
    if(NOT EXISTS "${PARSE_SCRIPT}")
        message(WARNING "Catch2 test category parser script not found: ${PARSE_SCRIPT}")
        return()
    endif()

    set(install_test_file "${ARGV3}")
    if(install_test_file)
        set(python_args ${yaml_file} ${target_name} ${working_dir} ${install_test_file})
    else()
        set(python_args ${yaml_file} ${target_name} ${working_dir})
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${PARSE_SCRIPT} ${python_args}
        OUTPUT_VARIABLE CMAKE_CATEGORY_CODE
        ERROR_VARIABLE PARSE_ERROR
        RESULT_VARIABLE PARSE_RESULT
    )

    if(NOT PARSE_RESULT EQUAL 0)
        message(WARNING "Failed to parse Catch2 test categories YAML: ${PARSE_ERROR}")
        return()
    endif()

    set(CATEGORY_CMAKE "${CMAKE_CURRENT_BINARY_DIR}/catch2_test_categories.cmake")
    file(WRITE "${CATEGORY_CMAKE}" "${CMAKE_CATEGORY_CODE}")

    message(STATUS "Generated Catch2 test category configuration: ${CATEGORY_CMAKE}")

    if(NOT EXISTS "${CATEGORY_CMAKE}")
        message(WARNING "Generated Catch2 test categories file not found: ${CATEGORY_CMAKE}")
        return()
    endif()

    include("${CATEGORY_CMAKE}")
endfunction()

# Function to apply category labels to CTest tests already registered in the
# current directory (no GTest discovery required).
#
# Optional 2nd parameter: install_test_file - path to a hand-rolled
#   CTestTestfile.cmake (or fragment) that the caller has written
#   add_test() lines into. When provided, the parser additionally appends
#   the generated label code to that file AND scans it for the test names
#   to emit explicit per-test set_tests_properties() calls -- ctest's own
#   script interpreter does not implement get_property(DIRECTORY ...
#   PROPERTY TESTS), and set_property(TEST ...) labels are not visible to
#   ctest -L, so the default runtime enumeration loop is a silent no-op at
#   ctest time. When omitted (autogenerated CTestTestfile case), the parser
#   uses the directory-property loop, which CMake materialises into
#   explicit set_tests_properties() lines at configure time.

# Returns the shared CTest category parser script path.
#
# Arguments:
#   out_var - Variable to receive the parser script path
# ~~~
function(_ctest_categories_parse_script out_var)
    set(PARSE_SCRIPT
        "${ROCM_LIBRARIES_ROOT}/shared/ctest/parse_ctest_categories.py"
    )
    if(NOT EXISTS "${PARSE_SCRIPT}")
        message(
            FATAL_ERROR
            "Test category parser script not found: ${PARSE_SCRIPT}"
        )
    endif()
    set(${out_var} "${PARSE_SCRIPT}" PARENT_SCOPE)
endfunction()


# Reads YAML category names for CMake target generation.
#
# Arguments:
#   yaml_file - Path to test_categories.yaml
#   out_var - Variable to receive the semicolon-separated category list
# ~~~
function(get_ctest_category_names yaml_file out_var)
    if(NOT Python3_FOUND)
        message(WARNING "Python3 not found, cannot parse test category names")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    if(NOT EXISTS "${yaml_file}")
        message(WARNING "Test categories YAML file not found: ${yaml_file}")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    _ctest_categories_parse_script(PARSE_SCRIPT)

    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${PARSE_SCRIPT} --print-categories ${yaml_file}
        OUTPUT_VARIABLE CATEGORY_NAMES
        ERROR_VARIABLE PARSE_ERROR
        RESULT_VARIABLE PARSE_RESULT
    )

    if(NOT PARSE_RESULT EQUAL 0)
        message(WARNING "Failed to parse test category names: ${PARSE_ERROR}")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    string(STRIP "${CATEGORY_NAMES}" CATEGORY_NAMES)
    set(${out_var} "${CATEGORY_NAMES}" PARENT_SCOPE)
endfunction()


# Applies category labels to already-registered CTest tests.
#
# Arguments:
#   yaml_file - Path to test_categories.yaml
#
# Optional named arguments:
#   INSTALL_TEST_FILE - Generated install-tree CTestTestfile.cmake to append labels to
#   EXPLICIT_TESTS - Known CTest test names to label without directory-scope enumeration
# ~~~
function(apply_ctest_category_labels yaml_file)
    cmake_parse_arguments(
        ARG
        ""
        "INSTALL_TEST_FILE"
        "EXPLICIT_TESTS"
        ${ARGN}
    )

    # Backward-compatible positional form:
    #   apply_ctest_category_labels(yaml_file install_test_file)
    set(install_test_file "${ARG_INSTALL_TEST_FILE}")
    if(NOT install_test_file AND ARG_UNPARSED_ARGUMENTS)
        list(GET ARG_UNPARSED_ARGUMENTS 0 install_test_file)
    endif()

    # Execute the Python script to generate CMake code
    if(NOT Python3_FOUND)
        message(WARNING "Python3 not found, cannot parse test categories YAML")
        return()
    endif()

    # Validate inputs
    set(_validation_failed FALSE)
    if(NOT EXISTS "${yaml_file}")
        message(WARNING "Test categories YAML file not found: ${yaml_file}")
        set(_validation_failed TRUE)
    endif()
    if(_validation_failed)
        return()
    endif()

    _ctest_categories_parse_script(PARSE_SCRIPT)

    set(python_args ${yaml_file})
    if(install_test_file)
        list(APPEND python_args ${install_test_file})
    endif()
    if(ARG_EXPLICIT_TESTS)
        list(JOIN ARG_EXPLICIT_TESTS ";" explicit_tests)
        list(APPEND python_args --explicit-tests "${explicit_tests}")
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${PARSE_SCRIPT} ${python_args}
        OUTPUT_VARIABLE CMAKE_CATEGORY_CODE
        ERROR_VARIABLE PARSE_ERROR
        RESULT_VARIABLE PARSE_RESULT
    )

    if(NOT PARSE_RESULT EQUAL 0)
        message(WARNING "Failed to parse test categories YAML: ${PARSE_ERROR}")
        return()
    endif()

    # Write the generated CMake code to a file and include it
    set(CATEGORY_CMAKE "${CMAKE_CURRENT_BINARY_DIR}/test_categories.cmake")
    file(WRITE "${CATEGORY_CMAKE}" "${CMAKE_CATEGORY_CODE}")

    message(STATUS "Generated test category configuration: ${CATEGORY_CMAKE}")

    # Verify the generated CMake file exists before including it
    if(NOT EXISTS "${CATEGORY_CMAKE}")
        message(
            WARNING
            "Generated test categories file not found: ${CATEGORY_CMAKE}"
        )
        return()
    endif()

    # Include and execute the generated CMake code. When an
    # install_test_file was provided, the generated code is shaped for
    # ctest's interpreter (explicit test names, no get_property loop)
    # and is meant to live inside the install file; including it here
    # would just emit `if(TEST x)` guards for tests that may not exist
    # in this directory scope -- harmless, but noisy. Build-tree callers
    # (no install file) are the consumers of the include() path.
    if(NOT install_test_file)
        include("${CATEGORY_CMAKE}")
    endif()
endfunction()
