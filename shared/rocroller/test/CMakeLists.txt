# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
        )
        FetchContent_MakeAvailable(googletest)
    else()
        message(FATAL_ERROR "Failed to find googletest")
    endif()
endif()

if(NOT ROCM_LIBS_SUPERBUILD)
    if(ROCROLLER_ENABLE_THEROCK)
        find_package(mxDataGenerator REQUIRED)
    else()
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../shared/mxdatagenerator" "${CMAKE_CURRENT_BINARY_DIR}/mxdatagenerator")
    endif()
endif()

add_executable(rocroller-tests)

if(ROCROLLER_ENABLE_CATCH AND (ROCROLLER_BUILD_TESTING OR BUILD_TESTING))
    add_executable(rocroller-tests-catch)
    if(ROCROLLER_ENABLE_ROCPROF_ANALYZER)
        add_executable(rocroller-tests-rocprofiler)
    endif()
    # When detecting tests gtest and catch test discovery modules run
    # something like:
    # execute_process(COMMAND ${_TEST_EXECUTOR} "${_TEST_EXECUTABLE}" --gtest_list_tests ${filter})
    # where TEST_EXECUTABLE is rocroller-tests-catch and TEST_EXECUTOR
    # is a command to prepend primarily for the purposes of running in
    # a cross compiled environment. As such, we can set the CROSSCOMPILING_EMULATOR
    # property on the test targets to use the cmake -E -env command providing
    # the LD_PRELOAD and ASAN_OPTIONS variables necessary to run sanitizer builds.
    # Note that the entries in the string are semicolon separated which is necessary
    # to ensure that the command supplied to execute_process is interpreted as a list
    # of strings rather than one white space delimited string.
    if(ROCROLLER_ENABLE_ASAN)
        rocroller_target_configure_sanitizers(rocroller-tests-catch PRIVATE)
        set_target_properties(rocroller-tests-catch
            PROPERTIES
                CROSSCOMPILING_EMULATOR "${CMAKE_COMMAND};-E;env;${asan_opts}"
        )
        if(ROCROLLER_ENABLE_ROCPROF_ANALYZER)
            rocroller_target_configure_sanitizers(rocroller-tests-rocprofiler PRIVATE)
            set_target_properties(rocroller-tests-rocprofiler
                PROPERTIES
                    CROSSCOMPILING_EMULATOR "${CMAKE_COMMAND};-E;env;${asan_opts}"
            )
        endif()
    endif()
endif()

# Component stand-alone test
add_executable(component-test)

if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(component-test PRIVATE)
    set_target_properties(component-test
        PROPERTIES
            CROSSCOMPILING_EMULATOR "${CMAKE_COMMAND};-E;env;${asan_opts}"
    )
    rocroller_target_configure_sanitizers(rocroller-tests PRIVATE)
    set_target_properties(rocroller-tests
        PROPERTIES
            CROSSCOMPILING_EMULATOR "${CMAKE_COMMAND};-E;env;${asan_opts}"
    )
endif()

if(ROCROLLER_ENABLE_GEMM_CLIENT_TESTS)
    find_package(Pytest REQUIRED) # requires pip install -r requirements.txt
    pytest_discover_tests(
        GEMMClientTests
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/client"
        ENVIRONMENT "ROCROLLER_BUILD_DIR=${PROJECT_BINARY_DIR}" "LD_PRELOAD=${ASAN_LIB_PATH}" "ASAN_OPTIONS=detect_leaks=0"
    )
endif()

add_executable(arch-gen-tests)
if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(arch-gen-tests PRIVATE)
    set_target_properties(arch-gen-tests
        PROPERTIES
            CROSSCOMPILING_EMULATOR "${CMAKE_COMMAND};-E;env;${asan_opts}"
    )
endif()

find_package(BLAS REQUIRED)
find_package(cblas)
find_package(OpenMP REQUIRED)

if(ROCROLLER_ENABLE_YAML_CPP)
    target_compile_definitions(rocroller-tests PRIVATE ROCROLLER_TESTS_USE_YAML_CPP)
else()
    target_compile_definitions(rocroller-tests PRIVATE ROCROLLER_USE_LLVM)
endif()

if(ROCROLLER_ENABLE_CATCH)
    add_subdirectory(catch)
    if(ROCROLLER_ENABLE_ROCPROF_ANALYZER)
        add_subdirectory(rocprofiler)
    endif()
endif()
add_subdirectory(common)
add_subdirectory(standalone)
add_subdirectory(unit)

if(ROCROLLER_ENABLE_TEST_DISCOVERY)
    set(TEST_REPORT_DIR "${PROJECT_BINARY_DIR}/test_report")

    include(GoogleTest)
    gtest_discover_tests(
        rocroller-tests
        XML_OUTPUT_DIR ${TEST_REPORT_DIR}
        TEST_FILTER "-*GPU_*"
        DISCOVERY_MODE PRE_TEST
        PROPERTIES
            "LABELS" "GTEST"
            "ENVIRONMENT" "${asan_opts}"
    )
    gtest_discover_tests(
        rocroller-tests
        XML_OUTPUT_DIR ${TEST_REPORT_DIR}
        TEST_FILTER "*GPU_*"
        PROPERTIES
            "LABELS" "GPU"
            "LABELS" "GTEST"
            "ENVIRONMENT" "${asan_opts}"
        DISCOVERY_MODE PRE_TEST
    )
    gtest_discover_tests(
        arch-gen-tests
        XML_OUTPUT_DIR ${TEST_REPORT_DIR}
        DISCOVERY_MODE PRE_TEST
        PROPERTIES
            "LABELS" "GTEST"
            "ENVIRONMENT" "${asan_opts}"
    )

    if(ROCROLLER_ENABLE_CATCH)
        if(ROCROLLER_ENABLE_FETCH)
            list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
        else()
            list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
        endif()

        include(Catch)
        catch_discover_tests(
            rocroller-tests-catch
            REPORTER JUnit
            OUTPUT_DIR ${TEST_REPORT_DIR}
            OUTPUT_SUFFIX ".xml"
            DISCOVERY_MODE PRE_TEST
            EXTRA_ARGS "--reporter console::out=-::colour-mode=ansi"
            TEST_SPEC "[gpu]"
            PROPERTIES "LABELS" "CATCH"
                       "LABELS" "GPU"
                       "ENVIRONMENT" "${asan_opts}"
            SKIP_REGULAR_EXPRESSION "[1-9][0-9]* skipped" "[1-9][0-9]* SKIPPED"
        )
        catch_discover_tests(
            rocroller-tests-catch
            REPORTER JUnit
            OUTPUT_DIR ${TEST_REPORT_DIR}
            OUTPUT_SUFFIX ".xml"
            DISCOVERY_MODE PRE_TEST
            EXTRA_ARGS "--reporter console::out=-::colour-mode=ansi"
            TEST_SPEC "~[gpu]"
            PROPERTIES "LABELS" "CATCH"
                       "ENVIRONMENT" "${asan_opts}"
            SKIP_REGULAR_EXPRESSION "[1-9][0-9]* skipped" "[1-9][0-9]* SKIPPED"
        )
        if(ROCROLLER_ENABLE_ROCPROF_ANALYZER)
            catch_discover_tests(
                rocroller-tests-rocprofiler
                REPORTER JUnit
                OUTPUT_DIR ${TEST_REPORT_DIR}
                OUTPUT_SUFFIX ".xml"
                DISCOVERY_MODE PRE_TEST
                EXTRA_ARGS "--reporter console::out=-::colour-mode=ansi"
                PROPERTIES "LABELS" "ROCPROFILER"
                           "LABELS" "CATCH"
                           "LABELS" "GPU"
                           "ENVIRONMENT" "${asan_opts}"
                SKIP_REGULAR_EXPRESSION "[1-9][0-9]* skipped" "[1-9][0-9]* SKIPPED"
            )
        endif()
    endif()

    if(NOT ROCROLLER_ENABLE_SLOW_TESTS)
        set(PYTHON_TEST_SKIP "-m not slow")
    endif()

    add_test(
        NAME PythonLint
        COMMAND flake8
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/scripts"
    )
    set_tests_properties(
        PythonLint
        PROPERTIES ENVIRONMENT "ROCROLLER_BUILD_DIR=${PROJECT_BINARY_DIR}"
                   "LABELS" "PYTHON"
    )
    add_test(
        NAME PythonTest
        COMMAND
            pytest -s --cov-report html:python_cov_html --cov=rrperf scripts
            ${PYTHON_TEST_SKIP} --junit-xml=${TEST_REPORT_DIR}/python_tests.xml
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    set_tests_properties(
        PythonTest
        PROPERTIES
            ENVIRONMENT
            "ROCROLLER_BUILD_DIR=${PROJECT_BINARY_DIR};PYTHONPATH=${PROJECT_SOURCE_DIR}/scripts/lib"
            "LABELS" "PYTHON"
    )
endif()

if(ROCROLLER_ENABLE_COVERAGE OR BUILD_CODE_COVERAGE)
    target_compile_options(rocroller PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(rocroller PRIVATE -fprofile-instr-generate)

    target_compile_options(common-test-utilities PRIVATE -fprofile-instr-generate -fcoverage-mapping)

    if(ROCROLLER_ENABLE_ARCH_GEN_TEST)
        target_compile_options(arch-gen-tests PRIVATE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(arch-gen-tests PRIVATE -fprofile-instr-generate)
    endif()

    target_compile_options(rocroller-tests PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(rocroller-tests PRIVATE -fprofile-instr-generate)

    target_compile_options(rocroller-tests-catch PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(rocroller-tests-catch PRIVATE -fprofile-instr-generate)

    if(ROCROLLER_ENABLE_ROCPROF_ANALYZER)
        target_compile_options(rocroller-tests-rocprofiler PRIVATE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(rocroller-tests-rocprofiler PRIVATE -fprofile-instr-generate)
    endif()
endif()

# Create install-time test file for distribution
set(INSTALL_TEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
file(WRITE "${INSTALL_TEST_FILE}"
[=[
# This is a test file generated by rocRoller for install time.
# Tests are defined with relative paths to work in the installed location.
]=]
)

set(_test_categories_yaml "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml")
include(${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake OPTIONAL)

set(_test_categories_catch2_yaml "${CMAKE_CURRENT_SOURCE_DIR}/test_categories_catch2.yaml")

if(COMMAND apply_test_category_labels)
    if(EXISTS "${_test_categories_yaml}")
        message(STATUS "YAML-based test categorization")
        apply_test_category_labels("rocroller-tests" "${_test_categories_yaml}" "${PROJECT_BINARY_DIR}" "${INSTALL_TEST_FILE}")
    else()
        message(STATUS "No test categorization (YAML missing)")
    endif()
else()
    message(STATUS "Test categorization not available (TestCategories.cmake not found)")
endif()

if(COMMAND apply_catch2_test_category_labels)
    if(EXISTS "${_test_categories_catch2_yaml}")
        message(STATUS "YAML-based Catch2 test categorization")
        apply_catch2_test_category_labels("rocroller-tests-catch" "${_test_categories_catch2_yaml}" "${PROJECT_BINARY_DIR}" "${INSTALL_TEST_FILE}")
    else()
        message(STATUS "No Catch2 test categorization (YAML missing)")
    endif()
else()
    message(STATUS "Catch2 test categorization not available (TestCategories.cmake not found)")
endif()

install(
    FILES "${INSTALL_TEST_FILE}"
    DESTINATION "${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}"
    COMPONENT tests
    RENAME "CTestTestfile.cmake"
)
