# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(GTest)
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

find_package(mxDataGenerator)
if(NOT mxDataGenerator_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        FetchContent_Declare(
            mxDataGenerator
            GIT_REPOSITORY ${MXDATAGENERATOR_GIT_URL}
            GIT_TAG ${MXDATAGENERATOR_GIT_TAG}
        )
        FetchContent_MakeAvailable(mxDataGenerator)
    else()
        message(FATAL_ERROR "Failed to find mxDataGenerator")
    endif()
endif()

add_executable(rocroller-tests)

if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(rocroller-tests PRIVATE)
endif()

if(ROCROLLER_ENABLE_CATCH AND (ROCROLLER_BUILD_TESTING OR BUILD_TESTING))
    add_executable(rocroller-tests-catch)
    if(ROCROLLER_ENABLE_ASAN)
        rocroller_target_configure_sanitizers(rocroller-tests-catch PRIVATE)
    endif()
endif()

# Component stand-alone test
add_executable(component-test)

if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(component-test PRIVATE)
endif()

if(ROCROLLER_ENABLE_ARCH_GEN_TEST)
    if(NOT ROCROLLER_ENABLE_LLD)
        # There are a few tests that require ReadELF functionality
        # which is disabled by default and on a deprecation path.
        # The tests that fail are missing the amdhsa.version key
        # and as such hit a fatal error that emits the following
        #     Key   key = amdhsa.version
        #      not found: {}
        # If observed, the test will be marked as skipped rather
        # than failing.
        if(ROCROLLER_ENABLE_LLVM)
            set(rocroller_skip_regex "FatalError\\(hsa_version.size\\(\\) == 2\\)")
        else()
            set(rocroller_skip_regex "key = amdhsa.version")
        endif()
    endif()
    find_package(Pytest REQUIRED) # requires pip install -r requirements.txt
    pytest_discover_tests(
        GEMMClientTests
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/client"
        ENVIRONMENT "ROCROLLER_BUILD_DIR=${PROJECT_BINARY_DIR}"
        PROPERTIES
            SKIP_REGULAR_EXPRESSION ${rocroller_skip_regex}
    )
    add_executable(arch-gen-tests)
    if(ROCROLLER_ENABLE_ASAN)
        rocroller_target_configure_sanitizers(arch-gen-tests PRIVATE)
    endif()
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
            SKIP_REGULAR_EXPRESSION "test/unit/ErrorTest.cpp:"
    )
    gtest_discover_tests(
        rocroller-tests
        XML_OUTPUT_DIR ${TEST_REPORT_DIR}
        TEST_FILTER "*GPU_*"
        PROPERTIES "LABELS" "GPU"
        DISCOVERY_MODE PRE_TEST
        PROPERTIES
            SKIP_REGULAR_EXPRESSION "test/unit/ErrorTest.cpp:"
    )
    if(ROCROLLER_ENABLE_ARCH_GEN_TEST)
        gtest_discover_tests(
            arch-gen-tests
            XML_OUTPUT_DIR ${TEST_REPORT_DIR}
            DISCOVERY_MODE PRE_TEST
        )
    endif()

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
            PROPERTIES "LABELS"
                       "CATCH"
                       "LABELS"
                       "GPU"
            SKIP_REGULAR_EXPRESSION "[1-9][0-9]* skipped"
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
            SKIP_REGULAR_EXPRESSION "[1-9][0-9]* skipped"
        )
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

    if(ROCROLLER_TESTS_SKIP_SLOW)
        set(PYTHON_TEST_SKIP "-m not slow")
    endif()

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

    target_compile_options(rocroller-no-rtti PRIVATE -fprofile-instr-generate -fcoverage-mapping)

    target_compile_options(common-test-utilities PRIVATE -fprofile-instr-generate -fcoverage-mapping)

    if(ROCROLLER_ENABLE_ARCH_GEN_TEST)
        target_compile_options(arch-gen-tests PRIVATE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(arch-gen-tests PRIVATE -fprofile-instr-generate)
    endif()

    target_compile_options(rocroller-tests PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(rocroller-tests PRIVATE -fprofile-instr-generate)

    target_compile_options(rocroller-tests-catch PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(rocroller-tests-catch PRIVATE -fprofile-instr-generate)
endif()
