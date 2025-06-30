################################################################################
#
# MIT License
#
# Copyright 2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

include(CTest)

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
            GIT_REPOSITORY "https://github.com/ROCm/mxDataGenerator.git"
            GIT_TAG b99e329c644d771be3ca026f4df5bb635714bd40
        )
        FetchContent_MakeAvailable(mxDataGenerator)
    else()
        message(FATAL_ERROR "Failed to find mxDataGenerator")
    endif()
endif()

add_executable(rocroller-tests)
if(ROCROLLER_ENABLE_CATCH AND (ROCROLLER_BUILD_TESTING OR BUILD_TESTING))
    add_executable(rocroller-tests-catch)
endif()

find_package(BLAS REQUIRED)
find_package(cblas REQUIRED)
find_package(OpenMP REQUIRED)

if(ROCROLLER_ENABLE_YAML_CPP)
    target_compile_definitions(rocroller-tests PUBLIC ROCROLLER_TESTS_USE_YAML_CPP)
endif()

if(ROCROLLER_ENABLE_CATCH)
    add_subdirectory(catch)
endif()
add_subdirectory(common)
add_subdirectory(unit)

# Component stand-alone test
add_executable(component-test "${ROCROLLER_TEST_DIR}/standalone/ComponentTest.cpp")
target_link_libraries(component-test PRIVATE roc::rocroller-interface ${msgpack_libs})

if(ROCROLLER_ENABLE_ARCH_GEN_TEST)
    find_package(Pytest REQUIRED) # requires pip install -r requirements.txt
    pytest_discover_tests(
        GEMMClientTests
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../client"
        ENVIRONMENT "ROCROLLER_BUILD_DIR=${PROJECT_BINARY_DIR}"
    )
    add_executable(arch-gen-tests "${ROCROLLER_TEST_DIR}/unit/GPUArchitectureGeneratorTest.cpp")
    target_include_directories(
        arch-gen-tests
            PRIVATE
                "${ROCROLLER_GPUARCH_DIR}/include"
    )
    target_link_libraries(
        arch-gen-tests
            PRIVATE
                roc::rocroller-interface
                rocroller::common-test-utilities
                rocroller::isa-spec-manager
                GTest::gtest_main
                GTest::gmock
                ${BLAS_LIBRARIES}
    )
endif()

if(ROCROLLER_ENABLE_TEST_DISCOVERY)
    enable_testing()
    set(TEST_REPORT_DIR "${PROJECT_BINARY_DIR}/test_report")

    include(GoogleTest)
    gtest_discover_tests(
        rocroller-tests
        XML_OUTPUT_DIR ${TEST_REPORT_DIR}
        TEST_FILTER "-*GPU_*"
        DISCOVERY_MODE PRE_TEST
    )
    gtest_discover_tests(
        rocroller-tests
        XML_OUTPUT_DIR ${TEST_REPORT_DIR}
        TEST_FILTER "*GPU_*"
        PROPERTIES "LABELS" "GPU"
        DISCOVERY_MODE PRE_TEST
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
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../scripts"
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
