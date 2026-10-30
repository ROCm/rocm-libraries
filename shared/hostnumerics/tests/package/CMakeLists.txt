# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(HostNumericsPackageTest LANGUAGES CXX)

enable_testing()
if(NOT HOSTNUMERICS_SOURCE_DIR)
    message(FATAL_ERROR "Set HOSTNUMERICS_SOURCE_DIR to the HostNumerics source tree.")
endif()
list(APPEND CMAKE_MODULE_PATH "${HOSTNUMERICS_SOURCE_DIR}/cmake")
include(HostNumericsTestDependencies)
find_package(
    HostNumerics
    CONFIG
    REQUIRED
    COMPONENTS Core Operations AMDGPULayout
    OPTIONAL_COMPONENTS BLAS
)

add_executable(hostnumerics-package-test main.cpp)
target_link_libraries(hostnumerics-package-test PRIVATE roc::hostnumerics-core GTest::gtest_main)
add_test(
    NAME hostnumerics-core-package-consumer
    COMMAND hostnumerics-package-test
)

add_executable(hostnumerics-operations-package-test validation.cpp)
target_link_libraries(hostnumerics-operations-package-test PRIVATE roc::hostnumerics GTest::gtest_main)
add_test(
    NAME hostnumerics-operations-package-consumer
    COMMAND hostnumerics-operations-package-test
)

add_executable(hostnumerics-comparison-package-test comparison.cpp)
target_link_libraries(hostnumerics-comparison-package-test PRIVATE roc::hostnumerics GTest::gtest_main)
add_test(
    NAME hostnumerics-comparison-package-consumer
    COMMAND hostnumerics-comparison-package-test
)

if(TARGET roc::hostnumerics-blas)
    add_executable(hostnumerics-blas-package-test blas.cpp)
    target_link_libraries(
        hostnumerics-blas-package-test
        PRIVATE
            roc::hostnumerics-blas
            GTest::gtest_main
    )
    add_test(
        NAME hostnumerics-blas-package-consumer
        COMMAND hostnumerics-blas-package-test
    )
endif()

add_executable(hostnumerics-mx-package-test mx.cpp)
target_link_libraries(
    hostnumerics-mx-package-test
    PRIVATE
        roc::hostnumerics
        GTest::gtest_main
)
add_test(
    NAME hostnumerics-mx-package-consumer
    COMMAND hostnumerics-mx-package-test
)

add_executable(
    hostnumerics-amd-gpu-layout-package-test
    amd_gpu_layout.cpp
)
target_link_libraries(
    hostnumerics-amd-gpu-layout-package-test
    PRIVATE
        roc::hostnumerics-amd-gpu-layout
        GTest::gtest_main
)
add_test(
    NAME hostnumerics-amd-gpu-layout-package-consumer
    COMMAND hostnumerics-amd-gpu-layout-package-test
)

add_executable(hostnumerics-version-package-test "${HOSTNUMERICS_SOURCE_DIR}/tests/core_test.cpp"
    "${HOSTNUMERICS_SOURCE_DIR}/tests/expected_test.cpp"
    "${HOSTNUMERICS_SOURCE_DIR}/tests/integer_conversion_test.cpp")
target_link_libraries(hostnumerics-version-package-test PRIVATE roc::hostnumerics GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(
    hostnumerics-version-package-test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "hostnumerics-version-package-consumer."
)
