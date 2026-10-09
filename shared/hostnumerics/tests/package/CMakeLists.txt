# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(HostNumericsPackageTest LANGUAGES CXX)

enable_testing()
find_package(
    HostNumerics
    CONFIG
    REQUIRED
    COMPONENTS Core Operations AMDGPULayout
    OPTIONAL_COMPONENTS BLAS
)

add_executable(hostnumerics-package-test main.cpp)
target_link_libraries(hostnumerics-package-test PRIVATE roc::hostnumerics-core)
add_test(
    NAME hostnumerics-core-package-consumer
    COMMAND hostnumerics-package-test
)

add_executable(hostnumerics-operations-package-test validation.cpp)
target_link_libraries(hostnumerics-operations-package-test PRIVATE roc::hostnumerics)
add_test(
    NAME hostnumerics-operations-package-consumer
    COMMAND hostnumerics-operations-package-test
)

add_executable(hostnumerics-comparison-package-test comparison.cpp)
target_link_libraries(hostnumerics-comparison-package-test PRIVATE roc::hostnumerics)
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
)
add_test(
    NAME hostnumerics-amd-gpu-layout-package-consumer
    COMMAND hostnumerics-amd-gpu-layout-package-test
)

add_executable(hostnumerics-version-package-test ../version_test.cpp)
target_link_libraries(hostnumerics-version-package-test PRIVATE roc::hostnumerics)
add_test(NAME hostnumerics-version-package-consumer COMMAND hostnumerics-version-package-test)
