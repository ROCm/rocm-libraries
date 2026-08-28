# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostNumericsPackageTest LANGUAGES CXX)

enable_testing()
find_package(
    ROCHostNumerics
    CONFIG
    REQUIRED
    COMPONENTS Core Operations AMDGPULayout
    OPTIONAL_COMPONENTS BLAS
)

add_executable(host-numerics-package-test main.cpp)
target_link_libraries(host-numerics-package-test PRIVATE roc::host-numerics-core)
add_test(
    NAME host-numerics-core-package-consumer
    COMMAND host-numerics-package-test
)

add_executable(host-numerics-operations-package-test validation.cpp)
target_link_libraries(host-numerics-operations-package-test PRIVATE roc::host-numerics)
add_test(
    NAME host-numerics-operations-package-consumer
    COMMAND host-numerics-operations-package-test
)

add_executable(host-numerics-comparison-package-test comparison.cpp)
target_link_libraries(host-numerics-comparison-package-test PRIVATE roc::host-numerics)
add_test(
    NAME host-numerics-comparison-package-consumer
    COMMAND host-numerics-comparison-package-test
)

if(TARGET roc::host-numerics-blas)
    add_executable(host-numerics-blas-package-test blas.cpp)
    target_link_libraries(
        host-numerics-blas-package-test
        PRIVATE
            roc::host-numerics-blas
    )
    add_test(
        NAME host-numerics-blas-package-consumer
        COMMAND host-numerics-blas-package-test
    )
endif()

add_executable(host-numerics-mx-package-test mx.cpp)
target_link_libraries(
    host-numerics-mx-package-test
    PRIVATE
        roc::host-numerics
)
add_test(
    NAME host-numerics-mx-package-consumer
    COMMAND host-numerics-mx-package-test
)

add_executable(
    host-numerics-amd-gpu-layout-package-test
    amd_gpu_layout.cpp
)
target_link_libraries(
    host-numerics-amd-gpu-layout-package-test
    PRIVATE
        roc::host-numerics-amd-gpu-layout
)
add_test(
    NAME host-numerics-amd-gpu-layout-package-consumer
    COMMAND host-numerics-amd-gpu-layout-package-test
)
