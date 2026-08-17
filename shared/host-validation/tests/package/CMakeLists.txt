# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostValidationPackageTest LANGUAGES CXX)

enable_testing()
find_package(ROCHostValidation CONFIG REQUIRED)

add_executable(host-validation-package-test main.cpp)
target_link_libraries(host-validation-package-test PRIVATE roc::host-validation-core)
add_test(
    NAME host-validation-core-package-consumer
    COMMAND host-validation-package-test
)

add_executable(host-validation-operations-package-test validation.cpp)
target_link_libraries(host-validation-operations-package-test PRIVATE roc::host-validation)
add_test(
    NAME host-validation-operations-package-consumer
    COMMAND host-validation-operations-package-test
)

add_executable(host-validation-comparison-package-test comparison.cpp)
target_link_libraries(host-validation-comparison-package-test PRIVATE roc::host-validation)
add_test(
    NAME host-validation-comparison-package-consumer
    COMMAND host-validation-comparison-package-test
)

add_executable(host-validation-typed-comparison-package-test typed_comparison.cpp)
target_link_libraries(host-validation-typed-comparison-package-test PRIVATE roc::host-validation)
add_test(
    NAME host-validation-typed-comparison-package-consumer
    COMMAND host-validation-typed-comparison-package-test
)

add_executable(host-validation-tiled-package-test tiled.cpp)
target_link_libraries(host-validation-tiled-package-test PRIVATE roc::host-validation-tiled)
add_test(
    NAME host-validation-tiled-package-consumer
    COMMAND host-validation-tiled-package-test
)

if(TARGET roc::host-validation-blas)
    add_executable(host-validation-blas-package-test blas.cpp)
    target_link_libraries(
        host-validation-blas-package-test
        PRIVATE
            roc::host-validation-blas
    )
    add_test(
        NAME host-validation-blas-package-consumer
        COMMAND host-validation-blas-package-test
    )
endif()

if(TARGET roc::host-validation-mx)
    add_executable(host-validation-mx-package-test mx.cpp)
    target_link_libraries(
        host-validation-mx-package-test
        PRIVATE
            roc::host-validation-mx
    )
    add_test(
        NAME host-validation-mx-package-consumer
        COMMAND host-validation-mx-package-test
    )
endif()

add_executable(
    host-validation-amd-gpu-layout-package-test
    amd_gpu_layout.cpp
)
target_link_libraries(
    host-validation-amd-gpu-layout-package-test
    PRIVATE
        roc::host-validation-amd-gpu-layout
)
add_test(
    NAME host-validation-amd-gpu-layout-package-consumer
    COMMAND host-validation-amd-gpu-layout-package-test
)
