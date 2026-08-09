# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(host-validation-tests host_validation_test.cpp)
target_link_libraries(host-validation-tests PRIVATE roc::host-validation)
add_test(NAME host-validation-tests COMMAND host-validation-tests)

add_executable(host-validation-comparison-policy-tests comparison_policy_test.cpp)
target_link_libraries(host-validation-comparison-policy-tests PRIVATE roc::host-validation)
add_test(NAME host-validation-comparison-policy-tests COMMAND host-validation-comparison-policy-tests)

add_executable(host-validation-tiled-tests tiled_backend_test.cpp)
target_link_libraries(host-validation-tiled-tests PRIVATE roc::host-validation-tiled)
add_test(NAME host-validation-tiled-tests COMMAND host-validation-tiled-tests)

add_executable(host-validation-core-tests tensor_core_test.cpp)
target_link_libraries(host-validation-core-tests PRIVATE roc::host-validation-core)
add_test(NAME host-validation-core-tests COMMAND host-validation-core-tests)

add_executable(host-validation-scalar-codec-tests scalar_codec_test.cpp)
target_link_libraries(host-validation-scalar-codec-tests PRIVATE roc::host-validation-core)
add_test(NAME host-validation-scalar-codec-tests COMMAND host-validation-scalar-codec-tests)

if(TARGET host-validation-blas)
    add_executable(host-validation-blas-tests blas_backend_test.cpp)
    target_link_libraries(
        host-validation-blas-tests
        PRIVATE
            roc::host-validation-blas
    )
    add_test(NAME host-validation-blas-tests COMMAND host-validation-blas-tests)
endif()

if(TARGET host-validation-mx)
    add_executable(host-validation-mx-tests mx_test.cpp)
    target_link_libraries(
        host-validation-mx-tests
        PRIVATE
            roc::host-validation-mx
    )
    add_test(NAME host-validation-mx-tests COMMAND host-validation-mx-tests)
endif()

add_test(
    NAME host-validation-core-boundary
    COMMAND
        "${CMAKE_COMMAND}"
        "-DCORE_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../include/roc/host_validation/tensor.hpp"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_core_boundary.cmake"
)

add_test(
    NAME host-validation-component-boundary
    COMMAND
        "${CMAKE_COMMAND}"
        -DHOST_VALIDATION_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_component_boundary.cmake"
)

if(HOST_VALIDATION_BUILD_BENCHMARKS)
    find_package(OpenMP QUIET COMPONENTS CXX)
    add_executable(
        host-validation-structured-sparsity-benchmark
        structured_sparsity_benchmark.cpp
    )
    target_link_libraries(
        host-validation-structured-sparsity-benchmark
        PRIVATE
            roc::host-validation
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(
            host-validation-structured-sparsity-benchmark
            PRIVATE
                OpenMP::OpenMP_CXX
        )
        target_compile_definitions(
            host-validation-structured-sparsity-benchmark
            PRIVATE
                HOST_VALIDATION_BENCHMARK_OPENMP=1
        )
    endif()

    add_executable(
        host-validation-comparison-benchmark
        comparison_benchmark.cpp
    )
    target_link_libraries(
        host-validation-comparison-benchmark
        PRIVATE
            roc::host-validation
    )
endif()
