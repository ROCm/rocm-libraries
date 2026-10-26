# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(HostNumericsTestDependencies)
include(GoogleTest)

add_executable(hostnumerics-layout-assignment-test layout_assignment_test.cpp)
target_link_libraries(hostnumerics-layout-assignment-test PRIVATE roc::hostnumerics-core)
add_test(NAME hostnumerics-layout-assignment-test COMMAND hostnumerics-layout-assignment-test)

include(HostNumericsExpectedParity)

set(HOSTNUMERICS_TEST_GTEST_SOURCE_DIR "${googletest_SOURCE_DIR}" PARENT_SCOPE)

add_executable(
    hostnumerics-periodic-tests
    periodic_initialization_test.cpp
    periodic_matmul_test.cpp
)
target_link_libraries(hostnumerics-periodic-tests PRIVATE roc::hostnumerics GTest::gtest_main)
add_test(NAME hostnumerics-periodic-tests COMMAND hostnumerics-periodic-tests)

add_executable(hostnumerics-prepared-iteration-tests prepared_iteration_test.cpp)
target_include_directories(hostnumerics-prepared-iteration-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(hostnumerics-prepared-iteration-tests PRIVATE roc::hostnumerics GTest::gtest_main)
add_test(NAME hostnumerics-prepared-iteration-tests COMMAND hostnumerics-prepared-iteration-tests)

add_executable(hostnumerics-freivalds-tests freivalds_test.cpp)
target_link_libraries(hostnumerics-freivalds-tests PRIVATE roc::hostnumerics GTest::gtest_main)
add_test(NAME hostnumerics-freivalds-tests COMMAND hostnumerics-freivalds-tests)

add_executable(hostnumerics-exact-verification-tests exact_verification_test.cpp)
target_link_libraries(hostnumerics-exact-verification-tests PRIVATE roc::hostnumerics GTest::gtest_main)
add_test(NAME hostnumerics-exact-verification-tests COMMAND hostnumerics-exact-verification-tests)

add_executable(
    hostnumerics-tests
    comparison_test.cpp
    comparison_metrics_test.cpp
    epilogue_test.cpp
    generation_comparison_test.cpp
    generation_test.cpp
    hostnumerics_test.cpp
    operation_aliasing_test.cpp
    index_selection_test.cpp
    reduction_test.cpp
    reference_gemm_test.cpp
    softmax_layer_norm_test.cpp
    strided_views_test.cpp
    structured_sparsity_test.cpp
    tensor_operations_test.cpp
)
target_link_libraries(hostnumerics-tests PRIVATE roc::hostnumerics GTest::gtest_main)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(hostnumerics-tests PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(
        hostnumerics-tests
        PRIVATE
            HOSTNUMERICS_TEST_OPENMP=1
    )
endif()
add_test(NAME hostnumerics-tests COMMAND hostnumerics-tests)

add_executable(hostnumerics-tutorial ../examples/tutorial.cpp)
target_link_libraries(hostnumerics-tutorial PRIVATE roc::hostnumerics)
add_test(NAME hostnumerics-tutorial COMMAND hostnumerics-tutorial)

add_executable(hostnumerics-generation-recipe-tests generation_recipe_test.cpp)
target_link_libraries(
    hostnumerics-generation-recipe-tests
    PRIVATE
        roc::hostnumerics
        GTest::gtest_main
)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(
        hostnumerics-generation-recipe-tests
        PRIVATE
            OpenMP::OpenMP_CXX
    )
endif()
add_test(
    NAME hostnumerics-generation-recipe-tests
    COMMAND hostnumerics-generation-recipe-tests
)

add_executable(
    hostnumerics-blocked-tests
    blocked_accumulation_test.cpp
    blocked_aliasing_test.cpp
    blocked_backend_test.cpp
    blocked_execution_test.cpp
    blocked_parallel_test.cpp
    blocked_scaling_test.cpp
    blocked_selection_plan_test.cpp
)
target_link_libraries(hostnumerics-blocked-tests PRIVATE roc::hostnumerics GTest::gtest_main)
add_test(NAME hostnumerics-blocked-tests COMMAND hostnumerics-blocked-tests)

# Allocation fault injection is isolated from the other test executables.
add_executable(hostnumerics-gemm-resource-failure-tests gemm_resource_failure_test.cpp)
target_include_directories(
    hostnumerics-gemm-resource-failure-tests
    PRIVATE "${PROJECT_SOURCE_DIR}/src"
)
target_link_libraries(
    hostnumerics-gemm-resource-failure-tests
    PRIVATE roc::hostnumerics GTest::gtest_main
)
if(TARGET hostnumerics-blas)
    target_link_libraries(
        hostnumerics-gemm-resource-failure-tests
        PRIVATE roc::hostnumerics-blas CBLAS::CBLAS
    )
    target_compile_definitions(
        hostnumerics-gemm-resource-failure-tests
        PRIVATE HOSTNUMERICS_TEST_BLAS=1
    )
endif()
add_test(
    NAME hostnumerics-gemm-resource-failure-tests
    COMMAND hostnumerics-gemm-resource-failure-tests
)

add_test(
    NAME hostnumerics-blocked-tests-four-threads
    COMMAND hostnumerics-blocked-tests
)
set_tests_properties(
    hostnumerics-blocked-tests-four-threads
    PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=4;OMP_DYNAMIC=FALSE"
        PROCESSORS 4
)

add_executable(
    hostnumerics-core-tests
    expected_test.cpp
    result_integration_test.cpp
    layout_test.cpp
    core_test.cpp
    tensor_basics_test.cpp
    tensor_conversion_test.cpp
    tensor_core_test.cpp
    tensor_encoded_copy_test.cpp
    tensor_layout_storage_test.cpp
    tensor_storage_test.cpp
    tensor_transformations_test.cpp
)
target_link_libraries(hostnumerics-core-tests PRIVATE roc::hostnumerics-core GTest::gtest_main)
gtest_discover_tests(
    hostnumerics-core-tests
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "hostnumerics-core-tests."
)

add_executable(
    hostnumerics-scalar-codec-tests
    binary_float_test.cpp
    scalar_codec_test.cpp
    scalar_conversion_test.cpp
    scalar_integer_codec_test.cpp
    scalar_rounding_test.cpp
    scalar_type_test.cpp
)
target_link_libraries(hostnumerics-scalar-codec-tests PRIVATE roc::hostnumerics-core GTest::gtest_main)
add_test(NAME hostnumerics-scalar-codec-tests COMMAND hostnumerics-scalar-codec-tests)

if(TARGET hostnumerics-blas)
    add_executable(hostnumerics-blas-tests blas_backend_test.cpp)
    target_link_libraries(
        hostnumerics-blas-tests
        PRIVATE
            roc::hostnumerics-blas
            GTest::gtest_main
    )
    add_test(
        NAME hostnumerics-blas-tests-one-thread
        COMMAND hostnumerics-blas-tests
    )
    add_test(
        NAME hostnumerics-blas-tests-multi-thread
        COMMAND hostnumerics-blas-tests
    )
    set(_hostnumerics_blas_one_thread_environment)
    set(_hostnumerics_blas_multi_thread_environment)
    foreach(thread_variable
            OMP_NUM_THREADS
            OPENBLAS_NUM_THREADS
            GOTO_NUM_THREADS
            BLIS_NUM_THREADS
            MKL_NUM_THREADS
            VECLIB_MAXIMUM_THREADS)
        list(
            APPEND
            _hostnumerics_blas_one_thread_environment
            "${thread_variable}=1"
        )
        list(
            APPEND
            _hostnumerics_blas_multi_thread_environment
            "${thread_variable}=2"
        )
    endforeach()
    set_tests_properties(
        hostnumerics-blas-tests-one-thread
        PROPERTIES
            ENVIRONMENT "${_hostnumerics_blas_one_thread_environment}"
    )
    set_tests_properties(
        hostnumerics-blas-tests-multi-thread
        PROPERTIES
            ENVIRONMENT "${_hostnumerics_blas_multi_thread_environment}"
            PROCESSORS 2
    )
endif()

add_executable(hostnumerics-mx-tests mx_test.cpp)
target_link_libraries(
    hostnumerics-mx-tests
    PRIVATE
        roc::hostnumerics
        GTest::gtest_main
)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(
        hostnumerics-mx-tests
        PRIVATE
            OpenMP::OpenMP_CXX
    )
endif()
add_test(NAME hostnumerics-mx-tests COMMAND hostnumerics-mx-tests)

add_executable(
    hostnumerics-amd-gpu-layout-tests
    amd_gpu_gfx1250_layout_test.cpp
    amd_gpu_gfx950_layout_test.cpp
    amd_gpu_layout_dispatch_test.cpp
    amd_gpu_pre_swizzle_test.cpp
)
target_link_libraries(
    hostnumerics-amd-gpu-layout-tests
    PRIVATE
        GTest::gtest_main
        roc::hostnumerics-amd-gpu-layout
)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(
        hostnumerics-amd-gpu-layout-tests
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
    )
endif()
gtest_discover_tests(
    hostnumerics-amd-gpu-layout-tests
    DISCOVERY_MODE PRE_TEST
)

add_test(
    NAME hostnumerics-core-boundary
    COMMAND
        "${CMAKE_COMMAND}"
        "-DCORE_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../include/roc/hostnumerics"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_core_boundary.cmake"
)

add_test(
    NAME hostnumerics-component-boundary
    COMMAND
        "${CMAKE_COMMAND}"
        -DHOSTNUMERICS_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_component_boundary.cmake"
)

add_test(
    NAME hostnumerics-amd-gpu-layout-boundary
    COMMAND
        "${CMAKE_COMMAND}"
        -DHOSTNUMERICS_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_amd_gpu_layout_boundary.cmake"
)

add_test(
    NAME hostnumerics-amd-gpu-layout-package
    COMMAND
        "${CMAKE_COMMAND}"
        "-DCOMPONENT_BINARY_DIR=${PROJECT_BINARY_DIR}"
        "-DHOSTNUMERICS_SOURCE_DIR=${HostNumerics_SOURCE_DIR}"
        "-DPACKAGE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/package"
        "-DPACKAGE_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/amd-gpu-layout-package-test"
        "-DTEST_GENERATOR=${CMAKE_GENERATOR}"
        "-DTEST_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DTEST_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DTEST_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
        "-DTEST_DISABLE_FIND_PACKAGE_OpenMP=${CMAKE_DISABLE_FIND_PACKAGE_OpenMP}"
        "-DTEST_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
        "-DTEST_GTEST_SOURCE_DIR=${googletest_SOURCE_DIR}"
        "-DTEST_GTEST_DIR=${GTest_DIR}"
        "-DTEST_CONFIG=$<CONFIG>"
        "-DCTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/run_amd_gpu_layout_package_test.cmake"
)

if(HOSTNUMERICS_BUILD_BENCHMARKS)
    find_package(OpenMP QUIET COMPONENTS CXX)
    add_executable(
        hostnumerics-structured-sparsity-benchmark
        structured_sparsity_benchmark.cpp
    )
    target_link_libraries(
        hostnumerics-structured-sparsity-benchmark
        PRIVATE
            roc::hostnumerics
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(
            hostnumerics-structured-sparsity-benchmark
            PRIVATE
                OpenMP::OpenMP_CXX
        )
        target_compile_definitions(
            hostnumerics-structured-sparsity-benchmark
            PRIVATE
                HOSTNUMERICS_BENCHMARK_OPENMP=1
        )
    endif()

    add_executable(
        hostnumerics-comparison-benchmark
        comparison_benchmark.cpp
    )
    target_link_libraries(
        hostnumerics-comparison-benchmark
        PRIVATE
            roc::hostnumerics
    )

    add_executable(
        hostnumerics-generation-benchmark
        generation_benchmark.cpp
    )
    target_link_libraries(
        hostnumerics-generation-benchmark
        PRIVATE
            roc::hostnumerics
    )

    add_executable(
        hostnumerics-tensor-conversion-benchmark
        tensor_conversion_benchmark.cpp
    )
    target_link_libraries(
        hostnumerics-tensor-conversion-benchmark
        PRIVATE
            roc::hostnumerics
    )

    add_executable(
        hostnumerics-gemm-benchmark
        gemm_benchmark.cpp
    )
    target_link_libraries(
        hostnumerics-gemm-benchmark
        PRIVATE
            roc::hostnumerics
    )
endif()
