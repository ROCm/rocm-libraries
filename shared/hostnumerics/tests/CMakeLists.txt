# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(hostnumerics-version-test version_test.cpp)
target_link_libraries(hostnumerics-version-test PRIVATE roc::hostnumerics)
add_test(NAME hostnumerics-version COMMAND hostnumerics-version-test)

find_package(GTest 1.12.1 QUIET)
if(NOT GTest_FOUND)
    if(HOSTNUMERICS_ENABLE_FETCH)
        include(FetchContent)
        # GoogleTest is a test-only implementation detail. Its default install
        # rules would otherwise contaminate the HostNumerics package.
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
        )
        FetchContent_MakeAvailable(googletest)
    else()
        message(FATAL_ERROR "HostNumerics tests require GTest 1.12.1 or newer.")
    endif()
endif()

add_executable(
    hostnumerics-tests
    comparison_test.cpp
    comparison_metrics_test.cpp
    epilogue_test.cpp
    generation_comparison_test.cpp
    generation_test.cpp
    hostnumerics_test.cpp
    operation_aliasing_test.cpp
    output_selection_test.cpp
    reduction_test.cpp
    reference_gemm_test.cpp
    softmax_layer_norm_test.cpp
    strided_views_test.cpp
    structured_sparsity_test.cpp
    tensor_operations_test.cpp
)
target_link_libraries(hostnumerics-tests PRIVATE roc::hostnumerics)
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
target_link_libraries(hostnumerics-blocked-tests PRIVATE roc::hostnumerics)
add_test(NAME hostnumerics-blocked-tests COMMAND hostnumerics-blocked-tests)
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
    layout_test.cpp
    shape_test.cpp
    tensor_basics_test.cpp
    tensor_conversion_test.cpp
    tensor_core_test.cpp
    tensor_encoded_copy_test.cpp
    tensor_layout_storage_test.cpp
    tensor_storage_test.cpp
    tensor_transformations_test.cpp
)
target_link_libraries(hostnumerics-core-tests PRIVATE roc::hostnumerics-core)
add_test(NAME hostnumerics-core-tests COMMAND hostnumerics-core-tests)

add_executable(
    hostnumerics-scalar-codec-tests
    binary_float_test.cpp
    scalar_codec_test.cpp
    scalar_conversion_test.cpp
    scalar_integer_codec_test.cpp
    scalar_rounding_test.cpp
    scalar_type_test.cpp
)
target_link_libraries(hostnumerics-scalar-codec-tests PRIVATE roc::hostnumerics-core)
add_test(NAME hostnumerics-scalar-codec-tests COMMAND hostnumerics-scalar-codec-tests)

if(TARGET hostnumerics-blas)
    add_executable(hostnumerics-blas-tests blas_backend_test.cpp)
    target_link_libraries(
        hostnumerics-blas-tests
        PRIVATE
            roc::hostnumerics-blas
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
include(GoogleTest)
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
        "-DPACKAGE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/package"
        "-DPACKAGE_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/amd-gpu-layout-package-test"
        "-DTEST_GENERATOR=${CMAKE_GENERATOR}"
        "-DTEST_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DTEST_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DTEST_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
        "-DTEST_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
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
