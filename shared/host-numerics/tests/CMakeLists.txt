# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(GTest 1.12.1 QUIET)
if(NOT GTest_FOUND)
    if(HOST_NUMERICS_ENABLE_FETCH)
        include(FetchContent)
        # GoogleTest is a test-only implementation detail. Its default install
        # rules would otherwise contaminate the host-numerics package.
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
        )
        FetchContent_MakeAvailable(googletest)
    else()
        message(FATAL_ERROR "Host-numerics tests require GTest 1.12.1 or newer.")
    endif()
endif()

add_executable(host-numerics-tests host_numerics_test.cpp)
target_link_libraries(host-numerics-tests PRIVATE roc::host-numerics)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(host-numerics-tests PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(
        host-numerics-tests
        PRIVATE
            HOST_NUMERICS_TEST_OPENMP=1
    )
endif()
add_test(NAME host-numerics-tests COMMAND host-numerics-tests)

add_executable(host-numerics-generation-recipe-tests generation_recipe_test.cpp)
target_link_libraries(
    host-numerics-generation-recipe-tests
    PRIVATE
        roc::host-numerics
)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(
        host-numerics-generation-recipe-tests
        PRIVATE
            OpenMP::OpenMP_CXX
    )
endif()
add_test(
    NAME host-numerics-generation-recipe-tests
    COMMAND host-numerics-generation-recipe-tests
)

add_executable(host-numerics-blocked-tests blocked_backend_test.cpp)
target_link_libraries(host-numerics-blocked-tests PRIVATE roc::host-numerics)
add_test(NAME host-numerics-blocked-tests COMMAND host-numerics-blocked-tests)
add_test(
    NAME host-numerics-blocked-tests-four-threads
    COMMAND host-numerics-blocked-tests
)
set_tests_properties(
    host-numerics-blocked-tests-four-threads
    PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=4;OMP_DYNAMIC=FALSE"
        PROCESSORS 4
)

add_executable(host-numerics-core-tests tensor_core_test.cpp)
target_link_libraries(host-numerics-core-tests PRIVATE roc::host-numerics-core)
add_test(NAME host-numerics-core-tests COMMAND host-numerics-core-tests)

add_executable(host-numerics-scalar-codec-tests scalar_codec_test.cpp)
target_link_libraries(host-numerics-scalar-codec-tests PRIVATE roc::host-numerics-core)
add_test(NAME host-numerics-scalar-codec-tests COMMAND host-numerics-scalar-codec-tests)

if(TARGET host-numerics-blas)
    add_executable(host-numerics-blas-tests blas_backend_test.cpp)
    target_link_libraries(
        host-numerics-blas-tests
        PRIVATE
            roc::host-numerics-blas
    )
    add_test(
        NAME host-numerics-blas-tests-one-thread
        COMMAND host-numerics-blas-tests
    )
    add_test(
        NAME host-numerics-blas-tests-multi-thread
        COMMAND host-numerics-blas-tests
    )
    set(_host_numerics_blas_one_thread_environment)
    set(_host_numerics_blas_multi_thread_environment)
    foreach(thread_variable
            OMP_NUM_THREADS
            OPENBLAS_NUM_THREADS
            GOTO_NUM_THREADS
            BLIS_NUM_THREADS
            MKL_NUM_THREADS
            VECLIB_MAXIMUM_THREADS)
        list(
            APPEND
            _host_numerics_blas_one_thread_environment
            "${thread_variable}=1"
        )
        list(
            APPEND
            _host_numerics_blas_multi_thread_environment
            "${thread_variable}=2"
        )
    endforeach()
    set_tests_properties(
        host-numerics-blas-tests-one-thread
        PROPERTIES
            ENVIRONMENT "${_host_numerics_blas_one_thread_environment}"
    )
    set_tests_properties(
        host-numerics-blas-tests-multi-thread
        PROPERTIES
            ENVIRONMENT "${_host_numerics_blas_multi_thread_environment}"
            PROCESSORS 2
    )
endif()

add_executable(host-numerics-mx-tests mx_test.cpp)
target_link_libraries(
    host-numerics-mx-tests
    PRIVATE
        roc::host-numerics
)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(
        host-numerics-mx-tests
        PRIVATE
            OpenMP::OpenMP_CXX
    )
endif()
add_test(NAME host-numerics-mx-tests COMMAND host-numerics-mx-tests)

add_executable(host-numerics-amd-gpu-layout-tests amd_gpu_layout_test.cpp)
target_link_libraries(
    host-numerics-amd-gpu-layout-tests
    PRIVATE
        GTest::gtest_main
        roc::host-numerics-amd-gpu-layout
)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(
        host-numerics-amd-gpu-layout-tests
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
    )
endif()
include(GoogleTest)
gtest_discover_tests(
    host-numerics-amd-gpu-layout-tests
    DISCOVERY_MODE PRE_TEST
)

add_test(
    NAME host-numerics-core-boundary
    COMMAND
        "${CMAKE_COMMAND}"
        "-DCORE_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../include/roc/host_numerics"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_core_boundary.cmake"
)

add_test(
    NAME host-numerics-component-boundary
    COMMAND
        "${CMAKE_COMMAND}"
        -DHOST_NUMERICS_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_component_boundary.cmake"
)

add_test(
    NAME host-numerics-amd-gpu-layout-boundary
    COMMAND
        "${CMAKE_COMMAND}"
        -DHOST_NUMERICS_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_amd_gpu_layout_boundary.cmake"
)

add_test(
    NAME host-numerics-amd-gpu-layout-package
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

if(HOST_NUMERICS_BUILD_BENCHMARKS)
    find_package(OpenMP QUIET COMPONENTS CXX)
    add_executable(
        host-numerics-structured-sparsity-benchmark
        structured_sparsity_benchmark.cpp
    )
    target_link_libraries(
        host-numerics-structured-sparsity-benchmark
        PRIVATE
            roc::host-numerics
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(
            host-numerics-structured-sparsity-benchmark
            PRIVATE
                OpenMP::OpenMP_CXX
        )
        target_compile_definitions(
            host-numerics-structured-sparsity-benchmark
            PRIVATE
                HOST_NUMERICS_BENCHMARK_OPENMP=1
        )
    endif()

    add_executable(
        host-numerics-comparison-benchmark
        comparison_benchmark.cpp
    )
    target_link_libraries(
        host-numerics-comparison-benchmark
        PRIVATE
            roc::host-numerics
    )

    add_executable(
        host-numerics-gemm-benchmark
        gemm_benchmark.cpp
    )
    target_link_libraries(
        host-numerics-gemm-benchmark
        PRIVATE
            roc::host-numerics
    )
endif()
