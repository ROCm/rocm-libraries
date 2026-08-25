# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(GTest 1.12.1 QUIET)
if(NOT GTest_FOUND)
    if(HOST_VALIDATION_ENABLE_FETCH)
        include(FetchContent)
        # GoogleTest is a test-only implementation detail. Its default install
        # rules would otherwise contaminate the host-validation package.
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
        )
        FetchContent_MakeAvailable(googletest)
    else()
        message(FATAL_ERROR "Host-validation tests require GTest 1.12.1 or newer.")
    endif()
endif()

add_executable(host-validation-tests host_validation_test.cpp)
target_link_libraries(host-validation-tests PRIVATE roc::host-validation)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(host-validation-tests PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(
        host-validation-tests
        PRIVATE
            HOST_VALIDATION_TEST_OPENMP=1
    )
endif()
add_test(NAME host-validation-tests COMMAND host-validation-tests)

add_executable(host-validation-generation-recipe-tests generation_recipe_test.cpp)
target_link_libraries(
    host-validation-generation-recipe-tests
    PRIVATE
        roc::host-validation
)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(
        host-validation-generation-recipe-tests
        PRIVATE
            OpenMP::OpenMP_CXX
    )
endif()
add_test(
    NAME host-validation-generation-recipe-tests
    COMMAND host-validation-generation-recipe-tests
)

add_executable(host-validation-blocked-tests blocked_backend_test.cpp)
target_link_libraries(host-validation-blocked-tests PRIVATE roc::host-validation-blocked)
add_test(NAME host-validation-blocked-tests COMMAND host-validation-blocked-tests)
add_test(
    NAME host-validation-blocked-tests-four-threads
    COMMAND host-validation-blocked-tests
)
set_tests_properties(
    host-validation-blocked-tests-four-threads
    PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=4;OMP_DYNAMIC=FALSE"
        PROCESSORS 4
)

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
    add_test(
        NAME host-validation-blas-tests-one-thread
        COMMAND host-validation-blas-tests
    )
    add_test(
        NAME host-validation-blas-tests-multi-thread
        COMMAND host-validation-blas-tests
    )
    set(_host_validation_blas_one_thread_environment)
    set(_host_validation_blas_multi_thread_environment)
    foreach(thread_variable
            OMP_NUM_THREADS
            OPENBLAS_NUM_THREADS
            GOTO_NUM_THREADS
            BLIS_NUM_THREADS
            MKL_NUM_THREADS
            VECLIB_MAXIMUM_THREADS)
        list(
            APPEND
            _host_validation_blas_one_thread_environment
            "${thread_variable}=1"
        )
        list(
            APPEND
            _host_validation_blas_multi_thread_environment
            "${thread_variable}=2"
        )
    endforeach()
    set_tests_properties(
        host-validation-blas-tests-one-thread
        PROPERTIES
            ENVIRONMENT "${_host_validation_blas_one_thread_environment}"
    )
    set_tests_properties(
        host-validation-blas-tests-multi-thread
        PROPERTIES
            ENVIRONMENT "${_host_validation_blas_multi_thread_environment}"
            PROCESSORS 2
    )
endif()

if(TARGET host-validation-mx)
    add_executable(host-validation-mx-tests mx_test.cpp)
    target_link_libraries(
        host-validation-mx-tests
        PRIVATE
            roc::host-validation-mx
    )
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(
            host-validation-mx-tests
            PRIVATE
                OpenMP::OpenMP_CXX
        )
    endif()
    add_test(NAME host-validation-mx-tests COMMAND host-validation-mx-tests)
endif()

add_executable(host-validation-amd-gpu-layout-tests amd_gpu_layout_test.cpp)
target_link_libraries(
    host-validation-amd-gpu-layout-tests
    PRIVATE
        GTest::gtest_main
        roc::host-validation-amd-gpu-layout
)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(
        host-validation-amd-gpu-layout-tests
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
    )
endif()
include(GoogleTest)
gtest_discover_tests(
    host-validation-amd-gpu-layout-tests
    DISCOVERY_MODE PRE_TEST
)

add_test(
    NAME host-validation-core-boundary
    COMMAND
        "${CMAKE_COMMAND}"
        "-DCORE_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../include/roc/host_validation"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_core_boundary.cmake"
)

add_test(
    NAME host-validation-component-boundary
    COMMAND
        "${CMAKE_COMMAND}"
        -DHOST_VALIDATION_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_component_boundary.cmake"
)

add_test(
    NAME host-validation-amd-gpu-layout-boundary
    COMMAND
        "${CMAKE_COMMAND}"
        -DHOST_VALIDATION_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_amd_gpu_layout_boundary.cmake"
)

add_test(
    NAME host-validation-amd-gpu-layout-package
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

    add_executable(
        host-validation-gemm-benchmark
        gemm_benchmark.cpp
    )
    target_link_libraries(
        host-validation-gemm-benchmark
        PRIVATE
            roc::host-validation-blocked
    )
endif()
