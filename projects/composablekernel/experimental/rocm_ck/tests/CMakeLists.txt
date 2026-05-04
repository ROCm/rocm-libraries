# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocm_ck unit tests — host-only (no HIP, no GPU).
#
# Build:
#   cmake -B build -S . -G Ninja
#   ninja -C build
#   ctest --test-dir build --output-on-failure
#
# Or from parent build:
#   ninja -C build rocm_ck_tests

cmake_minimum_required(VERSION 3.21)
project(rocm_ck_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# =============================================================================
# Google Test (via CK's FetchContent wrapper)
# =============================================================================

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/gtest.cmake)

# =============================================================================
# rocm_ck headers (INTERFACE library)
# =============================================================================

if(NOT TARGET rocm_ck)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../include rocm_ck_build)
endif()

# =============================================================================
# Host-side unit tests
# =============================================================================

set(ROCM_CK_HOST_TESTS
    test_arch_properties.cpp
    test_args.cpp
    test_datatype.cpp
    test_datatype_convert.cpp
    test_elementwise_spec.cpp
    test_fmha_bwd_common.cpp
    test_fmha_bwd_convert_dq.cpp
    test_fmha_bwd_dqdkdv.cpp
    test_fmha_bwd_ograd_dot_o.cpp
    test_fmha_bwd_validate_args.cpp
    test_gemm_spec.cpp
    test_layout.cpp
    test_physical_tensor.cpp
    test_resolve.cpp
    test_schema_compatibility.cpp
    test_signature.cpp
    test_validate.cpp
    test_verify.cpp
)

# Convenience target: `ninja rocm_ck_tests` builds all host tests
add_custom_target(rocm_ck_tests)

foreach(test_source ${ROCM_CK_HOST_TESTS})
    get_filename_component(test_name ${test_source} NAME_WE)

    add_executable(${test_name} ${test_source})

    target_link_libraries(${test_name} PRIVATE rocm_ck GTest::gtest_main)

    target_compile_options(${test_name} PRIVATE
        -Wno-global-constructors
        -Wno-undef
    )

    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES LABELS "rocm_ck")

    add_dependencies(rocm_ck_tests ${test_name})
endforeach()

# =============================================================================
# FMHA BWD compat test (needs example directory for registry.hpp)
# =============================================================================

add_executable(test_fmha_bwd_compat test_fmha_bwd_compat.cpp)
target_link_libraries(test_fmha_bwd_compat PRIVATE rocm_ck GTest::gtest_main)
target_include_directories(test_fmha_bwd_compat PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/06_rocm_ck_fmha_bwd)
target_compile_options(test_fmha_bwd_compat PRIVATE
    -Wno-global-constructors -Wno-undef)
add_test(NAME test_fmha_bwd_compat COMMAND test_fmha_bwd_compat)
set_tests_properties(test_fmha_bwd_compat PROPERTIES LABELS "rocm_ck")
add_dependencies(rocm_ck_tests test_fmha_bwd_compat)

# =============================================================================
# Compile-error tests (expected failures)
# =============================================================================

add_subdirectory(compile_fail)

# =============================================================================
# GPU integration tests (opt-in, requires HIP)
# =============================================================================

option(ROCM_CK_GPU_TESTS "Build GPU integration tests (requires HIP)" OFF)
if(ROCM_CK_GPU_TESTS)
    add_subdirectory(gpu)
endif()
