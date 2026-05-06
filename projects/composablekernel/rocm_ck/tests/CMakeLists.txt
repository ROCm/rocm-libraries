# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocm_ck tests
#
# Test tiers:
#   ROCM_CK_SMOKE  — Fast host-only tests (< 1s total). No GPU required.
#   ROCM_CK_KERNEL — GPU kernel tests. Require HIP and a GPU.
#
# Usage:
#   ninja smoke-rocm-ck        # build + run smoke tests
#   ninja build-smoke-rocm-ck  # build only (no run)
#   ninja check-rocm-ck        # run all rocm_ck tests
#
#   ctest -L ROCM_CK_SMOKE --output-on-failure

# Google Test (via CK's FetchContent wrapper)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/gtest.cmake)

# ---------------------------------------------------------------------------
# Helper function — reusable per-test setup
# ---------------------------------------------------------------------------
function(add_rocm_ck_test test_name)
    add_executable(${test_name} ${ARGN})
    target_link_libraries(${test_name} PRIVATE rocm_ck GTest::gtest_main GTest::gmock)
    target_compile_options(${test_name} PRIVATE
        -Wno-global-constructors           # GTest registration macros
        -Wno-undef                         # GTest internal headers
        -Wno-zero-as-null-pointer-constant # C++20 <=> comparisons to 0
    )
endfunction()

# ---------------------------------------------------------------------------
# Smoke tests (fast, host-only, no GPU)
# ---------------------------------------------------------------------------
set(ROCM_CK_SMOKE_TESTS
    unit/test_args.cpp
    unit/test_datatype.cpp
    unit/test_fixed_string.cpp
    unit/test_index_t.cpp
    unit/test_layout.cpp
    unit/test_physical_tensor.cpp
)

set(ROCM_CK_SMOKE_TARGETS)
foreach(test_source ${ROCM_CK_SMOKE_TESTS})
    get_filename_component(test_name ${test_source} NAME_WLE)
    add_rocm_ck_test(${test_name} ${test_source})
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES LABELS "ROCM_CK_SMOKE")
    list(APPEND ROCM_CK_SMOKE_TARGETS ${test_name})
endforeach()

# test_index_t verifies rocm_ck index types match ck_tile
target_include_directories(test_index_t PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# ---------------------------------------------------------------------------
# Convenience targets
# ---------------------------------------------------------------------------
add_custom_target(build-smoke-rocm-ck DEPENDS ${ROCM_CK_SMOKE_TARGETS})

add_custom_target(smoke-rocm-ck
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "ROCM_CK_SMOKE"
    DEPENDS build-smoke-rocm-ck
    USES_TERMINAL
    COMMENT "Running rocm_ck smoke tests...")

add_custom_target(check-rocm-ck
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "ROCM_CK"
    DEPENDS build-smoke-rocm-ck
    USES_TERMINAL
    COMMENT "Running all rocm_ck tests...")
