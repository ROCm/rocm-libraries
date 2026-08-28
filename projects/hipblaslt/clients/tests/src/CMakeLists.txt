# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hipblaslt-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/caching_library_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/secure_env_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/arch_revision_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/auxiliary_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_stats_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/host_numerics_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hip_buffer_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_problem_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_validation_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix_transform_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_ext_op.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/streamk5_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/streamk_multistream_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/synchronizer_size_check_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/row_uniformity_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ulp_gtest.cpp
)

# NOTE: secure_env_gtest.cpp (ROCM-26729) intentionally includes the header-only
# rocblaslt_secure_env.hpp by relative path rather than via a target include
# directory. Adding the internal rocblaslt/src/include dir to hipblaslt-test
# shadows the clients' own "utility.hpp" with the library's internal one and
# breaks the build of unrelated test sources, so it must not be added here.
