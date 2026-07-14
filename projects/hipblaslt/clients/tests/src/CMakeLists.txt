# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hipblaslt-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/caching_library_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/secure_env_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/auxiliary_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix_transform_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_ext_op.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/streamk5_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ulp_gtest.cpp
)

# secure_env_test.cpp (ROCM-26729) is a white-box unit test that includes the
# header-only rocblaslt_secure_env.hpp directly. That header lives with the
# other internal rocblaslt sources, whose directory is only added to the test
# include path in coverage builds, so expose it to the test target here as well
# so the smoke-tier guard compiles and runs on the normal PR gate.
target_include_directories(hipblaslt-test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include
)
