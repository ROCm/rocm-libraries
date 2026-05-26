# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hipblaslt-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mat_utils_mx_scale_format_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/auxiliary_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix_transform_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_ext_op.cpp
)

# The MX scale-format unit test pulls in rocblaslt_mx_scale_format.hpp, which
# lives under library/src/amd_detail/rocblaslt/include alongside the
# rocblaslt-types public header. We intentionally do NOT add the sibling
# library/src/amd_detail/rocblaslt/src/include path here - it contains a
# private "utility.hpp" that would otherwise shadow the client's own
# clients/common/include/utility.hpp.
target_include_directories(hipblaslt-test
    PRIVATE
        "${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include"
)
