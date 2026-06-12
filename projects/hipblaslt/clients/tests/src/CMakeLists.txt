# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hipblaslt-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/auxiliary_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix_transform_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_ext_op.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fp64_emulation_gtest.cpp
)

# fp64_emulation_gtest.cpp needs the rocblaslt-private fp64_emulation.hpp (and
# its rocblaslt.h dependency).  Scope those include dirs to this one source:
# rocblaslt/src/include also contains a utility.hpp that would otherwise shadow
# clients/common/include/utility.hpp for every other test source.
#
# hipblaslt-test is defined in clients/CMakeLists.txt, so the source-file
# property must be set in that target's directory scope - hence TARGET_DIRECTORY
# (a plain set_source_files_properties here would be silently ignored).
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/fp64_emulation_gtest.cpp
    TARGET_DIRECTORY hipblaslt-test
    PROPERTIES INCLUDE_DIRECTORIES
        "${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include;${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include"
)
