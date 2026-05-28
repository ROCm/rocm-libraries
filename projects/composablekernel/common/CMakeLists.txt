# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# ck_common — header-only common types shared across CK Tile, rocm_ck, and dispatcher.
# C++17 with optional C++20 headers (arch_properties). No HIP/device dependencies.

add_library(ck_common INTERFACE)
target_include_directories(ck_common INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

rocm_install(
    DIRECTORY include/ck_common
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
