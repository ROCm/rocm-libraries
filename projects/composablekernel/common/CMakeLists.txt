# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# ck_common — header-only common types shared across CK Tile, rocm_ck, and dispatcher.
# C++17 with optional C++20 headers (arch_properties). No HIP/device dependencies.

add_library(ck_common INTERFACE)
# C++20 baseline. arch_properties.hpp requires C++20 (designated initializers,
# std::popcount, consteval); since it is a public header on ck_common, the
# whole target advertises C++20 so downstream consumers don't have to bump
# the standard themselves.
target_compile_features(ck_common INTERFACE cxx_std_20)
target_include_directories(ck_common INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Install / export only when rocm-cmake is in scope (parent CK super-build).
# When ck_common is consumed standalone via add_subdirectory (e.g. dispatcher
# built on its own), rocm_install is not defined -- skip install in that case.
if(COMMAND rocm_install)
    rocm_install(
        DIRECTORY include/ck_common
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # Export the INTERFACE target so downstream find_package(composable_kernel)
    # consumers can link against it directly rather than reaching for the include
    # path.
    rocm_install(TARGETS ck_common EXPORT composable_kernelTargets)
endif()
