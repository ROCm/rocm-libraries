# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# hipCCL "legacy" superbuild (rocPRIM + hipCUB + rocThrust).
#
# Mirrors hipccl3/CMakeLists.txt's unified build/version/install/packaging
# behavior (one root project, one unified version, one unified install
# layout, one CPack package, one find_package(hipccl) entry point) so that
# building through the projects/hipccl/CMakeLists.txt layout selector with
# -DHIPCCL_BUILD_LEGACY=ON exposes the exact same CMake variable surface
# (HIPCCL_BUILD_ROCPRIM/HIPCUB/ROCTHRUST/ALL, etc.) as building the unified
# hipccl3 project. The difference is purely which copies of rocPRIM/hipCUB/
# rocThrust get built - this copy's library/algorithm code is the frozen,
# compatibility snapshot (see ../README.md), not actively kept in sync with
# upstream the way hipccl3's copies are intended to be.
#
# NOTE: unlike hipccl3, there is no libhipcxx here - hipccl2 never included a
# libhipcxx copy at all, so there is no HIPCCL_BUILD_LIBHIPCXX option.

cmake_minimum_required(VERSION 3.25.2)

# --- Unified hipCCL version ---
# Deliberately kept identical to hipccl3's HIPCCL_VERSION so that CI or other
# tooling built against one layout's version number doesn't need to special-
# case the other.
set(HIPCCL_VERSION_MAJOR 0)
set(HIPCCL_VERSION_MINOR 1)
set(HIPCCL_VERSION_PATCH 0)
set(HIPCCL_VERSION "${HIPCCL_VERSION_MAJOR}.${HIPCCL_VERSION_MINOR}.${HIPCCL_VERSION_PATCH}")

# --- Default install prefix ---
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")

project(hipccl VERSION ${HIPCCL_VERSION} LANGUAGES CXX)

# NOTE: shared with hipccl3 - these helper modules, the hipccl-config.cmake.in
# template below, and LICENSE are deduplicated at projects/hipccl/, not
# duplicated per-layout.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules")

include(GNUInstallDirs)
include(add_subdirectory_with_message)
include(fetch_rocm_cmake)
include(hipccl_finalize_package)

# --- Unified install layout ---
# All hipCCL headers install under <prefix>/include/hipccl/<component>, e.g.
# /opt/rocm/include/hipccl/thrust, /opt/rocm/include/hipccl/rocprim - same
# layout as hipccl3.
set(CMAKE_INSTALL_INCLUDEDIR "include/hipccl" CACHE PATH
    "Unified hipCCL include install directory" FORCE)

# --- Unified version propagation ---
# Force VERSION_STRING as a CACHE variable so it wins over each subproject's
# own `set(VERSION_STRING "4.7.0")` (guarded with
# `if(NOT DEFINED VERSION_STRING)` in these hipccl2 copies, same as hipccl3).
set(VERSION_STRING "${HIPCCL_VERSION}" CACHE STRING "Unified hipCCL component version" FORCE)

# --- Component selection ---
option(HIPCCL_BUILD_ROCPRIM "Build rocPRIM as part of the hipCCL legacy build" ON)
option(HIPCCL_BUILD_HIPCUB "Build hipCUB as part of the hipCCL legacy build" ON)
option(HIPCCL_BUILD_ROCTHRUST "Build rocThrust as part of the hipCCL legacy build" ON)

# One switch to build everything, matching hipccl3's HIPCCL_BUILD_ALL.
option(HIPCCL_BUILD_ALL "Build every hipCCL component (rocprim, hipcub, rocthrust)" OFF)
if(HIPCCL_BUILD_ALL)
    set(HIPCCL_BUILD_ROCPRIM ON CACHE BOOL "" FORCE)
    set(HIPCCL_BUILD_HIPCUB ON CACHE BOOL "" FORCE)
    set(HIPCCL_BUILD_ROCTHRUST ON CACHE BOOL "" FORCE)
endif()

if(HIPCCL_BUILD_ROCPRIM)
    add_subdirectory_with_message(
        COMPONENT rocprim
        PREFIX_PATH .
        EXPECT_TARGET rocprim
    )
endif()

if(HIPCCL_BUILD_HIPCUB)
    add_subdirectory_with_message(
        COMPONENT hipcub
        PREFIX_PATH .
        EXPECT_TARGET hipcub
    )
endif()

if(HIPCCL_BUILD_ROCTHRUST)
    add_subdirectory_with_message(
        COMPONENT rocthrust
        PREFIX_PATH .
        EXPECT_TARGET rocthrust
    )
endif()

# --- Unified packaging ---
# Mirrors hipccl3's root: one hipccl package instead of three (rocprim-dev,
# hipcub-dev, rocthrust-dev). rocPRIM/hipCUB/rocThrust's own standalone
# packaging is unconditionally disabled in these hipccl2 copies for the same
# reason as hipccl3's - see the Package section of each project's own
# CMakeLists.txt.
#
# NOT YET DONE / needs real review before treating this as more than a
# starting point:
#  - License aggregation (MIT + BSD-3-Clause + Apache-2.0, see ../LICENSE) is a
#    best-effort placeholder, not a verified legal/compliance declaration.
#  - Maintainer email is a placeholder, not a real assigned address/alias.
#
# NOTE: the mechanical CPack + find_package(hipccl) config calls (identical
# between hipccl2 and hipccl3 other than the DESCRIPTION string) are
# consolidated in cmake/modules/hipccl_finalize_package.cmake, shared with
# hipccl3/CMakeLists.txt.
hipccl_finalize_package(
    DESCRIPTION "hipCCL: unified rocPRIM, hipCUB, and rocThrust package (legacy layout)"
)
