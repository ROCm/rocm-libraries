# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Unified hipCCL superbuild (rocPRIM + hipCUB + rocThrust + libhipcxx).
#
# This mirrors the role NVIDIA/cccl plays for cub/thrust/libcudacxx: a single
# root project that can build all four components together, with one unified
# version and one unified install layout (<prefix>/include/hipccl/<component>).
#
# STATUS: first-pass scaffold. See inline NOTE(hipccl3) comments for the
# specific pieces that are known-incomplete placeholders rather than fully
# validated behavior.

cmake_minimum_required(VERSION 3.25.2)

# --- Unified hipCCL version ---
# NOTE(hipccl3): placeholder starting point. rocprim/hipcub/rocthrust today
# each ship their own VERSION_STRING (currently all "4.7.0" on develop); once
# this superbuild is the primary way these projects are released, bump this
# instead of the per-project values. Each of rocprim/hipcub/rocthrust's own
# CMakeLists.txt has been given a `if(NOT DEFINED VERSION_STRING)` guard so
# this value propagates down; libhipcxx has not been touched/wired in yet
# (see HIPCCL_BUILD_LIBHIPCXX below).
set(HIPCCL_VERSION_MAJOR 0)
set(HIPCCL_VERSION_MINOR 1)
set(HIPCCL_VERSION_PATCH 0)
set(HIPCCL_VERSION "${HIPCCL_VERSION_MAJOR}.${HIPCCL_VERSION_MINOR}.${HIPCCL_VERSION_PATCH}")

project(hipccl VERSION ${HIPCCL_VERSION} LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(GNUInstallDirs)
include(add_subdirectory_with_message)
include(fetch_rocm_cmake)
# NOTE(hipccl3): shared_third_party.cmake (copied alongside for parity with
# rocm-libraries) declares rocm_libs_declare_shared_deps() for hipdnn/miopen's
# nlohmann_json/GTest fetches. None of rocprim/hipcub/rocthrust/libhipcxx use
# it today, so it is not include()'d here - left in cmake/modules/ in case a
# future component needs the same dedup pattern.

# --- Unified install layout ---
# All hipCCL headers install under <prefix>/include/hipccl/<component>, e.g.
# /opt/rocm/include/hipccl/thrust, /opt/rocm/include/hipccl/rocprim.
# Verified: rocprim, hipcub, and rocthrust's install(...) rules all reference
# ${CMAKE_INSTALL_INCLUDEDIR} rather than a hardcoded path, so overriding it
# here is sufficient for those three (rocthrust's thrust/CMakeLists.txt even
# already appends its own "/thrust" suffix, matching the requested layout
# exactly). libhipcxx's install rules have NOT been audited against this
# override yet (see HIPCCL_BUILD_LIBHIPCXX below) - NOTE(hipccl3) placeholder.
set(CMAKE_INSTALL_INCLUDEDIR "include/hipccl" CACHE PATH
    "Unified hipCCL include install directory" FORCE)

# --- Unified version propagation ---
# Force VERSION_STRING as a CACHE variable so it wins over each subproject's
# own `set(VERSION_STRING "4.7.0")` (which is guarded with
# `if(NOT DEFINED VERSION_STRING)` in the hipccl3 copies of rocprim/hipcub/
# rocthrust specifically for this purpose).
set(VERSION_STRING "${HIPCCL_VERSION}" CACHE STRING "Unified hipCCL component version" FORCE)

# --- Component selection ---
option(HIPCCL_BUILD_ROCPRIM "Build rocPRIM as part of the unified hipCCL build" ON)
option(HIPCCL_BUILD_HIPCUB "Build hipCUB as part of the unified hipCCL build" ON)
option(HIPCCL_BUILD_ROCTHRUST "Build rocThrust as part of the unified hipCCL build" ON)
# NOTE(hipccl3): libhipcxx is present under projects/hipccl/hipccl3/libhipcxx
# (pinned to ROCm/libhipcxx@5ac455d737937ba2dfd1a4e85ad13f19a775f692) but is
# NOT wired into this superbuild yet. Its CMakeLists.txt is structured around
# standalone lit-based testing rather than being a consumable add_subdirectory
# target the way rocprim/hipcub/rocthrust are, and its install layout/version
# scheme has not been reconciled with the unified layout above. Defaulting
# this OFF and leaving the add_subdirectory() call commented out below is the
# deliberate placeholder for this first pass.
option(HIPCCL_BUILD_LIBHIPCXX "Build/test libhipcxx as part of the unified hipCCL build" OFF)

# One switch to build everything, matching the "build the entire hipccl
# project" requirement.
option(HIPCCL_BUILD_ALL "Build every hipCCL component (rocprim, hipcub, rocthrust, libhipcxx)" OFF)
if(HIPCCL_BUILD_ALL)
    set(HIPCCL_BUILD_ROCPRIM ON CACHE BOOL "" FORCE)
    set(HIPCCL_BUILD_HIPCUB ON CACHE BOOL "" FORCE)
    set(HIPCCL_BUILD_ROCTHRUST ON CACHE BOOL "" FORCE)
    set(HIPCCL_BUILD_LIBHIPCXX ON CACHE BOOL "" FORCE)
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

if(HIPCCL_BUILD_LIBHIPCXX)
    message(FATAL_ERROR
        "HIPCCL_BUILD_LIBHIPCXX is not yet implemented in the hipccl3 superbuild. "
        "libhipcxx's CMakeLists.txt (projects/hipccl/hipccl3/libhipcxx/CMakeLists.txt) "
        "needs to be reconciled with the unified version/install scheme before "
        "it can be added here. See the NOTE(hipccl3) comment above HIPCCL_BUILD_LIBHIPCXX."
    )
    # add_subdirectory(libhipcxx)
endif()
