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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(GNUInstallDirs)
include(add_subdirectory_with_message)
include(fetch_rocm_cmake)

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
#  - License aggregation (MIT + BSD-3-Clause + Apache-2.0, see ./LICENSE) is a
#    best-effort placeholder, not a verified legal/compliance declaration.
#  - Maintainer email is a placeholder, not a real assigned address/alias.
set(HIP_RUNTIME_MINIMUM 4.5.0)
rocm_package_add_dependencies(SHARED_DEPENDS "hip-runtime-amd >= ${HIP_RUNTIME_MINIMUM}")
rocm_package_add_deb_dependencies(STATIC_DEPENDS "hip-static-dev >= ${HIP_RUNTIME_MINIMUM}")
rocm_package_add_rpm_dependencies(STATIC_DEPENDS "hip-static-devel >= ${HIP_RUNTIME_MINIMUM}")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RPM_PACKAGE_LICENSE "MIT and BSD and ASL 2.0")
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}")

rocm_create_package(
    NAME hipccl
    DESCRIPTION "hipCCL: unified rocPRIM, hipCUB, and rocThrust package (legacy layout)"
    MAINTAINER "hipccl-maintainer@amd.com"
    HEADER_ONLY
)

# --- Unified CMake package config (find_package(hipccl)) ---
# Mirrors hipccl3's root - see cmake/package/hipccl-config.cmake.in for the
# delegation logic (find_dependency() into each component's own config) and
# the hipccl::hipccl umbrella target. rocPRIM/hipCUB/rocThrust's own
# rocm_export_targets() calls are untouched and keep installing their own
# independent <name>-config.cmake files unconditionally, so find_package(rocprim)
# etc. keep working standalone, exactly like hipccl3.
include(CMakePackageConfigHelpers)

set(HIPCCL_PACKAGE_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/hipccl")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package/hipccl-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/hipccl-config.cmake"
    INSTALL_DESTINATION "${HIPCCL_PACKAGE_CONFIG_INSTALL_DIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hipccl-config-version.cmake"
    VERSION "${HIPCCL_VERSION}"
    COMPATIBILITY AnyNewerVersion
    ARCH_INDEPENDENT
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/hipccl-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/hipccl-config-version.cmake"
    DESTINATION "${HIPCCL_PACKAGE_CONFIG_INSTALL_DIR}"
    COMPONENT devel
)
