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

# --- Default install prefix ---
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")

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

# NOTE(hipccl3): libhipcxx is a git submodule (see ../.gitmodules), pinned to
# the tip of ROCm/libhipcxx's amd-develop branch at the time it was added -
# not a filter-repo'd copy like rocprim/hipcub/rocthrust, so it doesn't carry
# preserved history the same way (it's tracked by its own upstream commit
# instead). Its own CMakeLists.txt already supports being add_subdirectory()'d
# cleanly: when nested (not the top-level project) it skips all of its own
# testing/packaging/install logic and just defines a libhipcxx::libhipcxx
# INTERFACE target (see libhipcxx/cmake/libhipcxxAddSubdir.cmake) - no naming
# collisions or CMAKE_CXX_FLAGS meddling to guard against here, unlike
# rocprim/hipcub/rocthrust.
#
# NOT YET DONE: libhipcxx isn't part of the unified CPack package or the
# find_package(hipccl) config below - only the build (add_subdirectory) is
# wired up so far.
option(HIPCCL_BUILD_LIBHIPCXX "Build libhipcxx as part of the unified hipCCL build" OFF)

# One switch to build everything, matching the "build the entire hipccl
# project" requirement.
#
# NOTE(hipccl3): deliberately does NOT include HIPCCL_BUILD_LIBHIPCXX, even
# though the option is now functional (see above).
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

if(HIPCCL_BUILD_LIBHIPCXX)
    # NOTE(hipccl3): libhipcxx is a git submodule (see ../.gitmodules) - an
    # uninitialized submodule leaves this directory present but empty, which
    # would otherwise surface as a confusing failure deep inside
    # add_subdirectory(libhipcxx) (e.g. "add_subdirectory given source
    # ... which is not an existing directory" or a missing-CMakeLists.txt
    # error with no obvious cause). Check for a file only present once the
    # submodule is actually checked out, and fail with an actionable message
    # instead.
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libhipcxx/CMakeLists.txt")
        message(FATAL_ERROR
            "HIPCCL_BUILD_LIBHIPCXX is ON, but the libhipcxx submodule "
            "(projects/hipccl/hipccl3/libhipcxx) does not look initialized. "
            "Run 'git submodule update --init projects/hipccl/hipccl3/libhipcxx' "
            "(or 'git submodule update --init --recursive' from the "
            "repository root) and re-configure."
        )
    endif()

    add_subdirectory_with_message(
        COMPONENT libhipcxx
        PREFIX_PATH .
        EXPECT_TARGET libhipcxx::libhipcxx
    )
endif()

# --- Unified packaging ---
# NOTE(hipccl3): mirrors NVIDIA CCCL's single cuda-cccl package (no separate
# libcub/libthrust/libcudacxx system packages) - one hipccl package instead of
# three (rocprim-dev, hipcub-dev, rocthrust-dev).
#
# rocprim's own CMakeLists.txt already skips its packaging block when nested
# (ROCPRIM_PROJECT_IS_TOP_LEVEL, based on the same CMAKE_CURRENT_SOURCE_DIR/
# CMAKE_SOURCE_DIR check used below). hipcub's and rocthrust's own
# CMakeLists.txt did NOT have that guard - each unconditionally called
# rocm_create_package() (which triggers include(CPack)) regardless of
# nesting, so building all three together in one configure meant whichever
# ran last (rocthrust) silently won and hipcub's package definition was
# discarded. Both have been given the same top-level guard, matching
# rocprim's existing pattern, and their internal rocprim-dev/rocprim-devel
# cross-dependency declarations are skipped too, since rocprim's headers are
# now bundled into this same package rather than being a separate one.
#
# This block re-declares the one dependency that's real (HIP runtime,
# previously declared only by rocprim's own now-skipped block) and creates
# the single combined package.
#
# NOTE(hipccl3): unconditional (not guarded by a "top-level" check). This
# root CMakeLists.txt is never meant to be add_subdirectory()'d as a mere
# sub-component of something else the way rocprim/hipcub/rocthrust are - so
# the unified package should always be produced whenever this file runs,
# regardless of what (if anything) sits above it (e.g. the
# projects/hipccl/CMakeLists.txt layout selector).
#
# NOT YET DONE / needs real review before treating this as more than a
# starting point:
#  - License aggregation (MIT + BSD-3-Clause + Apache-2.0, see ./LICENSE) is a
#    best-effort placeholder, not a verified legal/compliance declaration.
#  - libhipcxx isn't part of this package yet (not wired into the build at
#    all - see HIPCCL_BUILD_LIBHIPCXX above).
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
    DESCRIPTION "hipCCL: unified rocPRIM, hipCUB, and rocThrust package"
    MAINTAINER "hipccl-maintainer@amd.com"
    HEADER_ONLY
)

# --- Unified CMake package config (find_package(hipccl)) ---
# NOTE(hipccl3): mirrors NVIDIA CCCL's find_package(CCCL COMPONENTS [Thrust]
# [CUB] [libcudacxx]) entry point. This is a *separate* mechanism from the
# CPack packaging above - it controls what `find_package(hipccl)` resolves to
# (installed under ${CMAKE_INSTALL_LIBDIR}/cmake/hipccl/), not what `make
# package` produces. rocprim/hipcub/rocthrust's own rocm_export_targets()
# calls are untouched and keep installing their own independent
# <name>-config.cmake files (lib/cmake/rocprim/, lib/cmake/hipcub/,
# lib/cmake/rocthrust/) unconditionally, same as before - find_package(rocprim)
# etc. keep working standalone, exactly like CCCL keeps find_package(Thrust)
# etc. working. See cmake/package/hipccl-config.cmake.in for the delegation
# logic (find_dependency() into each component's own config) and the
# hipccl::hipccl umbrella target.
#
# NOTE(hipccl3): unconditional for the same reason as the packaging block
# above - this file is never "just a sub-component", so find_package(hipccl)
# support should always be generated whenever this CMakeLists.txt runs.
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
