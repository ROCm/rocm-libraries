# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.11)
project(libs-common VERSION 1.2.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(dependencies)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path prefix" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix" FORCE)
    endif()
endif()

option(LIBS_COMMON_ENABLE_HIPBLAS "Enable hipBLAS common library" ON)
option(LIBS_COMMON_ENABLE_HIPSPARSE "Enable hipSPARSE common library" ON)

message(STATUS "LIBS_COMMON_ENABLE_HIPBLAS: ${LIBS_COMMON_ENABLE_HIPBLAS}")
message(STATUS "LIBS_COMMON_ENABLE_HIPSPARSE: ${LIBS_COMMON_ENABLE_HIPSPARSE}")

if(LIBS_COMMON_ENABLE_HIPBLAS)
    message(STATUS "Building hipblas-common")
    add_library(hipblas-common INTERFACE)
    add_library(roc::hipblas-common ALIAS hipblas-common)

    target_sources(
        hipblas-common
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/library/include/hipblas-common/hipblas-common.h>
    )
    target_include_directories(
        hipblas-common INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/library/include>
                                 $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    rocm_install_targets(
        TARGETS hipblas-common EXPORT hipblas-common-targets INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/library/include
    )
    rocm_export_targets(
        TARGETS
        roc::hipblas-common
        NAME
        hipblas-common
        EXPORT
        hipblas-common-targets
        NAMESPACE
        roc::
    )
endif()

if(LIBS_COMMON_ENABLE_HIPSPARSE)
    message(STATUS "Building hipsparse-common")
    add_library(hipsparse-common INTERFACE)
    add_library(roc::hipsparse-common ALIAS hipsparse-common)

    target_sources(
        hipsparse-common
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/library/include/hipsparse-common/hipsparse-common.h>
    )
    target_include_directories(
        hipsparse-common INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/library/include>
                                   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    rocm_install_targets(
        TARGETS hipsparse-common EXPORT hipsparse-common-targets INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/library/include
    )
    rocm_export_targets(
        TARGETS
        roc::hipsparse-common
        NAME
        hipsparse-common
        EXPORT
        hipsparse-common-targets
        NAMESPACE
        roc::
    )
endif()
