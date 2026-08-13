# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCMxLayoutTransforms VERSION 0.1.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(ROCMX_LAYOUT_TRANSFORMS_STANDALONE ON)
else()
    set(ROCMX_LAYOUT_TRANSFORMS_STANDALONE OFF)
endif()

option(
    ROCMX_LAYOUT_TRANSFORMS_BUILD_TESTING
    "Build ROCMxLayoutTransforms tests."
    ${ROCMX_LAYOUT_TRANSFORMS_STANDALONE}
)
option(
    ROCMX_LAYOUT_TRANSFORMS_ENABLE_FETCH
    "Allow ROCMxLayoutTransforms tests to fetch missing dependencies."
    ${ROCMX_LAYOUT_TRANSFORMS_STANDALONE}
)
option(
    ROCMX_LAYOUT_TRANSFORMS_ENABLE_OPENMP
    "Use OpenMP for large layout transformations when it is available."
    ON
)

add_library(mx-layout-transforms INTERFACE)
add_library(roc::mx-layout-transforms ALIAS mx-layout-transforms)

target_compile_features(mx-layout-transforms INTERFACE cxx_std_17)
target_include_directories(
    mx-layout-transforms
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(
    mx-layout-transforms
    INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/mx_layout_transforms/pre_swizzle.hpp"
)
set_target_properties(
    mx-layout-transforms
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS ON
)

set(ROCMX_LAYOUT_TRANSFORMS_WITH_OPENMP OFF)
if(ROCMX_LAYOUT_TRANSFORMS_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(mx-layout-transforms INTERFACE OpenMP::OpenMP_CXX)
        set(ROCMX_LAYOUT_TRANSFORMS_WITH_OPENMP ON)
    endif()
endif()

if(ROCMX_LAYOUT_TRANSFORMS_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

install(
    TARGETS mx-layout-transforms
    EXPORT ROCMxLayoutTransformsTargets
    FILE_SET HEADERS
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT devel
)

set(
    ROCMX_LAYOUT_TRANSFORMS_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/ROCMxLayoutTransforms"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROCMxLayoutTransformsConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ROCMxLayoutTransformsConfig.cmake"
    INSTALL_DESTINATION "${ROCMX_LAYOUT_TRANSFORMS_INSTALL_CMAKEDIR}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ROCMxLayoutTransformsConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(
    EXPORT ROCMxLayoutTransformsTargets
    FILE ROCMxLayoutTransformsTargets.cmake
    DESTINATION "${ROCMX_LAYOUT_TRANSFORMS_INSTALL_CMAKEDIR}"
    NAMESPACE roc::
    COMPONENT devel
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/ROCMxLayoutTransformsConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/ROCMxLayoutTransformsConfigVersion.cmake"
    DESTINATION "${ROCMX_LAYOUT_TRANSFORMS_INSTALL_CMAKEDIR}"
    COMPONENT devel
)
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md"
    DESTINATION "${CMAKE_INSTALL_DOCDIR}"
    COMPONENT docs
)
