# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostNumerics VERSION 0.1.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(HOST_NUMERICS_STANDALONE ON)
else()
    set(HOST_NUMERICS_STANDALONE OFF)
endif()

option(HOST_NUMERICS_BUILD_TESTING "Build host-numerics tests." ${HOST_NUMERICS_STANDALONE})
option(HOST_NUMERICS_BUILD_PYTHON "Build the host-numerics Python module."
       ${HOST_NUMERICS_STANDALONE})

add_library(host-numerics-core INTERFACE)
add_library(roc::host-numerics-core ALIAS host-numerics-core)
target_compile_features(host-numerics-core INTERFACE cxx_std_20)
target_include_directories(host-numerics-core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(host-numerics-core
    INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/version.hpp"
)

add_library(host-numerics INTERFACE)
add_library(roc::host-numerics ALIAS host-numerics)
target_link_libraries(host-numerics INTERFACE roc::host-numerics-core)

if(HOST_NUMERICS_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
if(HOST_NUMERICS_BUILD_PYTHON)
    add_subdirectory(python)
endif()

install(TARGETS host-numerics-core host-numerics
    EXPORT host-numerics-targets
    FILE_SET HEADERS DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(EXPORT host-numerics-targets
    FILE ROCHostNumericsTargets.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ROCHostNumerics"
    NAMESPACE roc::
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROCHostNumericsConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ROCHostNumericsConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ROCHostNumerics"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ROCHostNumericsConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ROCHostNumericsConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ROCHostNumericsConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ROCHostNumerics"
)
install(FILES LICENSE.md
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/ROCHostNumerics"
)
