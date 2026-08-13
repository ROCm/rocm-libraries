# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Temporary compatibility project for TheRock revisions that still configure
# shared/mxdatagenerator directly. It owns no numerical implementation or
# headers. Delete this directory after TheRock removes its mxDataGenerator
# support subproject and rocm-libraries advances to that revision.

cmake_minimum_required(VERSION 3.25.2)
project(mxDataGenerator VERSION 0.1.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

message(STATUS
    "The mxDataGenerator component has been removed. "
    "This empty package exists temporarily for pinned TheRock revisions."
)

add_library(mxDataGenerator INTERFACE)
add_library(roc::mxDataGenerator ALIAS mxDataGenerator)
set_target_properties(
    mxDataGenerator
    PROPERTIES
        DEPRECATION
            "roc::mxDataGenerator is a temporary empty compatibility target; use ROCHostValidation APIs instead."
)

set(mxdatagenerator_install_cmakedir
    "${CMAKE_INSTALL_LIBDIR}/cmake/mxDataGenerator"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/mxDataGeneratorConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mxDataGeneratorConfig.cmake"
    INSTALL_DESTINATION "${mxdatagenerator_install_cmakedir}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/mxDataGeneratorConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(
    TARGETS mxDataGenerator
    EXPORT mxDataGeneratorTargets
    COMPONENT devel
)
install(
    EXPORT mxDataGeneratorTargets
    FILE mxDataGeneratorTargets.cmake
    DESTINATION "${mxdatagenerator_install_cmakedir}"
    NAMESPACE roc::
    COMPONENT devel
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/mxDataGeneratorConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/mxDataGeneratorConfigVersion.cmake"
    DESTINATION "${mxdatagenerator_install_cmakedir}"
    COMPONENT devel
)
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/mxDataGenerator"
    COMPONENT docs
)
