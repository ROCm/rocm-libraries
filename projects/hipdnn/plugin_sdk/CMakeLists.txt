# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(HIP_DNN_PLUGIN_SDK_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/plugin_sdk")
set(HIP_DNN_PLUGIN_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Create the header-only interface library
add_library(hipdnn_plugin_sdk INTERFACE)

# Link to hipdnn_sdk for logging and utilities
target_link_libraries(hipdnn_plugin_sdk INTERFACE hipdnn_sdk)

# Set up include directories
target_include_directories(
    hipdnn_plugin_sdk INTERFACE $<BUILD_INTERFACE:${HIP_DNN_PLUGIN_SDK_INCLUDE_DIR}>
                                $<INSTALL_INTERFACE:${HIP_DNN_PLUGIN_SDK_INSTALL_INCLUDE_DIR}>
)

# Export targets for build tree
export(
    TARGETS hipdnn_plugin_sdk
    FILE "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk/hipdnn_plugin_sdkTargets.cmake"
)

# Install the library target
install(TARGETS hipdnn_plugin_sdk EXPORT hipdnn_plugin_sdk_targets COMPONENT Development)

# cmake-lint: disable=E1122
install(
    DIRECTORY ${HIP_DNN_PLUGIN_SDK_INCLUDE_DIR}/
    DESTINATION ${HIP_DNN_PLUGIN_SDK_INSTALL_INCLUDE_DIR}
    COMPONENT Development
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
)

# Install export targets
install(EXPORT hipdnn_plugin_sdk_targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk
        FILE hipdnn_plugin_sdkTargets.cmake COMPONENT Development
)

# Configure package config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_plugin_sdkConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk/hipdnn_plugin_sdkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk
)

# Install package config file
install(
    FILES
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk/hipdnn_plugin_sdkConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk
    COMPONENT Development
)

# Add tests subdirectory
add_subdirectory(tests)
