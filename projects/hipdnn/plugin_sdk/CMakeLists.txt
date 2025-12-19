# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(CMakePackageConfigHelpers)

set(HIP_DNN_PLUGIN_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(hipdnn_plugin_sdk INTERFACE)

target_link_libraries(hipdnn_plugin_sdk INTERFACE hipdnn_sdk)

set(HIP_DNN_PLUGIN_SDK_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/plugin_sdk")

target_include_directories(
    hipdnn_plugin_sdk INTERFACE $<BUILD_INTERFACE:${HIP_DNN_PLUGIN_SDK_INCLUDE_DIR}>
                                $<INSTALL_INTERFACE:${HIP_DNN_PLUGIN_SDK_INSTALL_INCLUDE_DIR}>
)

# Generate the config file from template
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_plugin_sdkConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk/hipdnn_plugin_sdkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk
)

export(
    TARGETS hipdnn_plugin_sdk
    FILE "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk/hipdnn_plugin_sdkTargets.cmake"
)

install(TARGETS hipdnn_plugin_sdk EXPORT hipdnn_plugin_sdk_targets COMPONENT Development)

# cmake-lint: disable=E1122
install(
    DIRECTORY ${HIP_DNN_PLUGIN_SDK_INCLUDE_DIR}/
    DESTINATION ${HIP_DNN_PLUGIN_SDK_INSTALL_INCLUDE_DIR}
    COMPONENT Development
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
)

install(EXPORT hipdnn_plugin_sdk_targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk
        FILE hipdnn_plugin_sdkTargets.cmake COMPONENT Development
)

install(
    FILES
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk/hipdnn_plugin_sdkConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk
    COMPONENT Development
)

add_subdirectory(tests)
