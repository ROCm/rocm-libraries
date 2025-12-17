# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(CMakePackageConfigHelpers)

set(HIP_DNN_DATA_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(hipdnn_data_sdk INTERFACE)

set(HIP_DNN_DATA_SDK_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/data_sdk")

target_include_directories(
    hipdnn_data_sdk INTERFACE $<BUILD_INTERFACE:${HIP_DNN_DATA_SDK_INCLUDE_DIR}>
                              $<INSTALL_INTERFACE:${HIP_DNN_DATA_SDK_INSTALL_INCLUDE_DIR}>
)

# Generate the config file from template
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_data_sdkConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk/hipdnn_data_sdkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk
)

export(
    TARGETS hipdnn_data_sdk
    FILE "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk/hipdnn_data_sdkTargets.cmake"
)

install(TARGETS hipdnn_data_sdk EXPORT hipdnn_data_sdk_targets COMPONENT Development)

install(
    DIRECTORY ${HIP_DNN_DATA_SDK_INCLUDE_DIR}/
    DESTINATION ${HIP_DNN_DATA_SDK_INSTALL_INCLUDE_DIR}
    COMPONENT Development
    FILES_MATCHING
    PATTERN "*.hpp"
)

install(EXPORT hipdnn_data_sdk_targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk
        FILE hipdnn_data_sdkTargets.cmake COMPONENT Development
)

install(
    FILES
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk/hipdnn_data_sdkConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk
    COMPONENT Development
)

add_subdirectory(tests)
