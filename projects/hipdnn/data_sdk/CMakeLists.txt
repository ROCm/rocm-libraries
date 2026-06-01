# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(CMakePackageConfigHelpers)

# Version handling
hipdnn_setup_version(hipdnn_data_sdk)

# Set project with version
project(hipdnn_data_sdk VERSION ${HIPDNN_DATA_SDK_VERSION} LANGUAGES CXX)

set(HIP_DNN_DATA_SDK_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/data_sdk")
set(HIP_DNN_DATA_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(hipdnn_data_sdk INTERFACE)

# Generate version header
hipdnn_generate_version_header(hipdnn_data_sdk)

target_include_directories(
    hipdnn_data_sdk INTERFACE $<BUILD_INTERFACE:${HIP_DNN_DATA_SDK_INCLUDE_DIR}>
                              $<INSTALL_INTERFACE:${HIP_DNN_DATA_SDK_INSTALL_INCLUDE_DIR}>
)
target_link_libraries(hipdnn_data_sdk INTERFACE ${CMAKE_DL_LIBS})

export(TARGETS hipdnn_data_sdk
       FILE "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk/hipdnn_data_sdkTargets.cmake"
)

install(TARGETS hipdnn_data_sdk EXPORT hipdnn_data_sdk_targets
        COMPONENT Development
)

# cmake-lint: disable=E1122
install(
    DIRECTORY ${HIP_DNN_DATA_SDK_INCLUDE_DIR}/
    DESTINATION ${HIP_DNN_DATA_SDK_INSTALL_INCLUDE_DIR}
    COMPONENT Development
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.h.in" EXCLUDE
)

# Install generated version header
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/hipdnn_data_sdk/version.h"
    DESTINATION ${HIP_DNN_DATA_SDK_INSTALL_INCLUDE_DIR}/hipdnn_data_sdk
    COMPONENT Development
)

install(EXPORT hipdnn_data_sdk_targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk
        FILE hipdnn_data_sdkTargets.cmake COMPONENT Development
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_data_sdkConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk/hipdnn_data_sdkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk
    PATH_VARS HIPDNN_PLUGIN_ENGINE_SUBDIR HIPDNN_FULL_INSTALL_PLUGIN_ENGINE_DIR
              HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR
              HIPDNN_PLUGIN_HEURISTIC_SUBDIR HIPDNN_FULL_INSTALL_PLUGIN_HEURISTIC_DIR
              HIPDNN_RELATIVE_INSTALL_PLUGIN_HEURISTIC_DIR
)

# Generate package version file for find_package version checking
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk/hipdnn_data_sdkConfigVersion.cmake"
    VERSION ${HIPDNN_DATA_SDK_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk/hipdnn_data_sdkConfig.cmake"
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk/hipdnn_data_sdkConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk
    COMPONENT Development
)

add_subdirectory(tests)
