# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(CMakePackageConfigHelpers)

# Version handling
hipdnn_setup_version(hipdnn_test_sdk)

# Set project with version
project(hipdnn_test_sdk VERSION ${HIPDNN_TEST_SDK_VERSION} LANGUAGES CXX)

# Read data_sdk version for minimum requirement
hipdnn_get_component_version(hipdnn_data_sdk HIPDNN_TEST_SDK_DATA_SDK_MIN_VERSION)

set(HIP_DNN_TEST_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(hipdnn_test_sdk INTERFACE)

# Generate version header
hipdnn_generate_version_header(hipdnn_test_sdk)

find_package(hip REQUIRED)
target_link_libraries(hipdnn_test_sdk INTERFACE hipdnn_flatbuffers_sdk hipdnn_frontend hip::host)

set(HIP_DNN_TEST_SDK_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/test_sdk")

target_include_directories(
    hipdnn_test_sdk INTERFACE $<BUILD_INTERFACE:${HIP_DNN_TEST_SDK_INCLUDE_DIR}>
                              $<INSTALL_INTERFACE:${HIP_DNN_TEST_SDK_INSTALL_INCLUDE_DIR}>
)

# Generate the config file from template
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_test_sdkConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk/hipdnn_test_sdkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk
)

export(
    TARGETS hipdnn_test_sdk
    FILE "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk/hipdnn_test_sdkTargets.cmake"
)

install(TARGETS hipdnn_test_sdk EXPORT hipdnn_test_sdk_targets COMPONENT Development)

# cmake-lint: disable=E1122
install(
    DIRECTORY ${HIP_DNN_TEST_SDK_INCLUDE_DIR}/
    DESTINATION ${HIP_DNN_TEST_SDK_INSTALL_INCLUDE_DIR}
    COMPONENT Development
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
    PATTERN "*.h.in" EXCLUDE
)

# Install generated version header
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/hipdnn_test_sdk/version.h"
    DESTINATION ${HIP_DNN_TEST_SDK_INSTALL_INCLUDE_DIR}/hipdnn_test_sdk
    COMPONENT Development
)

install(EXPORT hipdnn_test_sdk_targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk
        FILE hipdnn_test_sdkTargets.cmake COMPONENT Development
)

# Generate package version file for find_package version checking
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk/hipdnn_test_sdkConfigVersion.cmake"
    VERSION ${HIPDNN_TEST_SDK_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk/hipdnn_test_sdkConfig.cmake"
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk/hipdnn_test_sdkConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk
    COMPONENT Development
)

add_subdirectory(tests)
