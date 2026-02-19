# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(CMakePackageConfigHelpers)

# Read data_sdk version for minimum requirement
hipdnn_get_data_sdk_version(HIPDNN_FRONTEND_DATA_SDK_MIN_VERSION)

set(HIP_DNN_FRONTEND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(tests)

add_library(hipdnn_frontend INTERFACE)

target_link_libraries(hipdnn_frontend INTERFACE hipdnn_data_sdk hipdnn_backend)

set(HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/frontend")

target_include_directories(
    hipdnn_frontend INTERFACE $<BUILD_INTERFACE:${HIP_DNN_FRONTEND_INCLUDE_DIR}>
                              $<INSTALL_INTERFACE:${HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR}>
)

# Warn if user manually defines HIPDNN_FRONTEND_SKIP_JSON_LIB
if(CMAKE_CXX_FLAGS MATCHES "HIPDNN_FRONTEND_SKIP_JSON_LIB")
    message(WARNING
        "HIPDNN_FRONTEND_SKIP_JSON_LIB should not be manually defined. "
        "Use -DHIPDNN_ENABLE_JSON=OFF instead to disable JSON support.")
endif()

# Automatically define HIPDNN_FRONTEND_SKIP_JSON_LIB when JSON is disabled
if(NOT HIPDNN_ENABLE_JSON)
    target_compile_definitions(hipdnn_frontend INTERFACE HIPDNN_FRONTEND_SKIP_JSON_LIB)
endif()

# Generate the config file from template
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_frontendConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend
)

export(
    TARGETS hipdnn_frontend
    FILE "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend/hipdnn_frontendTargets.cmake"
)

install(TARGETS hipdnn_frontend EXPORT hipdnn_frontend_targets COMPONENT Development)

# cmake-lint: disable=E1122
install(
    DIRECTORY ${HIP_DNN_FRONTEND_INCLUDE_DIR}/
    DESTINATION ${HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR}
    COMPONENT Development
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
)

install(EXPORT hipdnn_frontend_targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend
        FILE hipdnn_frontendTargets.cmake COMPONENT Development
)

install(
    FILES
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend
    COMPONENT Development
)
