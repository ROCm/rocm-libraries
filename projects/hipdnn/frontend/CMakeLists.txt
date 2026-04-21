# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(CMakePackageConfigHelpers)

# Version handling
hipdnn_setup_version(hipdnn_frontend)

# Set project with version
project(hipdnn_frontend VERSION ${HIPDNN_FRONTEND_VERSION} LANGUAGES CXX)

# Read data_sdk version for minimum requirement
hipdnn_get_component_version(hipdnn_data_sdk HIPDNN_FRONTEND_DATA_SDK_MIN_VERSION)

# Read backend version for minimum requirement
hipdnn_get_component_version(hipdnn_backend HIPDNN_FRONTEND_BACKEND_MIN_VERSION)

set(HIP_DNN_FRONTEND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(hipdnn_frontend INTERFACE)

# Generate version header
hipdnn_generate_version_header(hipdnn_frontend)

target_link_libraries(hipdnn_frontend INTERFACE hipdnn_data_sdk hipdnn_backend)

set(HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/frontend")

target_include_directories(
    hipdnn_frontend INTERFACE $<BUILD_INTERFACE:${HIP_DNN_FRONTEND_INCLUDE_DIR}>
                              $<INSTALL_INTERFACE:${HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR}>
)

# Export installed hipdnn_frontend include paths as system include paths.
set_target_properties(hipdnn_frontend PROPERTIES
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
        "$<INSTALL_INTERFACE:${HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR}>"
)

target_compile_definitions(
    hipdnn_frontend INTERFACE
    $<$<BOOL:${HIPDNN_FRONTEND_SKIP_JSON_LIB}>:HIPDNN_FRONTEND_SKIP_JSON_LIB>
    $<$<BOOL:${HIPDNN_ENABLE_SDPA}>:HIPDNN_ENABLE_SDPA>
)

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
    PATTERN "*.h.in" EXCLUDE
)

# Install generated version header
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/hipdnn_frontend/version.h"
    DESTINATION ${HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR}/hipdnn_frontend
    COMPONENT Development
)

install(EXPORT hipdnn_frontend_targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend
        FILE hipdnn_frontendTargets.cmake COMPONENT Development
)

# Generate package version file for find_package version checking
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend/hipdnn_frontendConfigVersion.cmake"
    VERSION ${HIPDNN_FRONTEND_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake"
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend/hipdnn_frontendConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend
    COMPONENT Development
)

add_subdirectory(tests)
