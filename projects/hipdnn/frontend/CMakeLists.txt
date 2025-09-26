# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(HIP_DNN_FRONTEND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(tests)

add_library(hipdnn_frontend INTERFACE)

target_compile_definitions(hipdnn_frontend INTERFACE COMPONENT_NAME="hipdnn_frontend")

target_link_libraries(hipdnn_frontend INTERFACE hipdnn_sdk hipdnn_backend)

SET(HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/frontend")

target_include_directories(hipdnn_frontend INTERFACE
    $<BUILD_INTERFACE:${HIP_DNN_FRONTEND_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR}>
)

install( 
    TARGETS hipdnn_frontend hipdnn_sdk hipdnn_backend FlatBuffers spdlog_header_only
    EXPORT hipdnn-frontend
)

export(
    TARGETS hipdnn_frontend hipdnn_sdk hipdnn_backend FlatBuffers spdlog_header_only
    FILE  ${CMAKE_BINARY_DIR}/lib/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake
)

install(
    DIRECTORY ${HIP_DNN_FRONTEND_INCLUDE_DIR}/
    DESTINATION ${HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR}
)

install(EXPORT hipdnn-frontend
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend
    FILE hipdnn_frontendConfig.cmake
)
