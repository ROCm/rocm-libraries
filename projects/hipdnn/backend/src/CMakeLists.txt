# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(hip REQUIRED)

set(HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR "${PROJECT_BINARY_DIR}/backend/include/")
set(HIP_DNN_BACKEND_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/backend")
set(HIP_DNN_BACKEND_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_library(hipdnn_backend_private
    descriptors/BackendDescriptor.cpp
    descriptors/EngineConfigDescriptor.cpp
    descriptors/EngineDescriptor.cpp
    descriptors/EngineHeuristicDescriptor.cpp
    descriptors/ExecutionPlanDescriptor.cpp
    descriptors/GraphDescriptor.cpp
    descriptors/VariantDescriptor.cpp
    descriptors/DescriptorFactory.cpp
    handle/HandleFactory.cpp
    handle/Handle.cpp
    logging/Logging.cpp
    FlatbufferUtilities.cpp
    plugin/EnginePlugin.cpp
    plugin/EnginePluginResourceManager.cpp
    plugin/PluginCore.cpp
    plugin/SharedLibrary.cpp
    LastErrorManager.cpp
    $<$<BOOL:${WIN32}>:PlatformUtils.windows.cpp>
    $<$<BOOL:${UNIX}>:PlatformUtils.linux.cpp>
)

target_include_directories(hipdnn_backend_private 
    PUBLIC 
        ${HIP_DNN_BACKEND_INCLUDE_DIR}
        ${HIP_DNN_BACKEND_SRC_DIR}
        ${HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR}
)

target_compile_options(hipdnn_backend_private PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(hipdnn_backend_private PUBLIC hip::host hipdnn_sdk)
target_link_libraries(hipdnn_backend_private PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(hipdnn_backend_private PRIVATE HIPDNN_BACKEND_COMPILATION)

clang_tidy_check(hipdnn_backend_private)

add_library(hipdnn_backend SHARED 
    HipdnnBackend.cpp
)

target_include_directories(hipdnn_backend 
    SYSTEM PUBLIC 
        ${HIP_INCLUDE_DIRS}
    PUBLIC 
        $<BUILD_INTERFACE:${HIP_DNN_BACKEND_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${HIP_DNN_BACKEND_INSTALL_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR}>
)

target_compile_options(hipdnn_backend PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
# This definition is used to enable logging in the backend.  It is #ifdef'ed in the backend code.
# This is to stop the frontend/plugin from having lazy log initialization which will end up with the same
# logger name. Frontend/plugin code will need to init the logger themselves so they have a different name.
target_compile_definitions(hipdnn_backend PRIVATE HIPDNN_BACKEND_COMPILATION)
target_link_libraries(hipdnn_backend PRIVATE hipdnn_backend_private)
target_link_libraries(hipdnn_backend PRIVATE "-Xlinker --exclude-libs=ALL") # HIDE symbols from linked static libs
set_target_properties(hipdnn_backend PROPERTIES 
    CXX_VISIBILITY_PRESET hidden
)

clang_tidy_check(hipdnn_backend)

include(GenerateExportHeader)
generate_export_header(hipdnn_backend 
    EXPORT_FILE_NAME 
    ${HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR}/hipdnn_backend_export.h
)

install( 
    TARGETS hipdnn_backend
    EXPORT hipdnn_backend_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Runtime
        NAMELINK_COMPONENT Development
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Development
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Runtime
)

export(
    TARGETS hipdnn_backend
    FILE ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_backend/hipdnn_backendConfig.cmake
)

install(
    DIRECTORY ${HIP_DNN_BACKEND_INCLUDE_DIR}/
    DESTINATION ${HIP_DNN_BACKEND_INSTALL_INCLUDE_DIR}
    COMPONENT Development
    FILES_MATCHING PATTERN "*.h"
)

install(
    FILES ${HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR}/hipdnn_backend_export.h
    DESTINATION ${HIP_DNN_BACKEND_INSTALL_INCLUDE_DIR}
    COMPONENT Development
)

install(
    EXPORT hipdnn_backend_targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_backend
    FILE hipdnn_backendConfig.cmake
    COMPONENT Development
)
