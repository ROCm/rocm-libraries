# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(CMakePackageConfigHelpers)

# Version handling
hipdnn_setup_version(hipdnn_backend)

# Set project with version
project(hipdnn_backend VERSION ${HIPDNN_BACKEND_VERSION} LANGUAGES CXX)

find_package(hip REQUIRED)

set(HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR "${PROJECT_BINARY_DIR}/backend/include/")
set(HIP_DNN_BACKEND_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/backend")
set(HIP_DNN_BACKEND_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_library(
    hipdnn_backend_private
    descriptors/BatchnormBackwardOperationDescriptor.cpp
    descriptors/BatchnormInferenceOperationDescriptor.cpp
    descriptors/BatchnormOperationDescriptor.cpp
    descriptors/BatchnormInferenceVarianceExtOperationDescriptor.cpp
    descriptors/BackendDescriptor.cpp
    descriptors/BlockScaleDequantizeOperationDescriptor.cpp
    descriptors/BlockScaleQuantizeOperationDescriptor.cpp
    descriptors/CustomOpOperationDescriptor.cpp
    descriptors/ConvolutionBwdOperationDescriptor.cpp
    descriptors/ConvolutionFwdOperationDescriptor.cpp
    descriptors/ConvolutionWrwOperationDescriptor.cpp
    descriptors/DataTypeConversion.cpp
    descriptors/DescriptorAttributeUtils.cpp
    descriptors/DescriptorFactory.cpp
    descriptors/EngineConfigDescriptor.cpp
    descriptors/EngineDescriptor.cpp
    descriptors/EngineHeuristicDescriptor.cpp
    descriptors/ExecutionPlanDescriptor.cpp
    descriptors/GraphDescriptor.cpp
    descriptors/KnobDescriptor.cpp
    descriptors/KnobSettingDescriptor.cpp
    descriptors/LayernormOperationDescriptor.cpp
    descriptors/MatmulOperationDescriptor.cpp
    descriptors/PointwiseOperationDescriptor.cpp
    descriptors/ResampleFwdOperationDescriptor.cpp
    descriptors/ReductionOperationDescriptor.cpp
    descriptors/RMSNormOperationDescriptor.cpp
    descriptors/SdpaBwdOperationDescriptor.cpp
    descriptors/SdpaFwdOperationDescriptor.cpp
    descriptors/TensorDescriptor.cpp
    descriptors/NodeFactory.cpp
    descriptors/VariantDescriptor.cpp
    handle/HandleFactory.cpp
    handle/Handle.cpp
    logging/GraphLogger.cpp
    logging/Logging.cpp
    FlatbufferUtilities.cpp
    plugin/EnginePlugin.cpp
    plugin/EnginePluginResourceManager.cpp
    plugin/HeuristicPlugin.cpp
    plugin/HeuristicPluginResourceManager.cpp
    plugin/PluginCore.cpp
    plugin/SharedLibrary.cpp
    utilities/EngineOrdering.cpp
    LastErrorManager.cpp
    $<$<BOOL:${WIN32}>:PlatformUtils.windows.cpp>
    $<$<BOOL:${UNIX}>:PlatformUtils.linux.cpp>
)

target_include_directories(
    hipdnn_backend_private PUBLIC ${HIP_DNN_BACKEND_INCLUDE_DIR} ${HIP_DNN_BACKEND_SRC_DIR}
                                  ${HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/include/
)

target_compile_options(hipdnn_backend_private PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS} ${HIPDNN_NO_RTTI_OPTIONS})
target_link_libraries(hipdnn_backend_private PUBLIC hip::host hipdnn_data_sdk hipdnn_plugin_sdk hipdnn_flatbuffers_sdk)
# Backend implementation files use nlohmann::json unconditionally (graph (de)serialization,
# JSON logging). Link as PRIVATE so the dep stays inside the .so and isn't propagated to consumers.
if(NOT TARGET nlohmann_json::nlohmann_json)
    hipdnn_add_dependency(nlohmann_json VERSION ${HIPDNN_NLOHMANN_JSON_VERSION})
endif()
hipdnn_add_dependency_includes(hipdnn_backend_private nlohmann_json::nlohmann_json)
target_link_libraries(hipdnn_backend_private PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(hipdnn_backend_private PRIVATE HIPDNN_BACKEND_COMPILATION)

# Enable spdlog support via shared cmake function
hipdnn_add_dependency(spdlog VERSION ${HIPDNN_SPDLOG_VERSION})
hipdnn_enable_spdlog(hipdnn_backend_private)

clang_tidy_check(hipdnn_backend_private)

add_library(hipdnn_backend SHARED HipdnnBackend.cpp)

hipdnn_generate_version_header(hipdnn_backend)

target_include_directories(
    hipdnn_backend
    PUBLIC $<BUILD_INTERFACE:${HIP_DNN_BACKEND_INCLUDE_DIR}>
           $<INSTALL_INTERFACE:${HIP_DNN_BACKEND_INSTALL_INCLUDE_DIR}>
           $<BUILD_INTERFACE:${HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR}>
)

# Backend headers include HIP headers, so both backend and consumers need this definition
# Use PUBLIC to ensure it's exported to consumers
target_compile_definitions(hipdnn_backend PUBLIC __HIP_PLATFORM_AMD__)

target_compile_options(hipdnn_backend PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS} ${HIPDNN_NO_RTTI_OPTIONS})
# This definition is used to enable logging in the backend.  It is #ifdef'ed in the backend code.
# This is to stop the frontend/plugin from having lazy log initialization which will end up with the
# same logger name. Frontend/plugin code will need to init the logger themselves so they have a
# different name.
target_compile_definitions(hipdnn_backend PRIVATE HIPDNN_BACKEND_COMPILATION)
target_link_libraries(hipdnn_backend PRIVATE hipdnn_backend_private)
# Export dependencies for consumers because backend public API needs HIP runtime dep
target_link_libraries(hipdnn_backend INTERFACE hip::host)
# HIDE symbols from linked static libs
if(NOT WIN32)
    target_link_libraries(hipdnn_backend PRIVATE "-Xlinker --exclude-libs=ALL")
endif()
set_target_properties(hipdnn_backend PROPERTIES CXX_VISIBILITY_PRESET hidden)

clang_tidy_check(hipdnn_backend)

include(GenerateExportHeader)
generate_export_header(
    hipdnn_backend EXPORT_FILE_NAME ${HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR}/hipdnn_backend_export.h
)

install(
    TARGETS hipdnn_backend
    EXPORT hipdnn_backend_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Runtime NAMELINK_COMPONENT Development
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Runtime
)

export(
    TARGETS hipdnn_backend
    FILE ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_backend/hipdnn_backendTargets.cmake
)

install(
    DIRECTORY ${HIP_DNN_BACKEND_INCLUDE_DIR}/
    DESTINATION ${HIP_DNN_BACKEND_INSTALL_INCLUDE_DIR}
    COMPONENT Development
    FILES_MATCHING
    PATTERN "*.h"
)

install(FILES ${HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR}/hipdnn_backend_export.h
        DESTINATION ${HIP_DNN_BACKEND_INSTALL_INCLUDE_DIR} COMPONENT Development
)

install(EXPORT hipdnn_backend_targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_backend
        FILE hipdnn_backendTargets.cmake COMPONENT Development
)

# Generate the Config.cmake file from template
configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/../cmake/hipdnn_backendConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_backend/hipdnn_backendConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_backend
)

# Generate package version file for find_package version checking
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_backend/hipdnn_backendConfigVersion.cmake"
    VERSION ${HIPDNN_BACKEND_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_backend/hipdnn_backendConfig.cmake"
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_backend/hipdnn_backendConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_backend COMPONENT Development
)
