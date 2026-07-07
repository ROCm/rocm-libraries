# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_library(
    rocke_client_impl OBJECT
    RockeClientContainer.cpp
    RockeClientHandle.cpp
    PluginPublic.cpp
    engines/RockeClientEngine.cpp
)

target_include_directories(rocke_client_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
dnn_provider_generate_version_header(rocke_client rocke_client_impl ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(rocke_client_impl PUBLIC hip::host hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk)

target_compile_options(rocke_client_impl PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(rocke_client_impl PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)
clang_tidy_check(rocke_client_impl)

add_library(rocke_client_private STATIC $<TARGET_OBJECTS:rocke_client_impl>)
target_include_directories(rocke_client_private PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/include
)
target_link_libraries(rocke_client_private PUBLIC hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk rocke_client_impl)

add_library(rocke_client SHARED $<TARGET_OBJECTS:rocke_client_impl>)
target_link_libraries(rocke_client PRIVATE hip::host hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk)

if(NOT WIN32)
    target_link_libraries(rocke_client PRIVATE "-Xlinker --exclude-libs=ALL")
endif()

target_compile_options(rocke_client PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(
    rocke_client
    PROPERTIES OUTPUT_NAME "rocke-client"
               CXX_VISIBILITY_PRESET hidden
               LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               INSTALL_RPATH "${ROCM_PATH}/lib"
               INSTALL_RPATH_USE_LINK_PATH TRUE
               BUILD_WITH_INSTALL_RPATH TRUE
)

set_target_properties(rocke_client PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN "lib/hipdnn_plugins/engines")

if(ROCKE_CLIENT_ENABLE_TESTS)
    add_subdirectory(tests)
    add_subdirectory(integration_tests)
endif()
