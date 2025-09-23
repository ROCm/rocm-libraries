# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(GNUInstallDirs)
SET(HIP_DNN_SDK_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/sdk")

hipdnn_add_dependency(flatbuffers VERSION 23.1.21)

# Hack to workaround an issue with how flatbuffers sets up its INSTALL_INTERFACE.
# internally it uses the following: $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include>
# This end ups resolving to /opt/rocm/include/include which isnt correct.  I cant find a way
# to set this value when adding the dependency so after we init the dependency, I then force the
# INTERFACE_INCLUDE_DIRECTORIES to be the correct value required for the hipdnn_sdk
get_target_property(current_includes FlatBuffers INTERFACE_INCLUDE_DIRECTORIES)
list(REMOVE_ITEM current_includes "$<INSTALL_INTERFACE:include/include>")
list(APPEND current_includes "$<INSTALL_INTERFACE:${HIP_DNN_SDK_INSTALL_INCLUDE_DIR}>")
set_target_properties(FlatBuffers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${current_includes}"
)

set(HIP_DNN_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(SCHEMA_FILES
    schemas/batchnorm_attributes.fbs
    schemas/batchnorm_backward_attributes.fbs
    schemas/batchnorm_inference_attributes.fbs
    schemas/convolution_fwd_attributes.fbs
    schemas/data_types.fbs
    schemas/engine_config.fbs
    schemas/engine_details.fbs
    schemas/graph.fbs
    schemas/pointwise_attributes.fbs
    schemas/tensor_attributes.fbs )

_save_var(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS)

SET(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS "--gen-object-api;--gen-mutable;--gen-compare;--defaults-json;--scoped-enums")
build_flatbuffers(
    "${SCHEMA_FILES}" #flatbuffers_schemas
    "" # schema_include_dirs
    generate_hipdnn_sdk_headers  #custom_target_name
    "" # additional_dependencies
    ${HIP_DNN_SDK_INCLUDE_DIR}/hipdnn_sdk/data_objects  # generated_includes_dir
    ""  # binary_schemas_dir
    "" #copy_text_schemas_dir
)

# Flatc build has some warnings that trigger and print to the console.  Since we dont want these we can
# suppress all warnings by using the -w compile flag.  This command prepends -w to the compile flats for
# the flatc target.
set_target_properties(flatc PROPERTIES COMPILE_FLAGS "-w")


_restore_var(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS)

hipdnn_add_dependency(spdlog VERSION 1.15.2)

hipdnn_add_dependency(nlohmann_json VERSION 3.12.0)

add_library(hipdnn_sdk INTERFACE)
target_link_libraries(hipdnn_sdk INTERFACE FlatBuffers spdlog_header_only nlohmann_json)
add_dependencies(hipdnn_sdk generate_hipdnn_sdk_headers)

target_include_directories(hipdnn_sdk INTERFACE
    $<BUILD_INTERFACE:${HIP_DNN_SDK_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${HIP_DNN_SDK_INSTALL_INCLUDE_DIR}>
)

# This is required to use hip/hip_fp16.h and hip/hip_bfloat16.h in any code that includes the sdk
# By doing this, we don't need to link to hip::device
target_compile_definitions(hipdnn_sdk INTERFACE __HIPCC__)

# Export plugin directory variables for external plugin developers
set(HIPDNN_PLUGIN_ENGINE_SUBDIR "${HIPDNN_PLUGIN_ENGINE_SUBDIR}")
set(HIPDNN_INSTALL_PLUGIN_ENGINE_DIR "${HIPDNN_INSTALL_PLUGIN_ENGINE_DIR}")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_sdk_plugin_config.cmake.in"
    "${CMAKE_BINARY_DIR}/lib/cmake/hipdnn_sdk/hipdnn_sdk_plugin_config.cmake"
    @ONLY
)

install(
    TARGETS hipdnn_sdk FlatBuffers spdlog_header_only nlohmann_json
    EXPORT hipdnn_sdk_targets
)

export(
    TARGETS hipdnn_sdk FlatBuffers spdlog_header_only nlohmann_json
    FILE  ${CMAKE_BINARY_DIR}/lib/cmake/hipdnn_sdk/hipdnn_sdkConfig.cmake
)

install(
    DIRECTORY ${HIP_DNN_SDK_INCLUDE_DIR}/ ${HIP_DNN_FLATBUFFERS_INCLUDE_DIR}/ ${HIP_DNN_SPDLOG_INCLUDE_DIR}/ ${HIP_DNN_NLOHMANN_JSON_INCLUDE_DIR}/
    DESTINATION ${HIP_DNN_SDK_INSTALL_INCLUDE_DIR}
)

install(EXPORT hipdnn_sdk_targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_sdk
    FILE hipdnn_sdkConfig.cmake
)

# Install the plugin config file
install(FILES
    "${CMAKE_BINARY_DIR}/lib/cmake/hipdnn_sdk/hipdnn_sdk_plugin_config.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_sdk
)

add_subdirectory(tests)
