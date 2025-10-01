# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(HIP_DNN_SDK_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/sdk")
set(HIP_DNN_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

hipdnn_add_dependency(flatbuffers VERSION ${HIPDNN_FLATBUFFERS_VERSION})
hipdnn_add_dependency(spdlog VERSION ${HIPDNN_SPDLOG_VERSION})
hipdnn_add_dependency(nlohmann_json VERSION ${HIPDNN_NLOHMANN_JSON_VERSION})

# For dependencies we are going to check if they weren't imported, and then install it along with the SDK headers
# to allow it to work for local builds that want to install. If the dependecy is imported, we will instead use find_package
# and not install the headers.
set(_hipdnn_sdk_local_export_targets)
set(_hipdnn_sdk_local_export_includes)
if(TARGET FlatBuffers)
    # Add alis if it doesn't already exist
    if(NOT TARGET flatbuffers::flatbuffers)
        add_library(flatbuffers::flatbuffers ALIAS FlatBuffers)
    endif()

    get_target_property(_is_imported FlatBuffers IMPORTED)
    if(NOT _is_imported)
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
        list(APPEND _hipdnn_sdk_local_export_targets FlatBuffers)
        # Flatbuffer header structure is different, we need to add the 'flatbuffers' subdir
        list(APPEND _hipdnn_sdk_local_export_includes ${HIP_DNN_FLATBUFFERS_INCLUDE_DIR}/flatbuffers)
    endif()
endif()

if(TARGET spdlog_header_only)
    # Add alis if it doesn't already exist
    if(NOT TARGET spdlog::spdlog_header_only)
        add_library(spdlog::spdlog_header_only ALIAS spdlog_header_only)
    endif()

    get_target_property(_is_imported spdlog_header_only IMPORTED)
    if(NOT _is_imported)
        list(APPEND _hipdnn_sdk_local_export_targets spdlog_header_only)
        # spdlog header structure is different, we need to add the 'spdlog' subdir
        list(APPEND _hipdnn_sdk_local_export_includes ${HIP_DNN_SPDLOG_INCLUDE_DIR}/spdlog)
    endif()
endif()

if(TARGET nlohmann_json)
    # Add alis if it doesn't already exist
    if(TARGET nlohmann_json AND NOT TARGET nlohmann_json::nlohmann_json)
        add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
    endif()

    get_target_property(_is_imported nlohmann_json IMPORTED)
    if(NOT _is_imported)
        list(APPEND _hipdnn_sdk_local_export_targets nlohmann_json)
        list(APPEND _hipdnn_sdk_local_export_includes ${HIP_DNN_NLOHMANN_JSON_INCLUDE_DIR})
    endif()
endif()

add_library(hipdnn_sdk INTERFACE)
target_link_libraries(hipdnn_sdk INTERFACE flatbuffers::flatbuffers spdlog::spdlog_header_only nlohmann_json::nlohmann_json)

target_include_directories(hipdnn_sdk INTERFACE
    $<BUILD_INTERFACE:${HIP_DNN_SDK_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${HIP_DNN_SDK_INSTALL_INCLUDE_DIR}>
)

# Automatically generate the flatbuffer headers from the schema files during build
if(HIP_DNN_GENERATE_SDK_HEADERS)
    set(SCHEMA_FILES
        schemas/batchnorm_attributes.fbs
        schemas/batchnorm_backward_attributes.fbs
        schemas/batchnorm_inference_attributes.fbs
        schemas/convolution_common.fbs
        schemas/convolution_fwd_attributes.fbs
        schemas/convolution_bwd_attributes.fbs
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
    
    add_dependencies(hipdnn_sdk generate_hipdnn_sdk_headers)
endif()

# This is required to use hip/hip_fp16.h and hip/hip_bfloat16.h in any code that includes the sdk
# By doing this, we don't need to link to hip::device
target_compile_definitions(hipdnn_sdk INTERFACE __HIPCC__)

export(
    TARGETS hipdnn_sdk
    ${_hipdnn_sdk_local_export_targets}
    FILE "${CMAKE_BINARY_DIR}/lib/cmake/hipdnn_sdk/hipdnn_sdkTargets.cmake"
)

install(
    TARGETS hipdnn_sdk
    ${_hipdnn_sdk_local_export_targets}
    EXPORT hipdnn_sdk_targets
)

install(
    DIRECTORY ${HIP_DNN_SDK_INCLUDE_DIR}/
    DESTINATION ${HIP_DNN_SDK_INSTALL_INCLUDE_DIR}
)

install(EXPORT hipdnn_sdk_targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_sdk
    FILE hipdnn_sdkTargets.cmake
)

# Install behavior changes depending on if we have local dependencies to export or not.
# If we do, then we need to install the headers for those dependencies and generate a config
# file that uses those headers. If we don't have any local dependencies to export, then we
# generate a config file that uses find_package to find the dependencies.
if(_hipdnn_sdk_local_export_includes)
    install(
        DIRECTORY ${_hipdnn_sdk_local_export_includes}/
        DESTINATION ${HIP_DNN_SDK_INSTALL_INCLUDE_DIR}
    )
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_sdkConfig_local.cmake.in"
        "${CMAKE_BINARY_DIR}/lib/cmake/hipdnn_sdk/hipdnn_sdkConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_sdk
        PATH_VARS HIPDNN_PLUGIN_ENGINE_SUBDIR HIPDNN_INSTALL_PLUGIN_ENGINE_DIR
    )
else()
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_sdkConfig_imported.cmake.in"
        "${CMAKE_BINARY_DIR}/lib/cmake/hipdnn_sdk/hipdnn_sdkConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_sdk
        PATH_VARS HIPDNN_PLUGIN_ENGINE_SUBDIR HIPDNN_INSTALL_PLUGIN_ENGINE_DIR
    )
endif()

install(
    FILES "${CMAKE_BINARY_DIR}/lib/cmake/hipdnn_sdk/hipdnn_sdkConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_sdk
)

add_subdirectory(tests)
