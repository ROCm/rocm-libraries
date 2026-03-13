# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(CMakePackageConfigHelpers)

# Version handling
hipdnn_setup_version(hipdnn_data_sdk)

# Set project with version
project(hipdnn_data_sdk VERSION ${HIPDNN_DATA_SDK_VERSION} LANGUAGES CXX)

set(HIP_DNN_DATA_SDK_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/data_sdk")
set(HIP_DNN_DATA_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

hipdnn_add_dependency(flatbuffers VERSION ${HIPDNN_FLATBUFFERS_VERSION})

if(NOT HIPDNN_FRONTEND_SKIP_JSON_LIB)
    hipdnn_add_dependency(nlohmann_json VERSION ${HIPDNN_NLOHMANN_JSON_VERSION})
endif()

# Supported FlatBuffers versions (oldest to newest) - update this list when adding new version support.
# Each version must have pre-generated headers in include/hipdnn_data_sdk/data_objects/v<version_tag>/
# 24.12.23 matches PyTorch's FlatBuffers version for compatibility
set(HIPDNN_SUPPORTED_FLATBUFFERS_VERSIONS "24.12.23" "${HIPDNN_FLATBUFFERS_VERSION}")

if(NOT HIPDNN_FLATBUFFERS_VERSION IN_LIST HIPDNN_SUPPORTED_FLATBUFFERS_VERSIONS)
    message(FATAL_ERROR
        "Unsupported FlatBuffers version: ${HIPDNN_FLATBUFFERS_VERSION}. "
        "Supported versions are: ${HIPDNN_SUPPORTED_FLATBUFFERS_VERSIONS}"
    )
endif()

# Compute versioned directory name from version string (e.g., "25.9.23" -> "v25_9_23")
string(REPLACE "." "_" _fb_version_tag "${HIPDNN_FLATBUFFERS_VERSION}")
set(HIPDNN_FLATBUFFERS_VERSION_DIR "v${_fb_version_tag}")
set(HIP_DNN_DATA_SDK_GENERATED_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/include/hipdnn_data_sdk/data_objects/${HIPDNN_FLATBUFFERS_VERSION_DIR}"
)

# For dependencies we are going to check if they weren't imported, and then install it along with
# the SDK headers to allow it to work for local builds that want to install. If the dependecy is
# imported, we will instead use find_package and not install the headers.
set(_hipdnn_data_sdk_local_export_targets)
set(_hipdnn_data_sdk_local_export_includes)

# Function to process data SDK dependencies and set up aliases and export lists Arguments:
# TARGET_NAME - The actual target name (e.g., FlatBuffers) ALIAS_NAME - The alias target name (e.g.,
# flatbuffers::flatbuffers) INCLUDE_SUBDIR - Optional subdirectory to append to include path (e.g.,
# flatbuffers)
function(_hipdnn_data_sdk_process_dependency TARGET_NAME ALIAS_NAME)
    # Parse optional arguments
    set(options "")
    set(oneValueArgs INCLUDE_SUBDIR INCLUDE_DIR)
    set(multiValueArgs "")
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT TARGET ${TARGET_NAME})
        return()
    endif()

    # Add alias if it doesn't already exist
    if(NOT TARGET ${ALIAS_NAME})
        add_library(${ALIAS_NAME} ALIAS ${TARGET_NAME})
    endif()

    # Check if target is imported (system-installed) or locally built
    get_target_property(_is_imported ${TARGET_NAME} IMPORTED)
    if(NOT _is_imported)
        # Add to local export targets
        list(APPEND _hipdnn_data_sdk_local_export_targets ${TARGET_NAME})
        set(_hipdnn_data_sdk_local_export_targets ${_hipdnn_data_sdk_local_export_targets} PARENT_SCOPE)

        # Add include directory if specified
        # When INCLUDE_SUBDIR is specified, we want to install the subdirectory WITH its name preserved
        # (e.g., .../include/flatbuffers/ should install as flatbuffers/ not flatten its contents)
        # So we use INCLUDE_DIR (the parent) in the list, and the install command will copy subdirs
        if(ARG_INCLUDE_DIR)
            list(APPEND _hipdnn_data_sdk_local_export_includes ${ARG_INCLUDE_DIR})
            set(_hipdnn_data_sdk_local_export_includes ${_hipdnn_data_sdk_local_export_includes} PARENT_SCOPE)
        endif()
    endif()
endfunction()

# Process flatbuffers dependency
if(TARGET FlatBuffers)
    # Hack to workaround an issue with how flatbuffers sets up its INSTALL_INTERFACE. internally it
    # uses the following: $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include> This end ups
    # resolving to /opt/rocm/include/include which isnt correct.  I cant find a way to set this
    # value when adding the dependency so after we init the dependency, I then force the
    # INTERFACE_INCLUDE_DIRECTORIES to be the correct value required for the hipdnn_data_sdk
    get_target_property(_is_imported FlatBuffers IMPORTED)
    if(NOT _is_imported)
        get_target_property(current_includes FlatBuffers INTERFACE_INCLUDE_DIRECTORIES)
        list(REMOVE_ITEM current_includes "$<INSTALL_INTERFACE:include/include>")
        list(APPEND current_includes "$<INSTALL_INTERFACE:${HIP_DNN_DATA_SDK_INSTALL_INCLUDE_DIR}>")
        set_target_properties(
            FlatBuffers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${current_includes}"
        )
    endif()
    _hipdnn_data_sdk_process_dependency(
        FlatBuffers flatbuffers::flatbuffers INCLUDE_DIR ${HIP_DNN_FLATBUFFERS_INCLUDE_DIR}
        INCLUDE_SUBDIR flatbuffers
    )
endif()

# Process nlohmann_json dependency
if(NOT HIPDNN_FRONTEND_SKIP_JSON_LIB)
    _hipdnn_data_sdk_process_dependency(
        nlohmann_json nlohmann_json::nlohmann_json INCLUDE_DIR ${HIP_DNN_NLOHMANN_JSON_INCLUDE_DIR}
    )
endif()

add_library(hipdnn_data_sdk INTERFACE)

# Generate version header
hipdnn_generate_version_header(hipdnn_data_sdk)

# Add includes from dependency targets
hipdnn_add_dependency_includes(hipdnn_data_sdk flatbuffers::flatbuffers)
if(NOT HIPDNN_FRONTEND_SKIP_JSON_LIB)
    hipdnn_add_dependency_includes(hipdnn_data_sdk nlohmann_json::nlohmann_json)
endif()

target_compile_definitions(
    hipdnn_data_sdk INTERFACE
    $<$<BOOL:${HIPDNN_FRONTEND_SKIP_JSON_LIB}>:HIPDNN_DATA_SDK_SKIP_JSON_LIB>
)

target_include_directories(
    hipdnn_data_sdk INTERFACE $<BUILD_INTERFACE:${HIP_DNN_DATA_SDK_INCLUDE_DIR}>
                              $<BUILD_INTERFACE:${HIP_DNN_DATA_SDK_GENERATED_INCLUDE_DIR}>
                              $<INSTALL_INTERFACE:${HIP_DNN_DATA_SDK_INSTALL_INCLUDE_DIR}>
)

# Automatically generate the flatbuffer headers from the schema files during build.
# When enabled, headers are regenerated for ALL supported FlatBuffers versions so that
# schema changes are reflected across every version directory.
if(HIPDNN_GENERATE_SDK_HEADERS)
    include(../cmake/FlatBuffersGenerate.cmake)

    hipdnn_generate_flatbuffer_headers(
        TARGET hipdnn_data_sdk
        SCHEMAS
            schemas/batchnorm_attributes.fbs
            schemas/batchnorm_backward_attributes.fbs
            schemas/batchnorm_inference_attributes.fbs
            schemas/batchnorm_inference_attributes_variance_ext.fbs
            schemas/block_scale_dequantize_attributes.fbs
            schemas/convolution_common.fbs
            schemas/convolution_fwd_attributes.fbs
            schemas/convolution_bwd_attributes.fbs
            schemas/convolution_wrw_attributes.fbs
            schemas/custom_op_attributes.fbs
            schemas/data_types.fbs
            schemas/engine_config.fbs
            schemas/engine_details.fbs
            schemas/graph.fbs
            schemas/knob_value.fbs
            schemas/layernorm_attributes.fbs
            schemas/matmul_attributes.fbs
            schemas/norm_common.fbs
            schemas/pointwise_attributes.fbs
            schemas/rmsnorm_attributes.fbs
            schemas/block_scale_quantize_attributes.fbs
            schemas/sdpa_attributes.fbs
            schemas/tensor_attributes.fbs
        SCHEMAS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/schemas"
        PRIMARY_VERSION ${HIPDNN_FLATBUFFERS_VERSION}
        SUPPORTED_VERSIONS ${HIPDNN_SUPPORTED_FLATBUFFERS_VERSIONS}
        GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/hipdnn_data_sdk/data_objects"
    )
endif()

export(TARGETS hipdnn_data_sdk ${_hipdnn_data_sdk_local_export_targets}
       FILE "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk/hipdnn_data_sdkTargets.cmake"
)

install(TARGETS hipdnn_data_sdk ${_hipdnn_data_sdk_local_export_targets} EXPORT hipdnn_data_sdk_targets
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

# Install behavior changes depending on if we have local dependencies to export or not. If we do,
# then we need to install the headers for those dependencies and generate a config file that uses
# those headers. If we don't have any local dependencies to export, then we generate a config file
# that uses find_package to find the dependencies.
if(_hipdnn_data_sdk_local_export_includes)
    # cmake-lint: disable=E1122
    # Install FlatBuffers headers to data_sdk/flatbuffers/ (not hipdnn_data_sdk/flatbuffers/)
    # so they're findable from the base include path without adding extra include directories
    # Exclude code generation utilities (codegen/, pch/) that aren't needed at runtime
    install(
        DIRECTORY ${_hipdnn_data_sdk_local_export_includes}/
        DESTINATION ${HIP_DNN_DATA_SDK_INSTALL_INCLUDE_DIR}
        COMPONENT Development
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "codegen" EXCLUDE
    )
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_data_sdkConfig_local.cmake.in"
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk/hipdnn_data_sdkConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk
        PATH_VARS HIPDNN_PLUGIN_ENGINE_SUBDIR HIPDNN_FULL_INSTALL_PLUGIN_ENGINE_DIR
                  HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR
    )
else()
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_data_sdkConfig_imported.cmake.in"
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk/hipdnn_data_sdkConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_data_sdk
        PATH_VARS HIPDNN_PLUGIN_ENGINE_SUBDIR HIPDNN_FULL_INSTALL_PLUGIN_ENGINE_DIR
                  HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR
    )
endif()

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
