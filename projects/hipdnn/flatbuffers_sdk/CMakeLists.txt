# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(CMakePackageConfigHelpers)

# Version handling
hipdnn_setup_version(hipdnn_flatbuffers_sdk)

# Set project with version
project(hipdnn_flatbuffers_sdk VERSION ${HIPDNN_FLATBUFFERS_SDK_VERSION} LANGUAGES CXX)

# Read data_sdk version for minimum requirement (needed for types.hpp shared type definitions)
hipdnn_get_component_version(hipdnn_data_sdk HIPDNN_FLATBUFFERS_SDK_DATA_SDK_MIN_VERSION)

set(HIP_DNN_FLATBUFFERS_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(HIP_DNN_FLATBUFFERS_SDK_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/flatbuffers_sdk")

# --- FlatBuffers dependency ---
hipdnn_add_dependency(flatbuffers VERSION ${HIPDNN_FLATBUFFERS_VERSION})

if(NOT HIPDNN_FRONTEND_SKIP_JSON_LIB)
    hipdnn_add_dependency(nlohmann_json VERSION ${HIPDNN_NLOHMANN_JSON_VERSION})
endif()

# Supported FlatBuffers versions (oldest to newest)
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
set(HIP_DNN_FLATBUFFERS_SDK_GENERATED_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/include/hipdnn_flatbuffers_sdk/data_objects/${HIPDNN_FLATBUFFERS_VERSION_DIR}"
)

# --- Dependency processing ---
# During the transition period, data_sdk already processes and exports FlatBuffers/nlohmann_json
# dependencies (aliases, include dirs, local export targets). Re-enable this block once data_sdk
# no longer provides these dependencies.
set(_hipdnn_flatbuffers_sdk_local_export_targets)
set(_hipdnn_flatbuffers_sdk_local_export_includes)

# --- Create the library target ---
add_library(hipdnn_flatbuffers_sdk INTERFACE)

# Generate version header
hipdnn_generate_version_header(hipdnn_flatbuffers_sdk)

# Link to data_sdk for shared type definitions (types.hpp) and the
# hipdnn_add_dependency_includes() function provided by data_sdk's CMake config.
target_link_libraries(hipdnn_flatbuffers_sdk INTERFACE hipdnn_data_sdk)

# Add FlatBuffers includes
hipdnn_add_dependency_includes(hipdnn_flatbuffers_sdk flatbuffers::flatbuffers)
if(NOT HIPDNN_FRONTEND_SKIP_JSON_LIB)
    hipdnn_add_dependency_includes(hipdnn_flatbuffers_sdk nlohmann_json::nlohmann_json)
endif()

target_compile_definitions(
    hipdnn_flatbuffers_sdk INTERFACE
    $<$<BOOL:${HIPDNN_FRONTEND_SKIP_JSON_LIB}>:HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB>
)

target_include_directories(
    hipdnn_flatbuffers_sdk INTERFACE $<BUILD_INTERFACE:${HIP_DNN_FLATBUFFERS_SDK_INCLUDE_DIR}>
                                     $<BUILD_INTERFACE:${HIP_DNN_FLATBUFFERS_SDK_GENERATED_INCLUDE_DIR}>
                                     $<INSTALL_INTERFACE:${HIP_DNN_FLATBUFFERS_SDK_INSTALL_INCLUDE_DIR}>
)

# --- Automatic FlatBuffer header generation from schemas ---
if(HIPDNN_GENERATE_SDK_HEADERS)
    include(../cmake/FlatBuffersGenerate.cmake)

    hipdnn_generate_flatbuffer_headers(
        TARGET hipdnn_flatbuffers_sdk
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
            schemas/device_properties.fbs
            schemas/engine_config.fbs
            schemas/engine_details.fbs
            schemas/graph.fbs
            schemas/knob_value.fbs
            schemas/layernorm_attributes.fbs
            schemas/matmul_attributes.fbs
            schemas/norm_common.fbs
            schemas/pointwise_attributes.fbs
            schemas/rmsnorm_attributes.fbs
            schemas/rmsnorm_backward_attributes.fbs
            schemas/block_scale_quantize_attributes.fbs
            schemas/sdpa_attributes.fbs
            schemas/sdpa_backward_attributes.fbs
            schemas/reduction_attributes.fbs
            schemas/tensor_attributes.fbs
        SCHEMAS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/schemas"
        PRIMARY_VERSION ${HIPDNN_FLATBUFFERS_VERSION}
        SUPPORTED_VERSIONS ${HIPDNN_SUPPORTED_FLATBUFFERS_VERSIONS}
        GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/hipdnn_flatbuffers_sdk/data_objects"
        OUTPUT_NAMESPACE hipdnn_flatbuffers_sdk
    )
endif()

# --- Export and install ---
export(TARGETS hipdnn_flatbuffers_sdk ${_hipdnn_flatbuffers_sdk_local_export_targets}
       FILE "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_flatbuffers_sdk/hipdnn_flatbuffers_sdkTargets.cmake"
)

install(TARGETS hipdnn_flatbuffers_sdk ${_hipdnn_flatbuffers_sdk_local_export_targets}
        EXPORT hipdnn_flatbuffers_sdk_targets COMPONENT Development
)

# cmake-lint: disable=E1122
install(
    DIRECTORY ${HIP_DNN_FLATBUFFERS_SDK_INCLUDE_DIR}/
    DESTINATION ${HIP_DNN_FLATBUFFERS_SDK_INSTALL_INCLUDE_DIR}
    COMPONENT Development
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.h.in" EXCLUDE
)

# Install generated version header
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/hipdnn_flatbuffers_sdk/version.h"
    DESTINATION ${HIP_DNN_FLATBUFFERS_SDK_INSTALL_INCLUDE_DIR}/hipdnn_flatbuffers_sdk
    COMPONENT Development
)

install(EXPORT hipdnn_flatbuffers_sdk_targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_flatbuffers_sdk
        FILE hipdnn_flatbuffers_sdkTargets.cmake COMPONENT Development
)

# Install behavior depends on local vs imported dependencies
if(_hipdnn_flatbuffers_sdk_local_export_includes)
    # cmake-lint: disable=E1122
    install(
        DIRECTORY ${_hipdnn_flatbuffers_sdk_local_export_includes}/
        DESTINATION ${HIP_DNN_FLATBUFFERS_SDK_INSTALL_INCLUDE_DIR}
        COMPONENT Development
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "codegen" EXCLUDE
    )
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_flatbuffers_sdkConfig_local.cmake.in"
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_flatbuffers_sdk/hipdnn_flatbuffers_sdkConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_flatbuffers_sdk
    )
else()
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_flatbuffers_sdkConfig_imported.cmake.in"
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_flatbuffers_sdk/hipdnn_flatbuffers_sdkConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_flatbuffers_sdk
    )
endif()

# Generate package version file for find_package version checking
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_flatbuffers_sdk/hipdnn_flatbuffers_sdkConfigVersion.cmake"
    VERSION ${HIPDNN_FLATBUFFERS_SDK_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_flatbuffers_sdk/hipdnn_flatbuffers_sdkConfig.cmake"
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_flatbuffers_sdk/hipdnn_flatbuffers_sdkConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_flatbuffers_sdk
    COMPONENT Development
)

add_subdirectory(tests)
