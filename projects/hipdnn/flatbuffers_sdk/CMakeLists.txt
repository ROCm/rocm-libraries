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
# HIPDNN_FLATBUFFERS_VERSION (set in projects/hipdnn/CMakeLists.txt) is the
# single source of truth. hipdnn_add_dependency uses it as the minimum for
# find_package and as the exact tag for FetchContent. After resolution, fail
# fast if a system-installed FlatBuffers satisfied the loose minimum but does
# not match exactly — the generated headers and runtime are version-locked,
# so a mismatch surfaces later as an opaque static_assert or link error.
# The validation logic is shared with the imported Config template via
# hipdnn_flatbuffers_version_check.cmake.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_flatbuffers_version_check.cmake)
hipdnn_add_dependency(flatbuffers VERSION ${HIPDNN_FLATBUFFERS_VERSION})
hipdnn_check_flatbuffers_version(
    EXPECTED ${HIPDNN_FLATBUFFERS_VERSION}
    CONTEXT  "hipDNN"
    DETAIL   "Point flatbuffers_DIR at the matching package, uninstall the system FlatBuffers so hipDNN can fetch the correct version, or update HIPDNN_FLATBUFFERS_VERSION to match."
)

# Per-library JSON gate. Defaults from the umbrella HIPDNN_SKIP_JSON_LIB when
# building as part of the top-level project; can be set independently when
# flatbuffers_sdk is built standalone.
if(DEFINED HIPDNN_SKIP_JSON_LIB)
    set(_hipdnn_flatbuffers_sdk_skip_default ${HIPDNN_SKIP_JSON_LIB})
else()
    set(_hipdnn_flatbuffers_sdk_skip_default OFF)
endif()
option(HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
    "Skip nlohmann/json.hpp from flatbuffers_sdk public headers."
    ${_hipdnn_flatbuffers_sdk_skip_default})
unset(_hipdnn_flatbuffers_sdk_skip_default)

if(NOT HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB AND NOT TARGET nlohmann_json::nlohmann_json)
    hipdnn_add_dependency(nlohmann_json VERSION ${HIPDNN_NLOHMANN_JSON_VERSION})
endif()


# Upstream FlatBuffers exposes FlatBuffers when fetched via add_subdirectory,
# while its installed package exposes flatbuffers::flatbuffers. Use the installed-package
# spelling internally so the build and exported config follow the same dependency model.
if(TARGET FlatBuffers AND NOT TARGET flatbuffers::flatbuffers)
    add_library(flatbuffers::flatbuffers ALIAS FlatBuffers)
endif()
# --- Create the library target ---
add_library(hipdnn_flatbuffers_sdk INTERFACE)

# Generate version header
hipdnn_generate_version_header(hipdnn_flatbuffers_sdk)

# Link to data_sdk for shared type definitions (types.hpp) and the
# hipdnn_add_dependency_includes() function provided by data_sdk's CMake config.
target_link_libraries(hipdnn_flatbuffers_sdk INTERFACE hipdnn_data_sdk)

# Add FlatBuffers includes
hipdnn_add_dependency_includes(hipdnn_flatbuffers_sdk flatbuffers::flatbuffers)
if(NOT HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB)
    hipdnn_add_dependency_includes(hipdnn_flatbuffers_sdk nlohmann_json::nlohmann_json)
endif()

# Scope the SKIP_JSON_LIB macro to installed (downstream) consumers only.
# In-tree consumers (e.g. the backend, which always needs json) must not see it,
# otherwise flatbuffers_sdk's to_json overloads in utilities/json/*.hpp disappear
# and the backend fails to build.
target_compile_definitions(
    hipdnn_flatbuffers_sdk INTERFACE
    $<INSTALL_INTERFACE:$<$<BOOL:${HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB}>:HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB>>
)

target_include_directories(
    hipdnn_flatbuffers_sdk INTERFACE $<BUILD_INTERFACE:${HIP_DNN_FLATBUFFERS_SDK_INCLUDE_DIR}>
                                     $<INSTALL_INTERFACE:${HIP_DNN_FLATBUFFERS_SDK_INSTALL_INCLUDE_DIR}>
)

# --- Automatic FlatBuffer header generation from schemas ---
if(HIPDNN_GENERATE_SDK_HEADERS)
    include(../cmake/FlatBuffersGenerate.cmake)

    # GENERATED_INCLUDE_DIR points directly at the data_objects/ directory because
    # the .fbs files declare `namespace hipdnn_flatbuffers_sdk.data_objects;`, so
    # flatc writes <name>_generated.h directly there. Downstream code includes via
    # <hipdnn_flatbuffers_sdk/data_objects/<name>_generated.h>, which resolves
    # against the include/ root (BUILD_INTERFACE) or the install include root.
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
            schemas/execution_plan.fbs
            schemas/graph.fbs
            schemas/knob_value.fbs
            schemas/layernorm_attributes.fbs
            schemas/layernorm_backward_attributes.fbs
            schemas/matmul_attributes.fbs
            schemas/norm_common.fbs
            schemas/pointwise_attributes.fbs
            schemas/rmsnorm_attributes.fbs
            schemas/rmsnorm_backward_attributes.fbs
            schemas/block_scale_quantize_attributes.fbs
            schemas/sdpa_attributes.fbs
            schemas/sdpa_backward_attributes.fbs
            schemas/reduction_attributes.fbs
            schemas/resample_common.fbs
            schemas/resample_fwd_attributes.fbs
            schemas/tensor_attributes.fbs
        SCHEMAS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/schemas"
        GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/hipdnn_flatbuffers_sdk/data_objects"
        OUTPUT_NAMESPACE hipdnn_flatbuffers_sdk
    )
endif()

# --- Export and install ---
export(TARGETS hipdnn_flatbuffers_sdk
       FILE "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_flatbuffers_sdk/hipdnn_flatbuffers_sdkTargets.cmake"
)

install(TARGETS hipdnn_flatbuffers_sdk EXPORT hipdnn_flatbuffers_sdk_targets COMPONENT Development)

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

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_flatbuffers_sdkConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_flatbuffers_sdk/hipdnn_flatbuffers_sdkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_flatbuffers_sdk
)

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

# Ship the shared version-check helper so the installed imported Config can
# include() it (single-source the comparison logic between build- and
# consumer-time validation).
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_flatbuffers_version_check.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_flatbuffers_sdk
    COMPONENT Development
)

add_subdirectory(tests)
