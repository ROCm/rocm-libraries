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
# single source of truth. hipdnn_add_dependency uses it as an exact find_package
# version and as the exact tag for FetchContent. A non-matching installed
# FlatBuffers package is ignored so hipDNN can fetch the supported version.
# The validation logic is shared with the imported Config template via
# hipdnn_flatbuffers_version_check.cmake.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_flatbuffers_version_check.cmake)
if(DEFINED _HIPDNN_DEFAULT_FLATBUFFERS_VERSION
       AND NOT HIPDNN_FLATBUFFERS_VERSION STREQUAL "${_HIPDNN_DEFAULT_FLATBUFFERS_VERSION}")
    message(
        WARNING
            "HIPDNN_FLATBUFFERS_VERSION was overridden to ${HIPDNN_FLATBUFFERS_VERSION}. "
            "hipDNN officially supports FlatBuffers ${_HIPDNN_DEFAULT_FLATBUFFERS_VERSION}; "
            "other versions may fail to build or generate incompatible headers."
    )
endif()

hipdnn_add_dependency(
    flatbuffers
    VERSION ${HIPDNN_FLATBUFFERS_VERSION}
    FIND_PACKAGE_ARGS EXACT
)
hipdnn_check_flatbuffers_version(
    EXPECTED ${HIPDNN_FLATBUFFERS_VERSION}
    CONTEXT  "hipDNN"
    DETAIL   "Point flatbuffers_DIR at the matching package, remove stale CMake cache entries, or update HIPDNN_FLATBUFFERS_VERSION to intentionally use a different version."
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

# The two flags are individually valid but the combination is not: SKIP_JSON_LIB compiles
# GraphContentKey::toJson()/fromJson() out of existence (GraphContentKey.hpp's
# #ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB guard), while the ingestor's
# WinnerCacheFile.hpp calls both unconditionally. In-tree this never surfaces because the
# macro is scoped to $<INSTALL_INTERFACE:> below -- an in-tree build of the ingestor never
# sees HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB regardless of this option's value, so CI cannot
# catch it. Only an installed consumer that also enables the ingestor hits the missing
# symbols, at their build time with no diagnostic pointing back here. Reject the pair now,
# at hipDNN's own configure time, so a broken installed artifact is never produced.
if(HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB AND HIPDNN_ENABLE_KERNEL_INGESTOR)
    message(FATAL_ERROR
        "HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB and HIPDNN_ENABLE_KERNEL_INGESTOR cannot both "
        "be enabled: the kernel ingestor's winner cache (WinnerCacheFile.hpp) calls "
        "GraphContentKey::toJson()/fromJson() unconditionally, and SKIP_JSON_LIB compiles "
        "those out of GraphContentKey.hpp. This combination fails to compile for any "
        "consumer that installs this hipDNN build and links the ingestor."
    )
endif()

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
    # Single source of truth for the schema set: reused below so the cache-key
    # custom command's DEPENDS can't drift out of sync with what flatc (and
    # therefore gen_cache_key.py, which reads the same schemas transitively
    # through graph.fbs's includes) actually consumes.
    set(_hipdnn_flatbuffers_sdk_schemas
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
        schemas/serialized_graph_and_plan.fbs
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
        schemas/resample_bwd_attributes.fbs
        schemas/tensor_attributes.fbs
        schemas/moe_grouped_matmul_attributes.fbs
        schemas/moe_grouped_matmul_bwd_attributes.fbs
    )
    hipdnn_generate_flatbuffer_headers(
        TARGET hipdnn_flatbuffers_sdk
        SCHEMAS ${_hipdnn_flatbuffers_sdk_schemas}
        SCHEMAS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/schemas"
        GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/hipdnn_flatbuffers_sdk/data_objects"
        OUTPUT_NAMESPACE hipdnn_flatbuffers_sdk
    )

    # Not generated by flatc: the cache-key annotations are invisible to its C++ output,
    # so gen_cache_key.py re-derives this header from the binary schema. Regenerated
    # whenever the schemas are, independent of HIPDNN_ENABLE_KERNEL_INGESTOR: the header
    # is schema-derived and installs either way, so gating it would let a schema edit
    # leave the committed copy stale with nothing to catch it.
    #
    # Python3 is optional here, matching cmake/Tests.cmake: a toolchain without it falls
    # back to the committed header rather than failing to configure.
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
        set(_cache_key_header
            "${CMAKE_CURRENT_SOURCE_DIR}/include/hipdnn_flatbuffers_sdk/data_objects/cachekey_generated.h"
        )
        # DEPENDS reuses _hipdnn_flatbuffers_sdk_schemas (same list flatc consumes above)
        # rather than re-listing graph.fbs's ~20 transitively-included attribute schemas:
        # gen_cache_key.py reads their fields through flatc just like the C++ generation
        # does, so a schema added to that list is automatically covered here too.
        list(TRANSFORM _hipdnn_flatbuffers_sdk_schemas
             PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/"
             OUTPUT_VARIABLE _cache_key_schema_depends
        )
        # Pass the flatc the build already resolved rather than letting the script search
        # PATH: a superbuild gets flatc as a CMake target, which is never on PATH.
        _hipdnn_resolve_flatc_command(_cache_key_flatc _cache_key_flatc_dependency)
        add_custom_command(
            OUTPUT "${_cache_key_header}"
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/gen_cache_key.py" --flatc
                    "${_cache_key_flatc}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/gen_cache_key.py"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/default_flatc_version.txt"
                    ${_cache_key_schema_depends} ${_cache_key_flatc_dependency}
            COMMENT "Generating cachekey_generated.h from the FlatBuffers schemas"
            VERBATIM
        )
        add_custom_target(
            generate_hipdnn_cache_key
            DEPENDS "${_cache_key_header}"
            COMMENT "Cache-key header generation"
        )
        add_dependencies(hipdnn_flatbuffers_sdk generate_hipdnn_cache_key)
    endif()
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
