# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(CMakePackageConfigHelpers)

# Version handling
hipdnn_setup_version(hipdnn_plugin_sdk)

# Set project with version
project(hipdnn_plugin_sdk VERSION ${HIPDNN_PLUGIN_SDK_VERSION} LANGUAGES CXX)

# Read data_sdk version for minimum requirement
hipdnn_get_component_version(hipdnn_data_sdk HIPDNN_PLUGIN_SDK_DATA_SDK_MIN_VERSION)

set(HIP_DNN_PLUGIN_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(hipdnn_plugin_sdk INTERFACE)

# Generate version header
hipdnn_generate_version_header(hipdnn_plugin_sdk)

target_link_libraries(hipdnn_plugin_sdk INTERFACE hipdnn_flatbuffers_sdk hipdnn_data_sdk)

# Generic GPU target table: data/gpu_generic_targets.json is the single source of truth.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ValidateGpuGenericTargets.cmake")
set(HIPDNN_GPU_GENERIC_TARGETS_JSON_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/data/gpu_generic_targets.json")
hipdnn_validate_gpu_generic_targets("${HIPDNN_GPU_GENERIC_TARGETS_JSON_SOURCE}")
set_property(
    DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${HIPDNN_GPU_GENERIC_TARGETS_JSON_SOURCE}"
)

file(READ "${HIPDNN_GPU_GENERIC_TARGETS_JSON_SOURCE}" _generic_targets_json)
string(JSON _generic_count LENGTH "${_generic_targets_json}" generics)
set(HIPDNN_GENERIC_TARGET_MEMBER_ARRAYS "")
set(HIPDNN_GENERIC_TARGET_ROWS "")
set(HIPDNN_GENERIC_TARGET_ROW_COUNT ${_generic_count})
math(EXPR _last_generic "${_generic_count} - 1")
# cmake-lint: disable=E1120
foreach(_g RANGE 0 ${_last_generic})
    string(JSON _generic MEMBER "${_generic_targets_json}" generics ${_g})
    string(TOUPPER "${_generic}" _array_name)
    string(REPLACE "-" "_" _array_name "${_array_name}")
    string(JSON _member_count LENGTH "${_generic_targets_json}" generics "${_generic}")
    math(EXPR _last_member "${_member_count} - 1")
    set(_member_literals "")
    foreach(_m RANGE 0 ${_last_member})
        string(JSON _member GET "${_generic_targets_json}" generics "${_generic}" ${_m})
        string(APPEND _member_literals "    \"${_member}\",\n")
    endforeach()
    string(APPEND HIPDNN_GENERIC_TARGET_MEMBER_ARRAYS
           "inline constexpr std::string_view ${_array_name}_MEMBERS[] = {\n${_member_literals}};\n\n")
    string(APPEND HIPDNN_GENERIC_TARGET_ROWS
           "    {\"${_generic}\", ${_array_name}_MEMBERS, ${_member_count}},\n")
endforeach()
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GpuGenericTargetsTable.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/hipdnn_plugin_sdk/GpuGenericTargetsTable.hpp" @ONLY
)

# Beside the build-tree Config, as the installed copy sits beside the installed Config.
set(_generic_targets_build_copy
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk/gpu_generic_targets.json"
)
configure_file("${HIPDNN_GPU_GENERIC_TARGETS_JSON_SOURCE}" "${_generic_targets_build_copy}" COPYONLY)
# CACHE INTERNAL so sibling subdirectories see it.
set(HIPDNN_PLUGIN_SDK_GPU_GENERIC_TARGETS_JSON "${_generic_targets_build_copy}"
    CACHE INTERNAL "Path to the generic GPU target table JSON shipped with the plugin SDK"
)

# The ingestor headers are wholly inside `#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR`, and they
# are installed whenever it is set. Carrying the definition on the exported target is what
# makes them usable: without it a consumer that find_package()s this SDK gets the headers
# on disk and they preprocess to nothing, so the types are simply absent with no
# diagnostic naming the cause. INTERFACE propagates to both the build tree and the
# installed hipdnn_plugin_sdkTargets.cmake, so in-tree and out-of-tree consumers agree.
if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    target_compile_definitions(hipdnn_plugin_sdk INTERFACE HIPDNN_ENABLE_KERNEL_INGESTOR)

    # DescriptorLoader.hpp includes <nlohmann/json.hpp> directly. Today that only compiles
    # because flatbuffers_sdk happens to add the includes transitively, and only
    # conditionally (HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB can turn it off); this target
    # needs its own dependency rather than relying on a sibling's implementation detail.
    if(NOT TARGET nlohmann_json::nlohmann_json)
        hipdnn_add_dependency(nlohmann_json VERSION ${HIPDNN_NLOHMANN_JSON_VERSION})
    endif()
    hipdnn_add_dependency_includes(hipdnn_plugin_sdk nlohmann_json::nlohmann_json)
endif()

set(HIP_DNN_PLUGIN_SDK_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/plugin_sdk")

target_include_directories(
    hipdnn_plugin_sdk INTERFACE $<BUILD_INTERFACE:${HIP_DNN_PLUGIN_SDK_INCLUDE_DIR}>
                                $<INSTALL_INTERFACE:${HIP_DNN_PLUGIN_SDK_INSTALL_INCLUDE_DIR}>
)

# Generate the config file from template
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_plugin_sdkConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk/hipdnn_plugin_sdkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk
)

export(
    TARGETS hipdnn_plugin_sdk
    FILE "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk/hipdnn_plugin_sdkTargets.cmake"
)

install(TARGETS hipdnn_plugin_sdk EXPORT hipdnn_plugin_sdk_targets COMPONENT Development)

# cmake-lint: disable=E1122
#
# ingestor/ headers are WIP (RFC 0017); this glob is unconditional, so exclude them
# explicitly unless the feature is enabled.
set(HIP_DNN_PLUGIN_SDK_INSTALL_EXCLUDE_PATTERNS "")
if(NOT HIPDNN_ENABLE_KERNEL_INGESTOR)
    # Matches the directory itself (not "ingestor/*"): under FILES_MATCHING, excluding
    # only the contents still creates the empty directory in the install tree.
    list(APPEND HIP_DNN_PLUGIN_SDK_INSTALL_EXCLUDE_PATTERNS PATTERN "ingestor" EXCLUDE)
endif()

install(
    DIRECTORY ${HIP_DNN_PLUGIN_SDK_INCLUDE_DIR}/
    DESTINATION ${HIP_DNN_PLUGIN_SDK_INSTALL_INCLUDE_DIR}
    COMPONENT Development
    FILES_MATCHING
    PATTERN "*.inl"
    PATTERN "*.hpp"
    PATTERN "*.h"
    PATTERN "*.h.in" EXCLUDE
    ${HIP_DNN_PLUGIN_SDK_INSTALL_EXCLUDE_PATTERNS}
)

# Install generated version header
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/hipdnn_plugin_sdk/version.h"
    DESTINATION ${HIP_DNN_PLUGIN_SDK_INSTALL_INCLUDE_DIR}/hipdnn_plugin_sdk
    COMPONENT Development
)

install(
    FILES "${HIPDNN_GPU_GENERIC_TARGETS_JSON_SOURCE}"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk
    COMPONENT Development
)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/hipdnn_plugin_sdk/GpuGenericTargetsTable.hpp"
    DESTINATION ${HIP_DNN_PLUGIN_SDK_INSTALL_INCLUDE_DIR}/hipdnn_plugin_sdk
    COMPONENT Development
)

install(EXPORT hipdnn_plugin_sdk_targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk
        FILE hipdnn_plugin_sdkTargets.cmake COMPONENT Development
)

# Generate package version file for find_package version checking
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk/hipdnn_plugin_sdkConfigVersion.cmake"
    VERSION ${HIPDNN_PLUGIN_SDK_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk/hipdnn_plugin_sdkConfig.cmake"
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk/hipdnn_plugin_sdkConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_plugin_sdk
    COMPONENT Development
)

add_subdirectory(tests)
