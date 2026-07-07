# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(CMakePackageConfigHelpers)

# Version handling: independent component version from frontend/version.json.
hipdnn_setup_version(hipdnn_frontend)
project(hipdnn_frontend VERSION ${HIPDNN_FRONTEND_VERSION} LANGUAGES CXX)

# Read data_sdk version for minimum requirement
hipdnn_get_component_version(hipdnn_data_sdk HIPDNN_FRONTEND_DATA_SDK_MIN_VERSION)

# Minimum compatible backend version, read from backend/version.json.
hipdnn_get_component_version(hipdnn_backend HIPDNN_FRONTEND_BACKEND_MIN_VERSION)

set(HIP_DNN_FRONTEND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Two interface targets over the same headers; consumers choose the backend
# loading mode at THEIR configure time by picking which one to link:
#   target_link_libraries(app PRIVATE hipdnn_frontend)          # direct-link
#   target_link_libraries(app PRIVATE hipdnn_frontend_dynamic)  # dlopen at runtime
#
# hipdnn_frontend_dynamic defines HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND (selecting
# HipdnnDynamicBackendWrapper) and does NOT link libhipdnn_backend, so a header-only
# consumer does not inherit a hard dependency (DT_NEEDED) on it. The backend's
# *headers* are still needed to compile the C API types in both modes, so the
# dynamic target borrows the backend's compile usage requirements (include dirs,
# __HIP_PLATFORM_AMD__, hip::host) via generator expressions WITHOUT linking the
# library. ($<COMPILE_ONLY:hipdnn_backend> would express this directly but needs
# CMake >= 3.27; this project pins 3.25.)
add_library(hipdnn_frontend INTERFACE)
add_library(hipdnn_frontend_dynamic INTERFACE)

# Generate the version header once; both targets consume the same generated
# <hipdnn_frontend/version.h> -- in the build tree via the include dir added here,
# in the install tree via the shared install include dir below.
hipdnn_generate_version_header(hipdnn_frontend)
target_include_directories(
    hipdnn_frontend_dynamic INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
)

# Per-library JSON gate. Defaults from the umbrella HIPDNN_SKIP_JSON_LIB when
# building as part of the top-level project; can be set independently when
# frontend is built standalone. Same name as the C++ macro defined on the target.
if(DEFINED HIPDNN_SKIP_JSON_LIB)
    set(_hipdnn_frontend_skip_default ${HIPDNN_SKIP_JSON_LIB})
else()
    set(_hipdnn_frontend_skip_default OFF)
endif()
option(HIPDNN_FRONTEND_SKIP_JSON_LIB
    "Skip nlohmann/json.hpp from frontend public headers."
    ${_hipdnn_frontend_skip_default})
unset(_hipdnn_frontend_skip_default)

if(NOT HIPDNN_FRONTEND_SKIP_JSON_LIB)
    if(NOT TARGET nlohmann_json::nlohmann_json)
        hipdnn_add_dependency(nlohmann_json VERSION ${HIPDNN_NLOHMANN_JSON_VERSION})
    endif()
    hipdnn_add_dependency_includes(hipdnn_frontend nlohmann_json::nlohmann_json)
    hipdnn_add_dependency_includes(hipdnn_frontend_dynamic nlohmann_json::nlohmann_json)
endif()

set(HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/frontend")

# Usage requirements shared by both targets.
foreach(_hipdnn_fe_target hipdnn_frontend hipdnn_frontend_dynamic)
    target_include_directories(
        ${_hipdnn_fe_target} INTERFACE
        $<BUILD_INTERFACE:${HIP_DNN_FRONTEND_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR}>
    )

    # Export installed hipdnn_frontend include paths as system include paths.
    set_target_properties(${_hipdnn_fe_target} PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
            "$<INSTALL_INTERFACE:${HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR}>"
    )

    # Scope the SKIP_JSON_LIB macro to installed (downstream) consumers only.
    # In-tree consumers (frontend tests, etc.) keep access to the json serialize/
    # deserialize methods so the build always succeeds; only external consumers of
    # the installed package see the gated-out public API.
    target_compile_definitions(
        ${_hipdnn_fe_target} INTERFACE
        $<$<BOOL:${HIPDNN_FRONTEND_SKIP_JSON_LIB}>:HIPDNN_FRONTEND_SKIP_JSON_LIB>
        $<$<BOOL:${HIPDNN_ENABLE_SDPA}>:HIPDNN_ENABLE_SDPA>
    )
endforeach()
unset(_hipdnn_fe_target)

# Direct-link target: link the backend library (plus data_sdk).
target_link_libraries(hipdnn_frontend INTERFACE hipdnn_data_sdk hipdnn_backend)

# Backend headers pull in HIP headers, so the dynamic target needs hip::host too.
# (The installed package Config gets HIP transitively via hipdnn_data_sdk's
# find_dependency(hip).)
if(NOT TARGET hip::host)
    find_package(hip REQUIRED)
endif()

# Runtime-load target: link only data_sdk + hip::host and select the dynamic
# wrapper; take the backend's compile usage requirements (include dirs,
# __HIP_PLATFORM_AMD__) via generator expressions so the headers compile, but
# never link libhipdnn_backend. The genexes resolve against the imported
# hipdnn_backend target (provided by find_dependency in the package Config) at the
# consumer's configure time.
target_link_libraries(hipdnn_frontend_dynamic INTERFACE hipdnn_data_sdk hip::host)
target_include_directories(hipdnn_frontend_dynamic INTERFACE
    $<TARGET_PROPERTY:hipdnn_backend,INTERFACE_INCLUDE_DIRECTORIES>
)
target_compile_definitions(hipdnn_frontend_dynamic INTERFACE
    $<TARGET_PROPERTY:hipdnn_backend,INTERFACE_COMPILE_DEFINITIONS>
    HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND
)

# Generate the config file from template
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_frontendConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend
)

export(
    TARGETS hipdnn_frontend hipdnn_frontend_dynamic
    FILE "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend/hipdnn_frontendTargets.cmake"
)

install(TARGETS hipdnn_frontend hipdnn_frontend_dynamic
        EXPORT hipdnn_frontend_targets COMPONENT Development)

# cmake-lint: disable=E1122
# The cuDNN-compatibility shim subtree (include/hipdnn_compatibility/) is excluded
# here and installed separately, gated behind HIPDNN_ENABLE_CUDNN_COMPATIBILITY
# below. REGEX matches each candidate file's full path (forward slashes), which
# contains "/hipdnn_compatibility/"; any EXCLUDE match wins regardless of clause
# order.
install(
    DIRECTORY ${HIP_DNN_FRONTEND_INCLUDE_DIR}/
    DESTINATION ${HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR}
    COMPONENT Development
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.h.in" EXCLUDE
    REGEX "/hipdnn_compatibility/" EXCLUDE
)

# cuDNN-compatibility shim headers — installed only when the shim is enabled,
# alongside hipdnn_frontend headers in the Development component (RFC 0012 §8.5).
if(HIPDNN_ENABLE_CUDNN_COMPATIBILITY)
    # cmake-lint: disable=E1122
    install(
        DIRECTORY ${HIP_DNN_FRONTEND_INCLUDE_DIR}/hipdnn_compatibility/
        DESTINATION ${HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR}/hipdnn_compatibility
        COMPONENT Development
        FILES_MATCHING
        PATTERN "*.h"
    )
endif()

# Install generated version header
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/hipdnn_frontend/version.h"
    DESTINATION ${HIP_DNN_FRONTEND_INSTALL_INCLUDE_DIR}/hipdnn_frontend
    COMPONENT Development
)

install(EXPORT hipdnn_frontend_targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend
        FILE hipdnn_frontendTargets.cmake COMPONENT Development
)

# Generate package version file for find_package version checking
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend/hipdnn_frontendConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake"
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend/hipdnn_frontendConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_frontend
    COMPONENT Development
)

add_subdirectory(tests)
