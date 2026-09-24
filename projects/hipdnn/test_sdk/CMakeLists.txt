# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(CMakePackageConfigHelpers)

# Version handling
hipdnn_setup_version(hipdnn_test_sdk)

# Set project with version
project(hipdnn_test_sdk VERSION ${HIPDNN_TEST_SDK_VERSION} LANGUAGES CXX)

# Read data_sdk version for minimum requirement
hipdnn_get_component_version(hipdnn_data_sdk HIPDNN_TEST_SDK_DATA_SDK_MIN_VERSION)

# Read plugin_sdk version for minimum requirement
hipdnn_get_component_version(hipdnn_plugin_sdk HIPDNN_TEST_SDK_PLUGIN_SDK_MIN_VERSION)

set(HIP_DNN_TEST_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(hipdnn_test_sdk INTERFACE)
add_library(hipdnn_test_sdk_dynamic INTERFACE)

# Generate version header
hipdnn_generate_version_header(hipdnn_test_sdk)

find_package(hip REQUIRED)
target_link_libraries(hipdnn_test_sdk INTERFACE hipdnn_flatbuffers_sdk hipdnn_plugin_sdk hipdnn_frontend hip::host)
target_link_libraries(
    hipdnn_test_sdk_dynamic INTERFACE hipdnn_flatbuffers_sdk hipdnn_plugin_sdk hipdnn_frontend_dynamic hip::host
)

set(HIP_DNN_TEST_SDK_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/test_sdk")

# src/AsanDefaultSuppressions.cpp defines __asan_default_suppressions(), which supplies the
# suppression text from inside the binary and so needs no file, no install rule and no path that
# survives relocation.
#
# It is attached INTERFACE so it compiles into each consuming executable. The symbol must end up in
# the executable itself: a definition inside a shared library, or inside a plain static archive, is
# not picked up. It must also appear exactly once per link, so only these two targets carry it and
# nothing else may add it -- a second definition is a duplicate-symbol link error, and a link that
# resolves two weak definitions silently keeps only one.
#
# The source is installed and named again under INSTALL_INTERFACE so consumers that resolve this
# package with find_package() -- a plugin or application built against an installed hipDNN -- compile
# it as well. CMake rewrites that path relative to the importing prefix, so a relocated install tree
# still resolves it.
#
# The gate is the producing build's, so a hipDNN installed without ASAN exports nothing here.
set(HIP_DNN_TEST_SDK_INSTALL_SRC_DIR "${CMAKE_INSTALL_DATADIR}/hipdnn/test_sdk")

if(BUILD_ADDRESS_SANITIZER
   OR THEROCK_SANITIZER STREQUAL "ASAN"
   OR THEROCK_SANITIZER STREQUAL "HOST_ASAN"
)
    foreach(_hipdnn_test_sdk_target hipdnn_test_sdk hipdnn_test_sdk_dynamic)
        target_sources(
            ${_hipdnn_test_sdk_target}
            INTERFACE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/AsanDefaultSuppressions.cpp>
                $<INSTALL_INTERFACE:${HIP_DNN_TEST_SDK_INSTALL_SRC_DIR}/AsanDefaultSuppressions.cpp>
        )
    endforeach()

    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/AsanDefaultSuppressions.cpp
            DESTINATION ${HIP_DNN_TEST_SDK_INSTALL_SRC_DIR} COMPONENT Development
    )
endif()

foreach(_hipdnn_test_sdk_target hipdnn_test_sdk hipdnn_test_sdk_dynamic)
    target_include_directories(
        ${_hipdnn_test_sdk_target}
        INTERFACE $<BUILD_INTERFACE:${HIP_DNN_TEST_SDK_INCLUDE_DIR}>
                  $<INSTALL_INTERFACE:${HIP_DNN_TEST_SDK_INSTALL_INCLUDE_DIR}>
    )
endforeach()

# Generate the config file from template
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipdnn_test_sdkConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk/hipdnn_test_sdkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk
)

export(
    TARGETS hipdnn_test_sdk hipdnn_test_sdk_dynamic
    FILE "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk/hipdnn_test_sdkTargets.cmake"
)

install(TARGETS hipdnn_test_sdk hipdnn_test_sdk_dynamic EXPORT hipdnn_test_sdk_targets COMPONENT Development)

# cmake-lint: disable=E1122
install(
    DIRECTORY ${HIP_DNN_TEST_SDK_INCLUDE_DIR}/
    DESTINATION ${HIP_DNN_TEST_SDK_INSTALL_INCLUDE_DIR}
    COMPONENT Development
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
    PATTERN "*.h.in" EXCLUDE
)

# Install generated version header
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/hipdnn_test_sdk/version.h"
    DESTINATION ${HIP_DNN_TEST_SDK_INSTALL_INCLUDE_DIR}/hipdnn_test_sdk
    COMPONENT Development
)

install(EXPORT hipdnn_test_sdk_targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk
        FILE hipdnn_test_sdkTargets.cmake COMPONENT Development
)

# Generate package version file for find_package version checking
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk/hipdnn_test_sdkConfigVersion.cmake"
    VERSION ${HIPDNN_TEST_SDK_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk/hipdnn_test_sdkConfig.cmake"
        "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk/hipdnn_test_sdkConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipdnn_test_sdk
    COMPONENT Development
)

add_subdirectory(tests)
