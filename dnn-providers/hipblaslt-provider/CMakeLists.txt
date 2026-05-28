# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
add_compile_definitions(__HIP_PLATFORM_AMD__)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ClangToolChain.cmake"
    CACHE STRING "Toolchain file"
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/VersionUtils.cmake)
dnn_provider_setup_version(hipblaslt_provider ${CMAKE_CURRENT_LIST_DIR})

project(hipblaslt-provider VERSION ${HIPBLASLT_PROVIDER_VERSION} LANGUAGES CXX)

if(NOT WIN32)
    # Only set if not already specified by user
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "Install path prefix" FORCE)
    endif()
    message(STATUS "Setting CMAKE_INSTALL_PREFIX to ${CMAKE_INSTALL_PREFIX}")
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE
    CACHE BOOL "Add paths to linker search and installed rpath"
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake
     ${CMAKE_CURRENT_SOURCE_DIR}/../cmake ${ROCM_PATH}/lib/cmake/hip ${HIP_DIR}/cmake
)

option(HIPDNN_SKIP_TESTS "Skip building tests" OFF)
option(ENABLE_CLANG_FORMAT "Enable Clang-Format checks" ON)
if(NOT ENABLE_CLANG_FORMAT)
    message(WARNING "Clang-Format checks are disabled. CI requires passing clang-format checks.")
endif()

list(
    PREPEND
    HIPDNN_WARNING_COMPILE_OPTIONS
    -Werror # Treat all warnings as errors
    -Wall # Enable most common warnings
    -Wextra # Enable additional warnings not covered by -Wall
    -Wpedantic # Enforce strict ISO C++ compliance
    -Wshadow # Warn about variable shadowing
    -Wnon-virtual-dtor # Warn if a class with virtual functions has a non-virtual destructor
    -Wold-style-cast # Warn about C-style casts
    -Wcast-align # Warn about potential performance issues with misaligned casts
    -Woverloaded-virtual # Warn if a base class function is hidden by a derived class function
                         # with the same name
    -Wconversion # Warn about implicit type conversions that may alter a value
    -Wsign-conversion # Warn about implicit conversions between signed and unsigned types
    -Wnull-dereference # Warn about dereferencing null pointers
    -Wdouble-promotion # Warn when a float is implicitly promoted to a double
    -Wformat=2 # Enable stricter format string checks
    -Winit-self # Warn about variables initialized with itself
    -Wunreachable-code # Warn about unreachable code
    -Wno-error=unused-command-line-argument # Disable error for unused command line arguments
    -Wswitch-default # Warn if a switch statement does not have a default case
    -Wno-ignored-attributes # disable warning about unused declspec(dllexport)
)

# NOTE: Workaround until llvm & hip cmake modules fixes symlink logic in their config files;
# remove when fixed. Added to prevent issues with not finding hip cmake.
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH} ${ROCM_PATH}/hip)

find_package(hip REQUIRED)

# SDK dependencies - use targets directly in superbuild, find_package otherwise
if(NOT TARGET hipdnn_data_sdk)
    find_package(hipdnn_data_sdk CONFIG REQUIRED)
endif()

if(NOT TARGET hipdnn_flatbuffers_sdk)
    find_package(hipdnn_flatbuffers_sdk CONFIG REQUIRED)
endif()

if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

# Sanitizer support (Address Sanitizer, etc.)
include(Sanitizers)

# Setup testing dependencies, tests & clang format
include(Dependencies)
if (NOT HIPDNN_SKIP_TESTS)
    fetch_gtest_dependency()
    set(SKIP_TEST_NAME_VALIDATION ON)
    include(Tests)
endif()
# TODO: Remove this clang-format probe once TheRock's ml-libs/CMakeLists.txt passes
# -DENABLE_CLANG_FORMAT=OFF for hipblasltprovider (as it already does for miopenprovider).
# The common ClangCheck.cmake calls findAndCheckClangFormat() which FATAL_ERRORs when
# clang-format is missing. TheRock's build container does not provide clang-format, so we
# probe for it here and skip the include if absent. Other providers (miopen, hip-kernel,
# integration-tests) rely on TheRock passing -DENABLE_CLANG_FORMAT=OFF instead.
if(ENABLE_CLANG_FORMAT)
    find_program(_HIPBLASLT_CLANG_FORMAT_PROBE NAMES clang-format-18 clang-format)
    if(_HIPBLASLT_CLANG_FORMAT_PROBE)
        include(ClangCheck)
    else()
        message(WARNING "ENABLE_CLANG_FORMAT=ON but clang-format not found; skipping format targets")
    endif()
endif()

if(WIN32)
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "bin")
else()
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "lib")
endif()

set(HIPDNN_BUILD_PLUGIN_ENGINE_DIR
    "${CMAKE_BINARY_DIR}/${HIPDNN_PLUGIN_ENGINE_ROOTDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}"
)

message(STATUS "Plugin build directory: ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}")
message(STATUS "Plugin install directory: ${HIPDNN_INSTALL_PLUGIN_ENGINE_DIR}")
file(MAKE_DIRECTORY ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR})

find_package(hipblaslt CONFIG REQUIRED)

add_library(
    hipblaslt_plugin_private
    engines/HipblasltEngine.cpp
    EngineManager.cpp
    HipblasltHandleFactory.cpp
    HipblasltContainer.cpp
    HipblasltPlugin.cpp
    HipblasltUtils.cpp
    # Matmul-related files
    engines/plans/HipblasltMatmulPlan.cpp
    engines/plans/HipblasltMatmulPlanBuilder.cpp
    HipblasltMatmulDesc.cpp
    HipblasltMatrixLayout.cpp
)
target_include_directories(hipblaslt_plugin_private PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
dnn_provider_generate_version_header(hipblaslt_provider hipblaslt_plugin_private ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(hipblaslt_plugin_private PUBLIC roc::hipblaslt hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk hip::host)

target_compile_options(hipblaslt_plugin_private PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

add_library(hipblaslt_plugin SHARED HipblasltPluginPublic.cpp)
target_link_libraries(hipblaslt_plugin PRIVATE hipblaslt_plugin_private)

if(NOT WIN32)
    target_link_libraries(
        hipblaslt_plugin PRIVATE "-Xlinker --exclude-libs=ALL"
    ) # Hide symbols from linked static libs
endif()

target_compile_options(hipblaslt_plugin PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(
    hipblaslt_plugin
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               INSTALL_RPATH "${ROCM_PATH}/lib"
               INSTALL_RPATH_USE_LINK_PATH TRUE
               BUILD_WITH_INSTALL_RPATH TRUE
)

if(NOT HIPDNN_SKIP_TESTS)
    add_subdirectory(tests)
    add_subdirectory(integration_tests)

    # Keep this after all build folders have been added so they have a chance to register their tests
    # using add_*_test_target()
    # Finalize test targets with prefix to avoid collisions in superbuild mode
    finalize_test_targets("${PROJECT_NAME}")

    # For standalone builds we need to install the ctest files
    if(NOT ROCM_LIBS_SUPERBUILD)
        install_provider_ctest_files("hipblaslt_plugin")
    endif()
endif()

install(TARGETS hipblaslt_plugin EXPORT hipblaslt-plugin
        LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
        RUNTIME DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
)
