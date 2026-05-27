# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# OBJECT library with all implementation
add_library(
    hip_kernel_provider_impl OBJECT
    HipKernelHandle.cpp
    HipKernelContainer.cpp
    HipKernelPluginPublic.cpp
    HipKernelUtils.cpp
    hip/HipKernel.cpp
    hip/HipProgram.cpp
)

add_subdirectory(engines)

target_include_directories(hip_kernel_provider_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../include
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/engines/
)
dnn_provider_generate_version_header(hip_kernel_provider hip_kernel_provider_impl ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(hip_kernel_provider_impl PUBLIC hip::host hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk hiprtc::hiprtc)

target_compile_options(hip_kernel_provider_impl PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(hip_kernel_provider_impl PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)
clang_tidy_check(hip_kernel_provider_impl)
add_clang_tidy_custom_target()

# Static library for unit tests to link against (provides access to internal implementation)
add_library(hip_kernel_provider_private STATIC $<TARGET_OBJECTS:hip_kernel_provider_impl>)
target_include_directories(hip_kernel_provider_private PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/include
)
target_link_libraries(hip_kernel_provider_private PUBLIC hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk hiprtc::hiprtc hip_kernel_provider_impl)
if(ENABLE_ASM_SDPA_ENGINE)
    target_compile_definitions(hip_kernel_provider_private PUBLIC HIPDNN_ENGINE_ASM_SDPA=1)
endif()

if(ENABLE_HIP_MLOPS_ENGINE)
    target_compile_definitions(hip_kernel_provider_private PUBLIC HIPDNN_ENGINE_HIP_MLOPS=1)
endif()

# hip_kernel_provider - SHARED library (the actual plugin)
add_library(hip_kernel_provider SHARED $<TARGET_OBJECTS:hip_kernel_provider_impl>)
target_link_libraries(hip_kernel_provider PRIVATE hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk hiprtc::hiprtc hip::host)

if(NOT WIN32)
    target_link_libraries(hip_kernel_provider PRIVATE "-Xlinker --exclude-libs=ALL")
endif()

target_compile_options(hip_kernel_provider PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(
    hip_kernel_provider
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               INSTALL_RPATH "${ROCM_PATH}/lib"
               INSTALL_RPATH_USE_LINK_PATH TRUE
               BUILD_WITH_INSTALL_RPATH TRUE
)

# Set RPATH origin for TheRock integration
set_target_properties(hip_kernel_provider PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN "lib/hipdnn_plugins/engines")

# ============================================================================
# Tests
# ============================================================================
if(HIPKERNELPROVIDER_ENABLE_TESTS)
    add_subdirectory(tests)
    add_subdirectory(integration_tests)

    # Keep this after all build folders have been added so they have a chance to register their tests
    # using add_*_test_target()
    if(HIPKERNELPROVIDER_ENABLE_COVERAGE)
        finalize_test_targets("${PROJECT_NAME}" ENABLE_COVERAGE)
    else()
        finalize_test_targets("${PROJECT_NAME}")
    endif()

    install_provider_ctest_files("hip_kernel_provider")
endif()

# Process all kernels marked for inlining
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/KernelEmbedding.cmake")
# Generate embedded kernel sources and includes at configure time
embed_kernel_sources(
    ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_sources.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_sources.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_includes.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_includes.hpp
)

set(HIP_KERNEL_PROVIDER_GENERATED_SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_sources.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_includes.cpp)

set_source_files_properties(${HIP_KERNEL_PROVIDER_GENERATED_SOURCES} TARGET_DIRECTORY hip_kernel_provider_impl PROPERTIES COMPILE_OPTIONS "-Wno-overlength-strings")

target_sources(hip_kernel_provider_impl PRIVATE "${HIP_KERNEL_PROVIDER_GENERATED_SOURCES}")
