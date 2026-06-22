# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# TEMPLATE ADAPTATION: Update the source file list below with your plugin's .cpp files. The
# three-target structure (OBJECT -> STATIC for tests + SHARED for plugin) should be kept as-is as it
# allows unit tests to link against plugin internals without loading the plugin as a shared library.

# =============================================================================
# OBJECT library: contains all plugin implementation source files.
# Both the shared library (actual plugin) and static library (for unit tests)
# are built from these same object files.
# =============================================================================
add_library(
    example_provider_impl_kernel_sources OBJECT
    ${KERNEL_GENERATED_SOURCES}
)

add_dependencies(example_provider_impl_kernel_sources example_provider_kernel_embed)

target_include_directories(example_provider_impl_kernel_sources PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../kernels)
target_link_libraries(example_provider_impl_kernel_sources PRIVATE example_provider_compile_options)

set_target_properties(example_provider_impl_kernel_sources PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
    CXX_CLANG_TIDY ""
)

add_library(
    example_provider_impl OBJECT
    ExampleProviderHandle.cpp
    ExampleProviderContainer.cpp
    ExampleProviderPluginPublic.cpp
    engines/ExampleProviderEngine.cpp
    engines/plans/ReluPlanBuilder.cpp
    engines/plans/ReluPlan.cpp
    engines/plans/ConvFwdPlanBuilder.cpp
    engines/plans/ConvFwdPlan.cpp
    hip/HipCompiledProgram.cpp
    hip/HipRunnableKernel.cpp
)

target_include_directories(example_provider_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(example_provider_impl PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../kernels)
target_link_libraries(example_provider_impl PUBLIC hipdnn_plugin_sdk hipdnn_data_sdk hip::host hiprtc::hiprtc)
target_link_libraries(example_provider_impl PRIVATE example_provider_compile_options)

set_target_properties(example_provider_impl PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

# =============================================================================
# Static library: exposes plugin internals for unit tests to link against.
# This is not installed or distributed. It exists only for testability.
# =============================================================================
add_library(example_provider_private STATIC
    $<TARGET_OBJECTS:example_provider_impl>
    $<TARGET_OBJECTS:example_provider_impl_kernel_sources>
)
target_include_directories(example_provider_private PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(example_provider_private PUBLIC hipdnn_plugin_sdk hipdnn_data_sdk example_provider_impl)

# =============================================================================
# Shared library: the actual plugin loaded at runtime by hipDNN.
# Visibility is set to hidden so only the C entry points are exported.
# =============================================================================
add_library(example_provider_plugin SHARED
    $<TARGET_OBJECTS:example_provider_impl>
    $<TARGET_OBJECTS:example_provider_impl_kernel_sources>
)
target_link_libraries(example_provider_plugin PRIVATE hipdnn_plugin_sdk hipdnn_data_sdk hip::host hiprtc::hiprtc)

if(NOT WIN32)
    target_link_libraries(example_provider_plugin PRIVATE "-Xlinker --exclude-libs=ALL")
endif()

set_target_properties(example_provider_plugin PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../.."
    INSTALL_RPATH_USE_LINK_PATH TRUE
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
