# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# =============================================================================
# OBJECT library: contains all plugin implementation source files.
# Both the shared library (actual plugin) and static library (for unit tests)
# are built from these same object files.
# =============================================================================
add_library(
    example_plugin_impl OBJECT
    ExamplePluginHandle.cpp
    ExamplePluginContainer.cpp
    ExamplePluginPublic.cpp
    engines/ExamplePluginEngine.cpp
    engines/plans/ReluPlanBuilder.cpp
    engines/plans/ReluPlan.cpp
    engines/plans/ConvFwdPlanBuilder.cpp
    engines/plans/ConvFwdPlan.cpp
    hip/HipCompiledProgram.cpp
    hip/HipRunnableKernel.cpp
    ${KERNEL_GENERATED_SOURCES}
)

add_dependencies(example_plugin_impl example_plugin_kernel_embed)

target_include_directories(example_plugin_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(example_plugin_impl PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../kernels)
target_link_libraries(example_plugin_impl PUBLIC hipdnn_plugin_sdk hipdnn_data_sdk hip::host hiprtc::hiprtc)

set_target_properties(example_plugin_impl PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

# =============================================================================
# Static library: exposes plugin internals for unit tests to link against.
# This is not installed or distributed -- it exists only for testability.
# =============================================================================
add_library(example_plugin_private STATIC $<TARGET_OBJECTS:example_plugin_impl>)
target_include_directories(example_plugin_private PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(example_plugin_private PUBLIC hipdnn_plugin_sdk hipdnn_data_sdk example_plugin_impl)

# =============================================================================
# Shared library: the actual plugin loaded at runtime by hipDNN.
# Visibility is set to hidden so only the C entry points are exported.
# =============================================================================
add_library(example_plugin SHARED $<TARGET_OBJECTS:example_plugin_impl>)
target_link_libraries(example_plugin PRIVATE hipdnn_plugin_sdk hipdnn_data_sdk hip::host hiprtc::hiprtc)

if(NOT WIN32)
    target_link_libraries(example_plugin PRIVATE "-Xlinker --exclude-libs=ALL")
endif()

set_target_properties(example_plugin PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    INSTALL_RPATH "${ROCM_PATH}/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
