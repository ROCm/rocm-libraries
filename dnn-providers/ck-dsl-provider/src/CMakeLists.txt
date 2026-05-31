# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Three-target shape (OBJECT -> STATIC for unit tests + SHARED for the
# loadable .so), mirroring miopen-provider and the SDK example. The
# OBJECT library carries the actual translation units so both downstream
# targets share one compile pass.

add_library(
    ck_dsl_provider_impl OBJECT
    CkDslContainer.cpp
    CkDslHandle.cpp
    CkDslPluginPublic.cpp
    adapters/conv_implicit_gemm/ConvImplicitGemmAdapter.cpp
    adapters/conv_implicit_gemm/ConvImplicitGemmPayload.cpp
    engines/conv_implicit_gemm/CkDslConvImplicitGemmEngine.cpp
    engines/conv_implicit_gemm/ConvImplicitGemmPlan.cpp
    engines/conv_implicit_gemm/ConvImplicitGemmPlanBuilder.cpp
    graph/GraphSignature.cpp
    perf/PerfMeasurement.cpp
    python/CompileServiceBridge.cpp
    python/EmbeddedInterpreter.cpp
    python/PythonError.cpp
    runtime/DeviceArch.cpp
    runtime/HipModule.cpp
    runtime/JitCache.cpp
    runtime/KernelArtifact.cpp
    runtime/LaunchAbi.cpp
)

target_include_directories(ck_dsl_provider_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    ck_dsl_provider_impl
    PUBLIC hipdnn_plugin_sdk hipdnn_data_sdk hipdnn_flatbuffers_sdk hip::host
           Python3::Python pybind11::embed
)

set_target_properties(ck_dsl_provider_impl PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

# Static library: exposes the plugin internals for future unit tests
# (kept around so the I-2/I-3 throwaway tests can link without going
# through the .so loader). Not installed.
add_library(ck_dsl_provider_private STATIC $<TARGET_OBJECTS:ck_dsl_provider_impl>)
target_include_directories(ck_dsl_provider_private PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    ck_dsl_provider_private
    PUBLIC hipdnn_plugin_sdk hipdnn_data_sdk hipdnn_flatbuffers_sdk ck_dsl_provider_impl
           Python3::Python pybind11::embed
)

# Shared library: the actual plugin loaded at runtime by hipDNN.
add_library(ck_dsl_provider_plugin SHARED $<TARGET_OBJECTS:ck_dsl_provider_impl>)
target_link_libraries(
    ck_dsl_provider_plugin
    PRIVATE hipdnn_plugin_sdk hipdnn_data_sdk hipdnn_flatbuffers_sdk hip::host
            Python3::Python pybind11::embed
)

if(NOT WIN32)
    target_link_libraries(ck_dsl_provider_plugin PRIVATE "-Xlinker --exclude-libs=ALL")
endif()

set_target_properties(ck_dsl_provider_plugin PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../.."
    INSTALL_RPATH_USE_LINK_PATH TRUE
    LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
)

# RPATH hint mirroring miopen-provider so TheRock can compute correct
# relative RPATHs when staging the install tree.
set_target_properties(ck_dsl_provider_plugin PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}")
