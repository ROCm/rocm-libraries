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
)

# The bridge + interpreter TUs include the MicroPython C API; they must compile
# against the generated embed package, so order this OBJECT library's compile
# after the static lib build and add its include dirs + frozen-module defines.
add_dependencies(ck_dsl_provider_impl ${CKDSL_MPY_BUILD_TARGET})
foreach(_dir IN LISTS CKDSL_MPY_INCLUDE_DIRS)
    target_include_directories(ck_dsl_provider_impl PUBLIC $<BUILD_INTERFACE:${_dir}>)
endforeach()
target_compile_definitions(ck_dsl_provider_impl
    PUBLIC MICROPY_MODULE_FROZEN_MPY=1 MICROPY_MODULE_FROZEN_STR=1
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
add_dependencies(ck_dsl_provider_private ${CKDSL_MPY_BUILD_TARGET})
# Whole-archive so executables linking this static lib (the unit + integration
# tests) also force-include the frozen-module table + native comgr module, just
# like the plugin .so does.
target_link_libraries(
    ck_dsl_provider_private
    PUBLIC hipdnn_plugin_sdk hipdnn_data_sdk hipdnn_flatbuffers_sdk ck_dsl_provider_impl
           "$<LINK_LIBRARY:WHOLE_ARCHIVE,${CKDSL_MPY_STATIC_LIB}>" ${CKDSL_MPY_COMGR_LIB} m
)

# Shared library: the actual plugin loaded at runtime by hipDNN.
add_library(ck_dsl_provider_plugin SHARED $<TARGET_OBJECTS:ck_dsl_provider_impl>)
add_dependencies(ck_dsl_provider_plugin ${CKDSL_MPY_BUILD_TARGET})
target_link_libraries(
    ck_dsl_provider_plugin
    PRIVATE hipdnn_plugin_sdk hipdnn_data_sdk hipdnn_flatbuffers_sdk hip::host
)

# Link the embedded-MicroPython archive whole: the frozen-module table and the
# native `comgr` module's MP_REGISTER_MODULE record are reached only via
# generated tables / the import machinery, so members must not be dropped by the
# archive's pull-on-demand resolution. --exclude-libs=ALL (below) still hides
# every one of these symbols from the plugin's dynamic export.
target_link_libraries(
    ck_dsl_provider_plugin
    PRIVATE -Wl,--whole-archive ${CKDSL_MPY_STATIC_LIB} -Wl,--no-whole-archive
            ${CKDSL_MPY_COMGR_LIB} m
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
