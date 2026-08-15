# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# ============================================================================
# AOT catalog engine
# ============================================================================
# Throwaway path that loads loose rocKE-authored .co/HSACO kernels described by
# data-only family.json files. This engine only LOADS the catalog at runtime; see
# README.md for the runtime/loader contract. Kernel authoring (families, producers,
# the family.json schema) lives in rocke/aot_catalog/ -- see its README.md.

target_sources(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/CatalogEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/launch/LaunchAbi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/catalog/Selection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/catalog/Catalog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/catalog/ModulePath.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/catalog/AotDebug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/catalog/TuneCache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/GemmAdapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/RmsNormAdapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/LayerNormAdapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/ActivationAdapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/SdpaAdapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/ConvFpropAdapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/CatalogPlan.cpp
)

# So sources here can #include "catalog/...", "launch/...", "ops/...", "plans/..."
# relative to this engine's own root (mirrors asm_sdpa_engine).
target_include_directories(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(hip_kernel_provider_impl PUBLIC HIPDNN_ENGINE_AOT_CATALOG=1)

# Where the runtime looks for <arch>/<family>/family.json + .co files.
#
# The catalog is placed BESIDE the plugin .so so the runtime can self-locate it
# relative to the loaded plugin binary (see catalog/ModulePath.cpp + defaultCatalogDir).
# The path is defined ONCE in the top-level CMakeLists (HIPDNN_AOT_CATALOG_RELDIR /
# _INSTALL_DIR) -- the same constants the rocke aot_catalog PRODUCER uses to emit and
# install the catalog -- so producer and runtime can never drift. This engine only
# READS the catalog; it does NOT build kernels and has NO dependency on rocke.
# Override the catalog location at runtime with HIPDNN_AOT_CATALOG_DIR.
target_compile_definitions(hip_kernel_provider_impl PRIVATE
    # Relative offset from the loaded plugin dir to the catalog (self-location).
    HIPDNN_AOT_CATALOG_RELDIR="${HIPDNN_AOT_CATALOG_RELDIR}"
    # Absolute install path -- last-resort fallback only, when self-location fails.
    HIPDNN_AOT_CATALOG_DIR="${CMAKE_INSTALL_PREFIX}/${HIPDNN_AOT_CATALOG_INSTALL_DIR}"
)

# dladdr (Linux) needs libdl on older glibc; ${CMAKE_DL_LIBS} is empty where it is
# already folded into libc, and on Windows. Propagate from the object lib to the plugin.
target_link_libraries(hip_kernel_provider_impl PUBLIC ${CMAKE_DL_LIBS})

# NOTE: kernel authoring + the family build now live in rocke/aot_catalog/ (owned by
# the KA teams), NOT here. The families COMPILE their .co at build time into
# ${HIPDNN_AOT_CATALOG_BUILD_DIR}/<arch>/<family>/ and install beside the plugin;
# this engine loads whatever is there. With rocke off (or no families for the arch)
# the catalog is empty, the engine declines every graph, and GPU parity tests skip.
