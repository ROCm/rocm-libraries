# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# ============================================================================
# AOT catalog engine
# ============================================================================
# Throwaway path that loads loose rocKE-authored .co/HSACO kernels described by
# data-only family.json files. Kernel authors add kernels by dropping a .co plus
# editing JSON. See docs/CATALOG_FORMAT.md.

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
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/SdpaAdapter.cpp
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
# The catalog is placed BESIDE the plugin .so, at <plugin-dir>/${_AOT_CATALOG_RELDIR},
# in BOTH the build tree and the install tree, so the runtime can self-locate it
# relative to the loaded plugin binary (see catalog/ModulePath.cpp + defaultCatalogDir):
#   * build:   ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}/hip_kernel_provider/aot_catalog
#   * install: <prefix>/${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/aot_catalog
# A single relative offset (${_AOT_CATALOG_RELDIR}) therefore works for both layouts.
# This is what lets a locally-built / force-loaded plugin read its OWN build-tree
# catalog instead of silently picking up a system install's. Override at runtime with
# the HIPDNN_AOT_CATALOG_DIR environment variable.
set(_AOT_CATALOG_RELDIR "hip_kernel_provider/aot_catalog")
set(AOT_CATALOG_BUILD_DIR "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}/${_AOT_CATALOG_RELDIR}")
target_compile_definitions(hip_kernel_provider_impl PRIVATE
    # Relative offset from the loaded plugin dir to the catalog (self-location).
    HIPDNN_AOT_CATALOG_RELDIR="${_AOT_CATALOG_RELDIR}"
    # Absolute install path -- last-resort fallback only, when self-location fails.
    HIPDNN_AOT_CATALOG_DIR="${CMAKE_INSTALL_PREFIX}/${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/${_AOT_CATALOG_RELDIR}"
)

# dladdr (Linux) needs libdl on older glibc; ${CMAKE_DL_LIBS} is empty where it is
# already folded into libc, and on Windows. Propagate from the object lib to the plugin.
target_link_libraries(hip_kernel_provider_impl PUBLIC ${CMAKE_DL_LIBS})

# The family library: each library/<arch>/<family>/ is a self-contained unit
# (family.json + CMakeLists.txt [+ co-located producer]). rocKE-AOT families
# COMPILE their .co at build time into ${AOT_CATALOG_BUILD_DIR}/<arch>/<family>/;
# prebuilt families would stage checked-in .co. rocKE is an OPTIONAL per-family
# dependency (see library/CMakeLists.txt): with no rocKE the families skip, the
# catalog is empty, and the engine declines every graph (GPU parity tests skip).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/library/CMakeLists.txt)
    add_subdirectory(library)
endif()
