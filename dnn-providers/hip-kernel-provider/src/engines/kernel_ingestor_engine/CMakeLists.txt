# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# The plugin resolves kpack descriptors, which read their kernel code from an archive on
# disk. The unit-test binary resolves embedded_source descriptors. That binary links this
# engine statically and embeds its own copy of the kernel sources.

target_sources(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/IngestorPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/KernelIngestorEngine.cpp
    # One file per pack, the native seam; its descriptors are installed JSON. Listed
    # explicitly rather than globbed -- CMake's own docs discourage GLOB for sources,
    # and CONFIGURE_DEPENDS does not re-run on removal under every generator.
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/PointwiseNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvNative.cpp
)

# Sources above are entirely inside `#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR`; the
# definition must reach this object library, not only the static library tests link.
target_compile_definitions(hip_kernel_provider_impl PUBLIC HIPDNN_ENABLE_KERNEL_INGESTOR=1)
