# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# The plugin resolves kpack descriptors, which read their kernel code from an archive on
# disk, so nothing is embedded there. The unit-test binary resolves both kinds: it reads
# kpack descriptors from a staged archive, and for embedded_source descriptors it links
# this engine statically and embeds its own copy of the kernel sources.

target_sources(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/IngestorPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/KernelIngestorEngine.cpp
    # Descriptors naming these packs are not in this tree. They arrive from the install tree:
    # test_descriptors/ for the test binaries, and descriptor-packaging/examples/descriptors
    # where the superbuild CI lane points the production source root.
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/PointwiseNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvNative.cpp
)

# Sources above are entirely inside `#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR`; the
# definition must reach this object library, not only the static library tests link.
target_compile_definitions(hip_kernel_provider_impl PUBLIC HIPDNN_ENABLE_KERNEL_INGESTOR=1)
