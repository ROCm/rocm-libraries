# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Embed the kernel sources this engine's packs compile at plan-build time. The list is
# shared with the test binary, which embeds its own copy of the same set.
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/KernelEmbedding.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/IngestorKernels.cmake")

foreach(kernel IN LISTS HIPDNN_INGESTOR_PACK_KERNELS)
    list(APPEND _ingestor_pack_kernel_sources "${HIPDNN_INGESTOR_PACK_KERNEL_DIR}/${kernel}.cpp")
endforeach()

add_kernels_for_embedding(
    TARGET hip_kernel_provider_impl
    FILES ${_ingestor_pack_kernel_sources}
)

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
