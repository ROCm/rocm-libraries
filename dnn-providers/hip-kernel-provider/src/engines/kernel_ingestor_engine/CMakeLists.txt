# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Embed the kernel sources this engine's packs compile at plan-build time.
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/KernelEmbedding.cmake")

add_kernels_for_embedding(
    TARGET hip_kernel_provider_impl FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/PointwiseAdd.cpp
)

target_sources(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/KernelIngestorEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/PointwiseAddDispatchHandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/PointwiseAddMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/PointwiseAddPack.cpp
)

# The sources above are entirely inside `#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR`, so the
# definition must reach the object library that compiles them, not only the static
# library the tests link. This mirrors how each other engine sets its own gate.
target_compile_definitions(hip_kernel_provider_impl PUBLIC HIPDNN_ENABLE_KERNEL_INGESTOR=1)
