# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/TestHandleDeviceResolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIngestorKernelCode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestKernelIngestorEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwiseAddDispatchHandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwiseAddMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestConvFwdPack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseSignature.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseDispatch.cpp
)

# The census suites, which live in the binary that is never invoked bare because each
# needs a shard and an architecture supplied by the entry that runs it. TestConvFwdPack
# stays above: it reads across the unit and unit_shared shards, so no single-shard entry
# can carry it and it states its inventory through the ordinary host run instead.
target_sources(hip_kernel_provider_census_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwisePacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDensePacks.cpp
)

# The Triton RMSNorm suites are gated on the production root's offer: they read a shard
# only the Triton family stages, and the census cases call loadedSet(), which throws
# rather than skipping when no shard is there. Arch-agnostic: the sources compile for any
# GPU_TARGETS and the device cases skip when this device has no shard, so this uses
# hkp_product_offers_engine(), not the arch-aware hkp_triton_rmsnorm_available().
#
# Split across the two binaries like the packs above: the census goes to the census
# binary, the matcher, dispatch and launch suites to the ordinary one. The per-arch gate
# is on the census registration in ../../CMakeLists.txt.
hkp_product_offers_engine(_hkp_triton_rmsnorm_sources "hipkernel:TritonRmsNorm"
                          HIPKERNELPROVIDER_ENABLE_TRITON)
if(_hkp_triton_rmsnorm_sources)
    target_sources(hip_kernel_provider_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestTritonRmsNormMatchers.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestTritonRmsNormDispatch.cpp
    )
    target_sources(hip_kernel_provider_census_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestTritonRmsNormPacks.cpp
    )
endif()
