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

# Gated on the production root's offer, not GPU_TARGETS: the suites read a shard only the
# Triton family stages (loadedSet() throws without one), and the device cases skip when
# this device has no shard. The census suite goes to the census binary, registered per
# arch in ../../CMakeLists.txt; the matcher and dispatch suites go to the ordinary one.
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
