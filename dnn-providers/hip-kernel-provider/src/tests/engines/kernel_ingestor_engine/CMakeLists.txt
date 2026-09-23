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

# The census suites. Each states one pack target's emitted inventory for one
# architecture, both supplied by the census entry that runs it, so they live in the
# binary that is never invoked bare. TestConvFwdPack stays above: it reads across the
# unit and unit_shared shards, so no single-shard entry can carry it and it states its
# inventory through the ordinary host run instead.
target_sources(hip_kernel_provider_census_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwisePacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDensePacks.cpp
)
