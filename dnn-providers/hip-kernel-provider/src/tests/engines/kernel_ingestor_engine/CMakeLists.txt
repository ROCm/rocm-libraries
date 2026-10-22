# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/TestHandleDeviceResolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIngestorKernelCode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestKernelIngestorEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestLaunchKernelDefinition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwiseAddDispatchHandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwiseAddMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestConvFwdPack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseSignature.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseDispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx1151WmmaAttentionMatchers.cpp
)

# The census suites, which live in the binary that is never invoked bare because each
# needs a shard and an architecture supplied by the entry that runs it. TestConvFwdPack
# stays above: it reads across the unit and unit_shared shards, so no single-shard entry
# can carry it and it states its inventory through the ordinary host run instead.
target_sources(hip_kernel_provider_census_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwisePacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDensePacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseCatalogTiles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx1151WmmaAttentionPacks.cpp
)

# FlyDSL's suites ride FlyDSL's gate, not the ingestor's: they read a shard only FlyDSL
# stages, and the cases that do it call loadedSet(), which throws rather than skipping when
# no shard is there. The sources above are the generic ingestor suites and stay ungated.
#
# Split across the two binaries on the same rule as the packs above: the census over the
# staged shard goes to the census binary, and the seam suites -- matchers, scoring,
# dispatch, the real launches -- stay in the ordinary one.
if(HIPKERNELPROVIDER_FLYDSL_ACTIVE)
    target_sources(hip_kernel_provider_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslRmsNormEngine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslRmsNormBwdEngine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslRmsNormObjects.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaRdnaEngine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaCdnaEngine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaObjects.cpp
    )
    # The architectures each op ships objects for, so a device case that finds none
    # staged on one of them fails instead of skipping (FlydslNormDeviceTestSupport.hpp).
    # Every op's list, so the SDPA suites and TestContainer can say which engines this
    # build must serve on the device too.
    foreach(_flydsl_op rmsnorm rmsnorm_bwd sdpa_cdna sdpa_cdna_dualwave sdpa_cdna_decode
                       sdpa_rdna sdpa_rdna_decode sdpa_aiw)
        string(REPLACE ";" "," _flydsl_archs "${HIPKERNELPROVIDER_FLYDSL_ARCHS_${_flydsl_op}}")
        string(TOUPPER "${_flydsl_op}" _flydsl_op_upper)
        target_compile_definitions(hip_kernel_provider_tests PRIVATE
            HIPKERNELPROVIDER_FLYDSL_ARCHS_${_flydsl_op_upper}="${_flydsl_archs}")
    endforeach()
    target_sources(hip_kernel_provider_census_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslRmsNormPacks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslRmsNormBwdPacks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaRdnaPacks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaRdnaDecodePacks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaAiwPacks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaCdnaPacks.cpp
    )
endif()
