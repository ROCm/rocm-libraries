# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/TestCodeObjectTarget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestHandleDeviceResolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIngestorKernelCode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIngestorPlanRestoreChecks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestKernelIngestorEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestLaunchKernelDefinition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestRestoredIngestorPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwiseAddDispatchHandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwiseAddMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestConvFwdPack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseSignature.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseDispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx1151WmmaAttentionMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaBwdMatcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestIngestorDispatchAliases.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPackLaunchValues.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/TestIngestorPlanHeader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/TestIngestorPlanCodec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/TestIngestorPlanLaunchContractGoldens.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/TestSerializableSourceKind.cpp
)

# The schema hash test compares the pinned constant with the hash of the schema file. Line
# ends are normalised to LF, so a CRLF checkout gives the same hash.
set(HKP_INGESTOR_PLAN_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../engines/kernel_ingestor_engine")
set(HKP_INGESTOR_PLAN_SCHEMA "${HKP_INGESTOR_PLAN_SOURCE_DIR}/serialization/ingestor_plan.fbs")
file(READ "${HKP_INGESTOR_PLAN_SCHEMA}" HKP_INGESTOR_PLAN_SCHEMA_TEXT)
string(REPLACE "\r\n" "\n" HKP_INGESTOR_PLAN_SCHEMA_TEXT "${HKP_INGESTOR_PLAN_SCHEMA_TEXT}")
string(SHA256 HKP_INGESTOR_PLAN_SCHEMA_HASH "${HKP_INGESTOR_PLAN_SCHEMA_TEXT}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${HKP_INGESTOR_PLAN_SCHEMA}")
set_property(
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/serialization/TestIngestorPlanHeader.cpp
    TARGET_DIRECTORY hip_kernel_provider_tests
    APPEND PROPERTY COMPILE_DEFINITIONS
    "HKP_INGESTOR_PLAN_SCHEMA_SHA256=\"${HKP_INGESTOR_PLAN_SCHEMA_HASH}\"")

# The census suites, which live in the binary that is never invoked bare because each
# needs a shard and an architecture supplied by the entry that runs it. TestConvFwdPack
# stays above: it reads across the unit and unit_shared shards, so no single-shard entry
# can carry it and it states its inventory through the ordinary host run instead.
target_sources(hip_kernel_provider_census_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwisePacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDensePacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseCatalogTiles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx1151WmmaAttentionPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx1151WmmaAttentionDecodePacks.cpp
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
    # The architectures each op has objects for, so a device case that finds none
    # staged on one of them fails instead of skipping (FlydslNormDeviceTestSupport.hpp).
    # The content list, not what this build packs: this binary is a host artifact that a
    # multi-arch build (TheRock RFC 0008) compiles in one architecture's shard and runs on
    # every other, so the test asks at run time whether the device's shard is staged.
    foreach(_flydsl_op rmsnorm rmsnorm_bwd sdpa_cdna sdpa_cdna_dualwave sdpa_cdna_decode
                       sdpa_rdna sdpa_rdna_decode sdpa_aiw)
        string(REPLACE ";" "," _flydsl_archs
                       "${HIPKERNELPROVIDER_FLYDSL_CONTENT_ARCHS_${_flydsl_op}}")
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
