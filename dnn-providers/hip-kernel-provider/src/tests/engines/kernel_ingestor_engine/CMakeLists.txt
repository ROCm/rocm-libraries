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
