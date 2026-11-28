# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/TestCodeObjectTarget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestHandleDeviceResolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIngestorKernelCode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIngestorPlanRestoreChecks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestKernelIngestorEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestLaunchKernelDefinition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestMultiLaunchEndToEnd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestRankingTraceSupport.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestUnpulledModelsRecord.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestRestoredIngestorPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwiseAddDispatchHandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwiseAddMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestStandalonePointwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestConvFwdPack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestAttentionDensePersistent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestAttentionDenseGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestAttentionDenseSignature.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestAttentionDenseDispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestAttentionDenseMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionUnified3dGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionUnified3dSplitRule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionUnified3dAdmission.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionUnified3dMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionUnified3dScorer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionUnified3dSignature.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionUnified3dArguments.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionUnified2DMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionUnified2DGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionUnified2DSignature.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionUnified2DDispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionUnified3DMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionUnified3DGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionUnified3DSignature.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionUnified3DDispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionUnified2DMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionUnified2DGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionUnified2DSignature.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionUnified2DDispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestUnifiedAttentionGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950ConvFwdMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950ConvFwdGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950ConvFwdAotSelection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionDenseMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx1151WmmaAttentionMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaBwdMatcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950ConvDgradGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950ConvDgradMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx1250ConvDgradGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx1250ConvDgradMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx1151WmmaAttentionPaged.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestIngestorDispatchAliases.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPackLaunchValues.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/TestIngestorPlanHeader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/TestIngestorPlanCodec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/TestIngestorPlanLaunchContractGoldens.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/TestSerializableSourceKind.cpp
)

# The gfx942 unified 3D host tests check the engine's C++ restatements of rocKE rules
# (admission, split rule, kernel arguments) against tables rocKE itself
# answers. Each table is generated here, in the build tree, from the rocKE source tree and
# is never committed, so it cannot drift from rocKE. The generators need only the standard
# library and rocKE's Python, so this runs whether or not the rocKE engine is built. The
# depfile lists every rocKE module a generator imported: changing any of them regenerates
# its table, and the test that includes it then checks the C++ against the new answers.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(_gfx942_u3d_packs "${CMAKE_CURRENT_SOURCE_DIR}/packs")
set(_gfx942_u3d_generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(_gfx942_u3d_tables)
foreach(_table admission:Admission split_rule:SplitRule arguments:Arguments)
    string(REPLACE ":" ";" _table "${_table}")
    list(GET _table 0 _script)
    list(GET _table 1 _name)
    set(_inc "${_gfx942_u3d_generated}/Gfx942AttentionUnified3d${_name}Table.inc")
    add_custom_command(
        OUTPUT "${_inc}"
        COMMAND "${Python3_EXECUTABLE}" -I
                "${_gfx942_u3d_packs}/gen_gfx942_attention_unified_3d_table.py"
                "${_gfx942_u3d_packs}/gen_gfx942_attention_unified_3d_${_script}.py"
                --out "${_inc}" --depfile "${_inc}.d"
        DEPENDS "${_gfx942_u3d_packs}/gen_gfx942_attention_unified_3d_table.py"
                "${_gfx942_u3d_packs}/gen_gfx942_attention_unified_3d_${_script}.py"
        DEPFILE "${_inc}.d"
        COMMENT "Generating the gfx942 unified 3D ${_script} table from rocKE"
        VERBATIM)
    list(APPEND _gfx942_u3d_tables "${_inc}")
endforeach()
# hip_kernel_provider_tests is created in the parent directory, so the custom commands
# above reach it through this target rather than as its sources.
add_custom_target(hkp_gfx942_unified_3d_test_tables DEPENDS ${_gfx942_u3d_tables}
                  COMMENT "gfx942 unified 3D test tables generated from rocKE")
add_dependencies(hip_kernel_provider_tests hkp_gfx942_unified_3d_test_tables)
target_include_directories(hip_kernel_provider_tests PRIVATE "${_gfx942_u3d_generated}")

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
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestStandalonePointwisePacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwisePacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDensePacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950ConvFwdAotPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950ConvFwdAotSeedPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionDenseCatalogTiles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionUnified3dPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionUnified3dModelBinding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionUnified2DPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionUnified2DCatalogTiles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionUnified2DPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionUnified2DCatalogTiles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionUnified3DPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950AttentionUnified3DCatalogTiles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionDensePacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionDenseCatalogTiles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx942AttentionDenseModelBinding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx1151WmmaAttentionPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx1151WmmaAttentionCatalog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx1151WmmaAttentionDecodePacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx950ConvDgradAotPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx1250ConvDgradAotPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestGfx1250ConvDgradAotModelBinding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestShippedEnginePredictors.cpp
)

# Every (engine, architecture key) the source tree binds an L1 model under, read from the
# UED role maps, so TestShippedEnginePredictors can name the bindings a build left out
# (an engine family that is off) instead of passing silently over them. Each records
# whether a bound model is trained (any adapter but native): a native estimate is compiled
# code with no selector revision to hold it to.
file(GLOB_RECURSE _hkp_source_ueds CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../engines/kernel_ingestor_engine/descriptors/*.ued.json")
set(_hkp_l1_bindings "")
set(_hkp_l1_binding_count 0)
foreach(_hkp_ued IN LISTS _hkp_source_ueds)
    file(READ "${_hkp_ued}" _hkp_ued_text)
    string(JSON _hkp_l1_keys ERROR_VARIABLE _hkp_no_l1 LENGTH "${_hkp_ued_text}" predict_engine)
    if(_hkp_no_l1 OR _hkp_l1_keys EQUAL 0)
        continue()
    endif()
    string(JSON _hkp_engine GET "${_hkp_ued_text}" name)

    # The adapter of every UHD in the bundle, by lowercase id.
    cmake_path(GET _hkp_ued PARENT_PATH _hkp_bundle)
    file(GLOB_RECURSE _hkp_uhds CONFIGURE_DEPENDS "${_hkp_bundle}/*.uhd.json")
    set(_hkp_native_ids "")
    foreach(_hkp_uhd IN LISTS _hkp_uhds)
        file(READ "${_hkp_uhd}" _hkp_uhd_text)
        string(JSON _hkp_uhd_id ERROR_VARIABLE _hkp_no_id GET "${_hkp_uhd_text}" id)
        string(JSON _hkp_adapter ERROR_VARIABLE _hkp_no_adapter GET "${_hkp_uhd_text}" adapter)
        if(NOT _hkp_no_id AND NOT _hkp_no_adapter AND _hkp_adapter STREQUAL "native")
            string(TOLOWER "${_hkp_uhd_id}" _hkp_uhd_id)
            list(APPEND _hkp_native_ids "${_hkp_uhd_id}")
        endif()
    endforeach()

    math(EXPR _hkp_last "${_hkp_l1_keys} - 1")
    # cmake-lint: disable=E1120
    foreach(_hkp_index RANGE ${_hkp_last})
        string(JSON _hkp_key MEMBER "${_hkp_ued_text}" predict_engine ${_hkp_index})
        string(JSON _hkp_kind TYPE "${_hkp_ued_text}" predict_engine "${_hkp_key}")
        set(_hkp_ids "")
        if(_hkp_kind STREQUAL "STRING")
            string(JSON _hkp_ids GET "${_hkp_ued_text}" predict_engine "${_hkp_key}")
        else()
            string(JSON _hkp_id_count LENGTH "${_hkp_ued_text}" predict_engine "${_hkp_key}")
            math(EXPR _hkp_id_last "${_hkp_id_count} - 1")
            # cmake-lint: disable=E1120
            foreach(_hkp_id_index RANGE ${_hkp_id_last})
                string(JSON _hkp_id GET "${_hkp_ued_text}" predict_engine "${_hkp_key}"
                       ${_hkp_id_index})
                list(APPEND _hkp_ids "${_hkp_id}")
            endforeach()
        endif()
        # Trained unless every bound id names a native UHD of the bundle.
        set(_hkp_trained false)
        foreach(_hkp_id IN LISTS _hkp_ids)
            string(TOLOWER "${_hkp_id}" _hkp_id)
            if(NOT _hkp_id IN_LIST _hkp_native_ids)
                set(_hkp_trained true)
            endif()
        endforeach()
        string(APPEND _hkp_l1_bindings
            "    Binding{\"${_hkp_engine}\", \"${_hkp_key}\", ${_hkp_trained}},\n")
        math(EXPR _hkp_l1_binding_count "${_hkp_l1_binding_count} + 1")
    endforeach()
endforeach()
string(CONCAT _hkp_l1_header
    "// Generated by ${CMAKE_CURRENT_LIST_FILE}. Do not edit.\n"
    "#pragma once\n\n"
    "#include <array>\n"
    "#include <string_view>\n\n"
    "namespace shipped_engine_predictor_sources\n{\n\n"
    "struct Binding\n{\n    std::string_view engine;\n    std::string_view archKey;\n"
    "    bool trained;\n};\n\n"
    "inline constexpr std::array<Binding, ${_hkp_l1_binding_count}> BINDINGS{{\n"
    "${_hkp_l1_bindings}}};\n\n"
    "} // namespace shipped_engine_predictor_sources\n")
# Rewritten only when the bindings change, so a reconfigure recompiles nothing.
set(_hkp_l1_header_path "${CMAKE_CURRENT_BINARY_DIR}/ShippedEnginePredictorSources.hpp")
set(_hkp_l1_previous "")
if(EXISTS "${_hkp_l1_header_path}")
    file(READ "${_hkp_l1_header_path}" _hkp_l1_previous)
endif()
if(NOT _hkp_l1_previous STREQUAL _hkp_l1_header)
    file(WRITE "${_hkp_l1_header_path}" "${_hkp_l1_header}")
endif()
target_include_directories(hip_kernel_provider_census_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

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
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaBwdEngine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaBwdSplitEngine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaBwdObjects.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaBwdRemovedKeys.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaBwdParity.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaBwdObjectsPerObject.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaBwdPhiloxOracle.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaBwdPhiloxDevice.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaObjects.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaDropout.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaHeadPad.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaAiwClasses.cpp
    )
    # The architectures each op has objects for, so a device case that finds none
    # staged on one of them fails instead of skipping (FlydslNormDeviceTestSupport.hpp).
    # The content list, not what this build packs: both test binaries are host artifacts that
    # a multi-arch build (TheRock RFC 0008) compiles in one architecture's shard and runs on
    # every other, so the tests ask at run time whether the device's shard is staged.
    foreach(_flydsl_op rmsnorm rmsnorm_bwd sdpa_cdna sdpa_cdna_dualwave sdpa_cdna_decode
                       sdpa_rdna sdpa_rdna_decode sdpa_aiw sdpa_dropout_mask
                       sdpa_head_pad sdpa_bwd_merged sdpa_bwd_prep sdpa_bwd_dkdv
                       sdpa_bwd_dq)
        string(REPLACE ";" "," _flydsl_archs
                       "${HIPKERNELPROVIDER_FLYDSL_CONTENT_ARCHS_${_flydsl_op}}")
        string(TOUPPER "${_flydsl_op}" _flydsl_op_upper)
        foreach(_hkp_target hip_kernel_provider_tests hip_kernel_provider_census_tests)
            target_compile_definitions(${_hkp_target} PRIVATE
                HIPKERNELPROVIDER_FLYDSL_ARCHS_${_flydsl_op_upper}="${_flydsl_archs}")
        endforeach()
    endforeach()
    # The gfx11, gfx90a and gfx942 verdict tables and the manifests whose hashes they name, staged beside the
    # descriptor trees so a relocated or staged build tree carries them: the split SDPA
    # backward suites (census and seam) resolve them from the executable's own directory like
    # the descriptor roots. configure_file re-copies when a source changes.
    set(HKP_FLYDSL_SDPA_BWD_DATA_RELDIR "${HIPKERNELPROVIDER_TEST_DESCRIPTOR_RELDIR}/flydsl_sdpa_bwd")
    set(_hkp_bwd_data "${HIPKERNELPROVIDER_TEST_DESCRIPTOR_BUILD_DIR}/flydsl_sdpa_bwd")
    set(_hkp_provider_root "${HKP_INGESTOR_PLAN_SOURCE_DIR}/../../..")
    configure_file("${_hkp_provider_root}/flydsl/tests/data/sdpa_bwd_probe_graphs.csv"
                   "${_hkp_bwd_data}/sdpa_bwd_probe_graphs.csv" COPYONLY)
    # Each entry is <verdict table arch>:<content directory the table's hashes name>.
    foreach(_hkp_bwd_table gfx11:gfx11-generic gfx90a:gfx90a gfx942:gfx942)
        string(REPLACE ":" ";" _hkp_bwd_table_parts "${_hkp_bwd_table}")
        list(GET _hkp_bwd_table_parts 0 _hkp_bwd_table_arch)
        list(GET _hkp_bwd_table_parts 1 _hkp_bwd_table_content)
        configure_file("${_hkp_provider_root}/flydsl/tests/data/sdpa_bwd_${_hkp_bwd_table_arch}_verdicts.csv"
                       "${_hkp_bwd_data}/sdpa_bwd_${_hkp_bwd_table_arch}_verdicts.csv" COPYONLY)
        foreach(_hkp_bwd_op sdpa_bwd_prep sdpa_bwd_dkdv sdpa_bwd_dq)
            set(_hkp_bwd_manifest
                "${HKP_INGESTOR_PLAN_SOURCE_DIR}/descriptors/FlyDSL/${_hkp_bwd_op}/${_hkp_bwd_table_content}/manifest.json")
            if(EXISTS "${_hkp_bwd_manifest}")
                configure_file("${_hkp_bwd_manifest}"
                               "${_hkp_bwd_data}/${_hkp_bwd_op}/${_hkp_bwd_table_content}/manifest.json" COPYONLY)
            endif()
        endforeach()
    endforeach()
    foreach(_hkp_verdict_target hip_kernel_provider_tests hip_kernel_provider_census_tests)
        target_compile_definitions(${_hkp_verdict_target} PRIVATE
            HKP_FLYDSL_SDPA_BWD_DATA_RELDIR="${HKP_FLYDSL_SDPA_BWD_DATA_RELDIR}")
    endforeach()
    target_sources(hip_kernel_provider_census_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslRmsNormPacks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslRmsNormBwdPacks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaRdnaPacks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaRdnaDecodePacks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaAiwPacks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaCdnaPacks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaBwdPacks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslSdpaBwdSplitPacks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslModelBinding.cpp
    )
endif()
