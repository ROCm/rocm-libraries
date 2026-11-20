# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# The SDPA determinism suite. The host self-check of its harness is compiled in every build:
# it needs no device, no SDPA and no ingestor.
target_sources(hip_kernel_provider_integration_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationSdpaDeterminismHarness.cpp
)

# The device cases of each engine group. A group's source compiles only when hipDNN was built
# with the SDPA frontend and the ingestor (the frontend graph builders the harness reuses are
# guarded the same way), and when the group has a gate of its own: the engine is built, the
# product root offers it, or FlyDSL ships its objects. Each gate sets a compile definition that
# selects the engine's table in the source; a source holds a table only under its definition,
# and compiles to nothing without any of them.
function(_hkp_determinism_add_group source)
    cmake_parse_arguments(ARG "" "" "DEFINITIONS" ${ARGN})
    if(NOT ARG_DEFINITIONS)
        return()
    endif()
    target_sources(hip_kernel_provider_integration_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/${source}
    )
    target_compile_definitions(hip_kernel_provider_integration_tests PRIVATE ${ARG_DEFINITIONS})
endfunction()

if(HIPDNN_ENABLE_KERNEL_INGESTOR AND NOT HIPDNN_ENABLE_SDPA)
    message(WARNING
            "HIPDNN_ENABLE_KERNEL_INGESTOR is ON but hipDNN was built with "
            "HIPDNN_ENABLE_SDPA=OFF. Skipping the SDPA determinism device cases. Rebuild hipDNN "
            "with -DHIPDNN_ENABLE_SDPA=ON to enable them.")
endif()

if(HIPDNN_ENABLE_KERNEL_INGESTOR AND HIPDNN_ENABLE_SDPA)
    # The prebuilt-object and HIP engines: built by their own switches.
    set(_hkp_determinism_definitions "")
    if(ENABLE_ASM_SDPA_ENGINE)
        list(APPEND _hkp_determinism_definitions HKP_DETERMINISM_ASM_SDPA=1)
    endif()
    if(ENABLE_HIP_FLASH2_ENGINE)
        list(APPEND _hkp_determinism_definitions HKP_DETERMINISM_HIP_FLASH2=1)
    endif()
    _hkp_determinism_add_group(IntegrationGpuSdpaDeterminismAsmFlash2.cpp
        DEFINITIONS ${_hkp_determinism_definitions})

    # The rocKE and WMMA engines: compiled whenever the product root offers the engine, on any
    # build arch, and skipped on other devices by the row's arch list (the rule of the attention
    # knob suites in ../CMakeLists.txt).
    set(_hkp_determinism_rocke
        Gfx950AttentionDense GFX950_DENSE
        Gfx942AttentionDense GFX942_DENSE
        Gfx942AttentionUnified2D GFX942_UNIFIED_2D
        Gfx942AttentionUnified3d GFX942_UNIFIED_3D
        Gfx950AttentionUnified2D GFX950_UNIFIED_2D
        Gfx950AttentionUnified3D GFX950_UNIFIED_3D
    )
    set(_hkp_determinism_definitions "")
    while(_hkp_determinism_rocke)
        list(POP_FRONT _hkp_determinism_rocke _engine _token)
        hkp_product_offers_engine(_offered "hipkernel:${_engine}")
        if(_offered)
            list(APPEND _hkp_determinism_definitions HKP_DETERMINISM_ROCKE_${_token}=1)
        endif()
    endwhile()
    _hkp_determinism_add_group(IntegrationGpuSdpaDeterminismRocke.cpp
        DEFINITIONS ${_hkp_determinism_definitions})

    set(_hkp_determinism_wmma
        Gfx1151WmmaAttention WMMA_PREFILL
        Gfx1151WmmaAttentionDecode WMMA_DECODE
    )
    set(_hkp_determinism_definitions "")
    while(_hkp_determinism_wmma)
        list(POP_FRONT _hkp_determinism_wmma _engine _token)
        hkp_product_offers_engine(_offered "hipkernel:${_engine}")
        if(_offered)
            list(APPEND _hkp_determinism_definitions HKP_DETERMINISM_${_token}=1)
        endif()
    endwhile()
    _hkp_determinism_add_group(IntegrationGpuSdpaDeterminismWmma.cpp
        DEFINITIONS ${_hkp_determinism_definitions})

    # The FlyDSL engines: the existing frontend lane's gate (../CMakeLists.txt) plus FlyDSL
    # content for every op the engine serves, so a build whose product root ships no content
    # for an op compiles no row of it.
    set(_hkp_determinism_flydsl_fwd
        sdpa_rdna RDNA
        sdpa_rdna_decode RDNA_DECODE
        sdpa_aiw AIW
        sdpa_cdna CDNA
        sdpa_cdna_decode CDNA_DECODE
        sdpa_cdna_dualwave CDNA_DUALWAVE
    )
    set(_hkp_determinism_flydsl_fwd_definitions "")
    set(_hkp_determinism_flydsl_bwd_definitions "")
    if(HIPKERNELPROVIDER_ENABLE_FLYDSL AND HIPKERNELPROVIDER_FLYDSL_ACTIVE)
        while(_hkp_determinism_flydsl_fwd)
            list(POP_FRONT _hkp_determinism_flydsl_fwd _op _token)
            if(HIPKERNELPROVIDER_FLYDSL_CONTENT_ARCHS_${_op})
                list(APPEND _hkp_determinism_flydsl_fwd_definitions
                    HKP_DETERMINISM_FLYDSL_SDPA_${_token}=1)
            endif()
        endwhile()
        if(HIPKERNELPROVIDER_FLYDSL_CONTENT_ARCHS_sdpa_bwd_merged)
            list(APPEND _hkp_determinism_flydsl_bwd_definitions
                HKP_DETERMINISM_FLYDSL_BWD_MERGED=1)
        endif()
        # The split engine is three ops: prep, dkdv and dq.
        if(HIPKERNELPROVIDER_FLYDSL_CONTENT_ARCHS_sdpa_bwd_prep
           AND HIPKERNELPROVIDER_FLYDSL_CONTENT_ARCHS_sdpa_bwd_dkdv
           AND HIPKERNELPROVIDER_FLYDSL_CONTENT_ARCHS_sdpa_bwd_dq)
            list(APPEND _hkp_determinism_flydsl_bwd_definitions
                HKP_DETERMINISM_FLYDSL_BWD_SPLIT=1)
        endif()
    endif()
    _hkp_determinism_add_group(IntegrationGpuSdpaDeterminismFlydslFwd.cpp
        DEFINITIONS ${_hkp_determinism_flydsl_fwd_definitions})
    _hkp_determinism_add_group(IntegrationGpuSdpaDeterminismFlydslBwd.cpp
        DEFINITIONS ${_hkp_determinism_flydsl_bwd_definitions})
endif()
