# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# The plugin resolves kpack descriptors, which read their kernel code from an archive on
# disk, so nothing is embedded there. The unit-test binary resolves both kinds: it reads
# kpack descriptors from a staged archive, and for embedded_source descriptors it links
# this engine statically and embeds its own copy of the kernel sources.

target_sources(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/IngestorPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/IngestorPlanCapture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/IngestorPlanRestore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/KernelIngestorEngine.cpp
    # Pointwise and Conv descriptors are test fixtures under test_descriptors/; a Pointwise
    # bundle also sits in descriptor-packaging/examples/descriptors, where the superbuild CI
    # lane points the production source root. Gfx950AttentionDense, Gfx950AttentionUnified2D,
    # Gfx950AttentionUnified3D, Gfx942AttentionDense, Gfx950ConvDgradAot and
    # Gfx1250ConvDgradAot ship from descriptors/; the two dense attention engines share
    # AttentionDenseNative.cpp, the two dgrad engines ConvDgradAotNative.cpp.
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/PointwiseNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/AttentionDenseNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/Gfx950AttentionDenseNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/Gfx942AttentionUnified3dNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/Gfx950AttentionUnified2DNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/Gfx950AttentionUnified3DNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/Gfx942AttentionDenseNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/Gfx1151WmmaAttentionNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvDgradAotNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/IngestorPlanHeader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/IngestorPlanCodec.cpp
)

# serialization/ingestor_plan_generated.h is committed, and the build compiles against it.
# When HIPKERNELPROVIDER_GENERATE_SERIALIZE_HEADERS is ON, the build regenerates it from the
# schema with the flags in serialization/flatc_flags.txt.
# The build finds flatc in this order: flatbuffers::flatc, flatc, the
# FLATBUFFERS_FLATC_EXECUTABLE variables, and then a search beside the FlatBuffers package.
# An imported target is visible in the directory that imports it and in its subdirectories.
# A standalone build imports FlatBuffers in the provider root.
# Thus flatbuffers::flatc is visible in this directory.
# A superbuild imports FlatBuffers in a sibling directory.
# Thus flatbuffers::flatc is not visible in this directory.
# With the option ON, a missing flatc is a configure error.
# With the option ON, a flatc version that is not the expected FlatBuffers version is a
# configure error.
block()
    set(HKP_INGESTOR_PLAN_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/serialization/ingestor_plan.fbs")
    set(HKP_INGESTOR_PLAN_HEADER
        "${CMAKE_CURRENT_SOURCE_DIR}/serialization/ingestor_plan_generated.h"
    )
    set(HKP_FLATC_FLAGS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/serialization/flatc_flags.txt")

    set(_hkp_flatc_command "")
    set(_hkp_flatc_dependency "")
    if(TARGET flatbuffers::flatc)
        set(_hkp_flatc_command "$<TARGET_FILE:flatbuffers::flatc>")
        set(_hkp_flatc_dependency flatbuffers::flatc)
    elseif(TARGET flatc)
        set(_hkp_flatc_command "$<TARGET_FILE:flatc>")
        set(_hkp_flatc_dependency flatc)
    else()
        foreach(_hkp_candidate IN ITEMS FLATBUFFERS_FLATC_EXECUTABLE FlatBuffers_FLATC_EXECUTABLE
                                        flatbuffers_FLATC_EXECUTABLE)
            if(DEFINED ${_hkp_candidate} AND NOT "${${_hkp_candidate}}" STREQUAL "")
                set(_hkp_flatc_command "${${_hkp_candidate}}")
                break()
            endif()
        endforeach()
        if(_hkp_flatc_command STREQUAL "")
            set(_hkp_flatc_hints "")
            if(DEFINED flatbuffers_DIR)
                get_filename_component(
                    _hkp_flatbuffers_prefix "${flatbuffers_DIR}/../../.." ABSOLUTE
                )
                list(APPEND _hkp_flatc_hints "${_hkp_flatbuffers_prefix}/bin")
            endif()
            find_program(_hkp_flatc_program NAMES flatc HINTS ${_hkp_flatc_hints} NO_CACHE)
            if(_hkp_flatc_program)
                set(_hkp_flatc_command "${_hkp_flatc_program}")
            endif()
        endif()
    endif()

    if(NOT HIPKERNELPROVIDER_GENERATE_SERIALIZE_HEADERS)
        message(STATUS
            "hip-kernel-provider: HIPKERNELPROVIDER_GENERATE_SERIALIZE_HEADERS is OFF, using the "
            "committed ${HKP_INGESTOR_PLAN_HEADER} as-is. Edits to "
            "${HKP_INGESTOR_PLAN_SCHEMA} will not reach the build."
        )
    else()
        if(_hkp_flatc_command STREQUAL "")
            message(FATAL_ERROR
                "hip-kernel-provider: HIPKERNELPROVIDER_GENERATE_SERIALIZE_HEADERS is ON but the "
                "FlatBuffers compiler (flatc) could not be found. Install flatc alongside the "
                "FlatBuffers package or set FLATBUFFERS_FLATC_EXECUTABLE. Without it "
                "${HKP_INGESTOR_PLAN_HEADER} cannot be regenerated and would go stale against "
                "${HKP_INGESTOR_PLAN_SCHEMA}. Set the option OFF to build against the committed "
                "header deliberately."
            )
        endif()
        # An in-tree flatc builds from the same source as the FlatBuffers library.
        set(_hkp_flatc_in_tree FALSE)
        if(NOT _hkp_flatc_dependency STREQUAL "")
            get_target_property(_hkp_flatc_imported ${_hkp_flatc_dependency} IMPORTED)
            if(NOT _hkp_flatc_imported)
                set(_hkp_flatc_in_tree TRUE)
            endif()
        endif()

        if(_hkp_flatc_in_tree)
            message(VERBOSE
                "hip-kernel-provider: flatc is built in this tree, so its version is not checked."
            )
        else()
            if(NOT _hkp_flatc_dependency STREQUAL "")
                get_target_property(_hkp_flatc_path ${_hkp_flatc_dependency} LOCATION)
            else()
                set(_hkp_flatc_path "${_hkp_flatc_command}")
            endif()

            if(DEFINED flatbuffers_VERSION)
                set(_hkp_expected_flatc_version "${flatbuffers_VERSION}")
                set(_hkp_expected_flatc_version_source flatbuffers_VERSION)
            elseif(DEFINED HIPDNN_FLATBUFFERS_VERSION)
                set(_hkp_expected_flatc_version "${HIPDNN_FLATBUFFERS_VERSION}")
                set(_hkp_expected_flatc_version_source HIPDNN_FLATBUFFERS_VERSION)
            else()
                message(FATAL_ERROR
                    "hip-kernel-provider: HIPKERNELPROVIDER_GENERATE_SERIALIZE_HEADERS is ON but "
                    "the expected FlatBuffers version could not be determined: flatbuffers_VERSION "
                    "and HIPDNN_FLATBUFFERS_VERSION are both undefined. The version of flatc "
                    "${_hkp_flatc_path} cannot be checked, so ${HKP_INGESTOR_PLAN_HEADER} could "
                    "be regenerated with a different FlatBuffers version. Set the option OFF to "
                    "build against the committed header deliberately."
                )
            endif()

            execute_process(
                COMMAND "${_hkp_flatc_path}" --version
                OUTPUT_VARIABLE _hkp_flatc_version_output
                ERROR_VARIABLE _hkp_flatc_version_error
                RESULT_VARIABLE _hkp_flatc_version_result
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_STRIP_TRAILING_WHITESPACE
            )
            set(_hkp_flatc_version "")
            if(_hkp_flatc_version_result EQUAL 0)
                if(_hkp_flatc_version_output MATCHES "([0-9]+\\.[0-9]+\\.[0-9]+)")
                    set(_hkp_flatc_version "${CMAKE_MATCH_1}")
                endif()
            endif()
            if(_hkp_flatc_version STREQUAL "")
                message(FATAL_ERROR
                    "hip-kernel-provider: HIPKERNELPROVIDER_GENERATE_SERIALIZE_HEADERS is ON but "
                    "flatc ${_hkp_flatc_path} did not report a version. The expected FlatBuffers "
                    "version is ${_hkp_expected_flatc_version}. flatc --version gave exit status "
                    "\"${_hkp_flatc_version_result}\", output \"${_hkp_flatc_version_output}\" and "
                    "error output \"${_hkp_flatc_version_error}\". Set the option OFF to build "
                    "against the committed header deliberately."
                )
            endif()
            if(NOT _hkp_flatc_version VERSION_EQUAL _hkp_expected_flatc_version)
                message(FATAL_ERROR
                    "hip-kernel-provider: HIPKERNELPROVIDER_GENERATE_SERIALIZE_HEADERS is ON but "
                    "flatc ${_hkp_flatc_path} reports FlatBuffers version ${_hkp_flatc_version}, "
                    "and the expected version is ${_hkp_expected_flatc_version} (from "
                    "${_hkp_expected_flatc_version_source}). A different flatc version can "
                    "generate a different ${HKP_INGESTOR_PLAN_HEADER}. Use the flatc of the "
                    "expected version, or set the option OFF to build against the committed "
                    "header deliberately."
                )
            endif()
            message(VERBOSE
                "hip-kernel-provider: flatc ${_hkp_flatc_path} reports FlatBuffers version "
                "${_hkp_flatc_version}; the expected version is ${_hkp_expected_flatc_version} "
                "(from ${_hkp_expected_flatc_version_source})."
            )
        endif()

        file(STRINGS "${HKP_FLATC_FLAGS_FILE}" _hkp_flatc_flag_lines)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${HKP_FLATC_FLAGS_FILE}")
        set(_hkp_flatc_flags "")
        foreach(_hkp_line IN LISTS _hkp_flatc_flag_lines)
            string(STRIP "${_hkp_line}" _hkp_stripped)
            if(_hkp_stripped STREQUAL "" OR _hkp_stripped MATCHES "^#")
                continue()
            endif()
            list(APPEND _hkp_flatc_flags "${_hkp_stripped}")
        endforeach()

        add_custom_command(
            OUTPUT "${HKP_INGESTOR_PLAN_HEADER}"
            COMMAND "${_hkp_flatc_command}" --cpp ${_hkp_flatc_flags}
                    -o "${CMAKE_CURRENT_SOURCE_DIR}/serialization"
                    "${HKP_INGESTOR_PLAN_SCHEMA}"
            DEPENDS "${HKP_INGESTOR_PLAN_SCHEMA}" "${HKP_FLATC_FLAGS_FILE}" ${_hkp_flatc_dependency}
            COMMENT "Generating ingestor_plan_generated.h"
            VERBATIM
        )
        add_custom_target(hkp_generate_ingestor_plan_header
            DEPENDS "${HKP_INGESTOR_PLAN_HEADER}"
            COMMENT "Checking ingestor_plan_generated.h"
        )
        add_dependencies(hip_kernel_provider_impl hkp_generate_ingestor_plan_header)
    endif()
endblock()

# Sources above are entirely inside `#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR`; the
# definition must reach this object library, not only the static library tests link.
target_compile_definitions(hip_kernel_provider_impl PUBLIC HIPDNN_ENABLE_KERNEL_INGESTOR=1)

# The FlyDSL pack rides its own gate on top of the ingestor's: the ACTIVE variable, not
# the option (see its definition in the provider's CMakeLists.txt). It holds whenever the
# production root has FlyDSL content, not only when this build packs objects for one of
# its targets: the seams are host code, shipped from one shard of a multi-arch build to
# every architecture. With no objects for the device, no descriptor names them, so the
# engines simply do not appear.
if(HIPKERNELPROVIDER_FLYDSL_ACTIVE)
    target_sources(hip_kernel_provider_impl PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/FlydslRmsNormNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/FlydslRmsNormBwdNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/FlydslSdpaRdnaNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/FlydslSdpaRdnaDecodeNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/FlydslSdpaAiwNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/FlydslSdpaCdnaNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/FlydslSdpaCdnaDualwaveNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/FlydslSdpaCdnaDecodeNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/FlydslSdpaDropoutMaskNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/FlydslSdpaHeadPadNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/FlydslSdpaBwdMergedNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/FlydslSdpaBwdSplitNative.cpp
    )
    target_compile_definitions(hip_kernel_provider_impl PUBLIC HIPDNN_ENGINE_FLYDSL=1)
endif()

# The conv engine group (descriptors/conv/): its packs ride their own gate on top of the
# ingestor's. Its kernels are compiled at runtime through HIPRTC, so the plugin embeds each
# source under the key its descriptors name -- the path relative to the descriptor, beside
# which the source lives -- and each header under its bare file name. The shipped root's
# conv/ family folder is packed only while the same option is ON (HKP_DESCRIPTOR_FAMILIES).
if(HIPKERNELPROVIDER_ENABLE_CONV)
    target_sources(hip_kernel_provider_impl PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvNaiveNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvNaiveProblem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvDepthwiseValuNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvDepthwiseValuProblem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvDepthwiseValuWrwNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvDepthwiseRowsNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvDepthwiseRowsProblem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvDepthwiseMfmaNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvDepthwiseMfmaProblem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvIgemmNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvIgemmNchwNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvIgemmNchwProblem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvWrwNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvWrwProblem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvStemBwdNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvStemBwdProblem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvWinogradNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvDirectNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvSmallcNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvGroupNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvFftNative.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvFftProblem.cpp
    )
    target_compile_definitions(hip_kernel_provider_impl PUBLIC HIPDNN_ENGINE_CONV=1)
    # One definition per served family (conv/ConvContent.cmake): IngestorPacks.cpp
    # registers a family's native pack only when its catalogs ship in this build. The
    # pack sources above still compile for every family, since families share them.
    foreach(_conv_family IN LISTS HIPKERNELPROVIDER_CONV_FAMILIES)
        string(TOUPPER "${_conv_family}" _conv_family_upper)
        target_compile_definitions(hip_kernel_provider_impl
            PUBLIC HIPDNN_ENGINE_CONV_${_conv_family_upper}=1)
    endforeach()
    include("${PROJECT_SOURCE_DIR}/src/cmake/KernelEmbedding.cmake")
    # Embed from the root the production pack packs, so each source sits beside the
    # descriptor that names it (hkp_verify_embedded_sources). A production root overridden
    # to a copy holding conv/ (the UHD pipeline's image build) supplies the sources; one
    # without conv/ falls back to the in-tree folder.
    if(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT
       AND IS_DIRECTORY "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}/conv")
        set(_hkp_conv_root "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}/conv")
    else()
        set(_hkp_conv_root "${CMAKE_CURRENT_SOURCE_DIR}/descriptors/conv")
    endif()
    set(HKP_CONV_KERNEL_SOURCES
        "${_hkp_conv_root}/naive/kernels/ConvNaive.cpp"
        "${_hkp_conv_root}/naive/kernels/ConvNaiveArgs.h"
        "${_hkp_conv_root}/depthwise_valu/kernels/ConvDepthwiseValu.cpp"
        "${_hkp_conv_root}/depthwise_valu/kernels/ConvDepthwiseValuArgs.h"
        "${_hkp_conv_root}/depthwise_valu/kernels/ConvDepthwiseValuCores.h"
        "${_hkp_conv_root}/depthwise_valu_wrw/kernels/ConvDepthwiseValuWrw.cpp"
        "${_hkp_conv_root}/depthwise_valu_wrw/kernels/ConvDepthwiseValuWrwArgs.h"
        "${_hkp_conv_root}/depthwise_valu_wrw/kernels/ConvDepthwiseHankelWrw.h"
        "${_hkp_conv_root}/depthwise_rows/kernels/ConvDepthwiseRows.cpp"
        "${_hkp_conv_root}/depthwise_rows/kernels/ConvDepthwiseRowsCores.h"
        "${_hkp_conv_root}/depthwise_mfma/kernels/ConvDepthwiseMfma.cpp"
        "${_hkp_conv_root}/depthwise_mfma/kernels/ConvDepthwiseMfmaArgs.h"
        "${_hkp_conv_root}/group/kernels/ConvGroup.cpp"
        "${_hkp_conv_root}/group/kernels/ConvGroupArgs.h"
        "${_hkp_conv_root}/group/kernels/ConvGroupCore.h"
        "${_hkp_conv_root}/igemm/kernels/ConvIgemm.cpp"
        "${_hkp_conv_root}/igemm/kernels/ConvIgemmArgs.h"
        "${_hkp_conv_root}/igemm/kernels/ConvIgemmCore.h"
        "${_hkp_conv_root}/igemm/kernels/ConvArchFamily.h"
        "${_hkp_conv_root}/igemm_nchw/kernels/ConvIgemmNchw.cpp"
        "${_hkp_conv_root}/igemm_nchw/kernels/ConvIgemmNchwArgs.h"
        "${_hkp_conv_root}/igemm_nchw/kernels/ConvIgemmNchwCore.h"
        "${_hkp_conv_root}/wrw/kernels/ConvWrw.cpp"
        "${_hkp_conv_root}/wrw/kernels/ConvWrwArgs.h"
        "${_hkp_conv_root}/wrw/kernels/ConvWrwCore.h"
        "${_hkp_conv_root}/wrw/kernels/ConvWrwDirect.h"
        "${_hkp_conv_root}/wrw/kernels/ConvWrwGeom.h"
        "${_hkp_conv_root}/stem_bwd/kernels/ConvStemBwd.cpp"
        "${_hkp_conv_root}/stem_bwd/kernels/ConvStemBwdArgs.h"
        "${_hkp_conv_root}/stem_bwd/kernels/ConvStemBwdCore.h"
        "${_hkp_conv_root}/winograd/kernels/ConvWinograd.cpp"
        "${_hkp_conv_root}/winograd/kernels/ConvWinogradArgs.h"
        "${_hkp_conv_root}/winograd/kernels/ConvWinogradCore.h"
        "${_hkp_conv_root}/winograd/kernels/ConvWinogradFused.h"
        "${_hkp_conv_root}/winograd/kernels/ConvWinogradWmma.h"
        "${_hkp_conv_root}/winograd/kernels/ConvWinogradWmmaReg.h"
        "${_hkp_conv_root}/winograd/kernels/ConvWinogradSreg.h"
        "${_hkp_conv_root}/winograd/kernels/ConvWinogradWmmaCoop.h"
        "${_hkp_conv_root}/winograd/kernels/ConvWinogradWrw.h"
        "${_hkp_conv_root}/direct/kernels/ConvDirect.cpp"
        "${_hkp_conv_root}/direct/kernels/ConvDirectArgs.h"
        "${_hkp_conv_root}/direct/kernels/ConvDirectCore.h"
        "${_hkp_conv_root}/direct/kernels/ConvDirectRows.h"
        "${_hkp_conv_root}/smallc/kernels/ConvSmallc.cpp"
        "${_hkp_conv_root}/smallc/kernels/ConvSmallcArgs.h"
        "${_hkp_conv_root}/smallc/kernels/ConvSmallcCore.h"
        "${_hkp_conv_root}/fft/kernels/ConvFft.cpp"
        "${_hkp_conv_root}/fft/kernels/ConvFftArgs.h"
        "${_hkp_conv_root}/fft/kernels/ConvFftCore.h"
        "${_hkp_conv_root}/fft/kernels/ConvFftTwiddles.h"
        CACHE INTERNAL "Kernel sources of the conv engine group, embedded for HIPRTC")
    set(HKP_CONV_KERNEL_KEYS
        kernels/ConvNaive.cpp
        ConvNaiveArgs.h
        kernels/ConvDepthwiseValu.cpp
        ConvDepthwiseValuArgs.h
        ConvDepthwiseValuCores.h
        kernels/ConvDepthwiseValuWrw.cpp
        ConvDepthwiseValuWrwArgs.h
        ConvDepthwiseHankelWrw.h
        kernels/ConvDepthwiseRows.cpp
        ConvDepthwiseRowsCores.h
        kernels/ConvDepthwiseMfma.cpp
        ConvDepthwiseMfmaArgs.h
        kernels/ConvGroup.cpp
        ConvGroupArgs.h
        ConvGroupCore.h
        kernels/ConvIgemm.cpp
        ConvIgemmArgs.h
        ConvIgemmCore.h
        ConvArchFamily.h
        kernels/ConvIgemmNchw.cpp
        ConvIgemmNchwArgs.h
        ConvIgemmNchwCore.h
        kernels/ConvWrw.cpp
        ConvWrwArgs.h
        ConvWrwCore.h
        ConvWrwDirect.h
        ConvWrwGeom.h
        kernels/ConvStemBwd.cpp
        ConvStemBwdArgs.h
        ConvStemBwdCore.h
        kernels/ConvWinograd.cpp
        ConvWinogradArgs.h
        ConvWinogradCore.h
        ConvWinogradFused.h
        ConvWinogradWmma.h
        ConvWinogradWmmaReg.h
        ConvWinogradSreg.h
        ConvWinogradWmmaCoop.h
        ConvWinogradWrw.h
        kernels/ConvDirect.cpp
        ConvDirectArgs.h
        ConvDirectCore.h
        ConvDirectRows.h
        kernels/ConvSmallc.cpp
        ConvSmallcArgs.h
        ConvSmallcCore.h
        kernels/ConvFft.cpp
        ConvFftArgs.h
        ConvFftCore.h
        ConvFftTwiddles.h
        CACHE INTERNAL "Embedding keys of HKP_CONV_KERNEL_SOURCES, in the same order")
    add_kernels_for_embedding(
        TARGET hip_kernel_provider_impl
        FILES ${HKP_CONV_KERNEL_SOURCES}
        KEYS ${HKP_CONV_KERNEL_KEYS}
    )
endif()
