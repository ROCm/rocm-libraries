# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# The plugin resolves kpack descriptors, which read their kernel code from an archive on
# disk, so nothing is embedded there. The unit-test binary resolves both kinds: it reads
# kpack descriptors from a staged archive, and for embedded_source descriptors it links
# this engine statically and embeds its own copy of the kernel sources.

target_sources(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/IngestorPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/KernelIngestorEngine.cpp
    # Pointwise and Conv descriptors are test fixtures under test_descriptors/; a Pointwise
    # bundle also sits in descriptor-packaging/examples/descriptors, where the superbuild CI
    # lane points the production source root. Gfx950AttentionDense ships from descriptors/.
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/PointwiseNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/Gfx950AttentionDenseNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/IngestorPlanHeader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/IngestorPlanCodec.cpp
)

# serialization/ingestor_plan_generated.h is committed, and the build compiles against it.
# When HIPKERNELPROVIDER_GENERATE_SERIALIZE_HEADERS is ON, regenerate it from the schema with
# hipDNN's flatc flags. Resolve flatc in hipDNN's order: flatbuffers::flatc, then flatc, then
# the FLATBUFFERS_FLATC_EXECUTABLE variables, then a search beside the FlatBuffers package.
# Imported targets are visible only in the directory that found the package, so an
# installed FlatBuffers usually resolves through the search. If no flatc resolves, keep the
# committed header.
block()
    set(HKP_INGESTOR_PLAN_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/serialization/ingestor_plan.fbs")
    set(HKP_INGESTOR_PLAN_HEADER
        "${CMAKE_CURRENT_SOURCE_DIR}/serialization/ingestor_plan_generated.h"
    )
    set(HKP_FLATC_FLAGS_FILE "${ROCM_LIBRARIES_ROOT}/projects/hipdnn/cmake/flatc_flags.txt")

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

    if(HIPKERNELPROVIDER_GENERATE_SERIALIZE_HEADERS AND NOT _hkp_flatc_command STREQUAL ""
       AND EXISTS "${HKP_FLATC_FLAGS_FILE}")
        file(STRINGS "${HKP_FLATC_FLAGS_FILE}" _hkp_flatc_flag_lines)
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
