# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Builds and ships the fused-RMSNorm "Kernel 2" reduce-and-apply code object as
# Fabian's row-major `row_div` kernel. This is the host-launched companion to the
# TensileLite PartialRMS "K1" GEMM under the row-major (free0 = N_hidden)
# convention: K1 writes per-token partial sums-of-squares indexed by WorkGroup0,
# then row_div reduces them across tiles and divides D in place by the RMS scale.
#
# row_div.s is a committed, hand-written wave64 (CDNA) kernel with the gfx950
# target baked into its .amdgcn_target directive, so we assemble it verbatim
# (no code generation) and install one .co per matching arch.

# Fused-RMSNorm reduce kernels (both hand-written wave64 gfx950 kernels):
#   row_div       - full-flow Kernel 2: reduce partials, compute rstd, apply to D in place.
#   row_rstd      - decomposed-flow producer Kernel 2: reduce partials, compute rstd, write it to
#                   the handoff buffer (no D apply) for the GEMM2 RstdScale consumer (Kernel 3).
#   row_div_quant - full-flow Kernel 2: reduce partials, apply rstd, and quantize to FP8.
set(row_kernels row_div row_rstd row_div_quant)
string(REGEX MATCHALL "gfx[a-z0-9]+" archs "${GPU_TARGETS}")
list(REMOVE_DUPLICATES archs)

set(previous_target "")

foreach(arch IN LISTS archs)
    # These are wave64 (CDNA) kernels authored for gfx950; skip other arches.
    if(NOT arch STREQUAL "gfx950")
        message(STATUS "hipblaslt device library row-div: skipping non-gfx950 arch ${arch}")
        continue()
    endif()

    message(STATUS "hipblaslt device library row-div: configuring target for ${arch}")
    tensilelite_offload_target(offload_arch "${arch}")

    foreach(kern IN LISTS row_kernels)
        set(kern_src "${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite/epilogues/kernels/${kern}.s")

        add_library(${kern}-obj-${arch} OBJECT)
        target_compile_options(${kern}-obj-${arch}
            PRIVATE
                -Wno-unused-command-line-argument -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=${offload_arch} -mwavefrontsize64 -c
        )
        # Stage the committed .s into the build dir so the object rule has a local source.
        add_custom_command(
            OUTPUT
                "${CMAKE_CURRENT_BINARY_DIR}/${kern}_${arch}.s"
            COMMAND ${CMAKE_COMMAND} -E copy "${kern_src}" "${CMAKE_CURRENT_BINARY_DIR}/${kern}_${arch}.s"
            DEPENDS "${kern_src}"
            COMMENT "Staging ${kern} assembly for ${arch}"
        )
        target_sources(${kern}-obj-${arch}
            PRIVATE
                "${CMAKE_CURRENT_BINARY_DIR}/${kern}_${arch}.s"
        )

        set(output_code_object_file "${CMAKE_CURRENT_BINARY_DIR}/${kern}_${arch}.co")
        add_custom_command(
            DEPENDS ${kern}-obj-${arch}
            OUTPUT "${output_code_object_file}"
            COMMAND ${CMAKE_CXX_COMPILER};-target;amdgcn-amdhsa;-Xlinker;$<TARGET_OBJECTS:${kern}-obj-${arch}>;-o;${output_code_object_file}
            COMMENT "Creating ${kern}_${arch}.co"
            COMMAND_EXPAND_LISTS
        )

        add_custom_target(${kern}-library-${arch} ALL
            DEPENDS
                "${output_code_object_file}"
                ${previous_target}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${HIPBLASLT_TENSILE_LIBPATH}/library/${arch}"
            COMMAND ${CMAKE_COMMAND} -E copy "${output_code_object_file}" "${HIPBLASLT_TENSILE_LIBPATH}/library/${arch}"
            COMMENT "Installing ${kern}_${arch}.co"
        )
        set(previous_target "${kern}-library-${arch}")
    endforeach()
endforeach()
