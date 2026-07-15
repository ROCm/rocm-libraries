# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Builds and ships the fused-RMSNorm "Kernel 2" reduce-and-apply code object
# (partial_rms_epilogue). This is the host-launched companion to the TensileLite
# PartialRMS "K1" GEMM: K1 writes per-tile partial sums-of-squares, then this kernel
# reduces them and applies the RMSNorm scale to D in place. Mirrors the ExtOp pipeline
# (generate .s -> assemble .o -> link .co -> copy into the arch library dir). The kernel
# reads M/N/eps/etc. from kernargs, so a single arch code object serves all problem sizes
# (subject to the M <= 32767 column-address limit enforced on the host).

set(ops_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite")
set(output_dir "${CMAKE_CURRENT_BINARY_DIR}")
string(REGEX MATCHALL "gfx[a-z0-9]+" archs "${GPU_TARGETS}")
list(REMOVE_DUPLICATES archs)

set(previous_target "")

foreach(arch IN LISTS archs)
    # partial_rms_epilogue is a wave64 (CDNA) kernel; skip wave32 (gfx11xx/gfx12xx) arches.
    if(arch MATCHES "^gfx1[12][0-9][0-9]$")
        message(STATUS "hipblaslt device library partial-rms: skipping wave32 arch ${arch}")
        continue()
    endif()

    message(STATUS "hipblaslt device library partial-rms: configuring target for ${arch}")
    tensilelite_offload_target(offload_arch "${arch}")
    set(prms_xnack_args)
    if(NOT "${offload_arch}" STREQUAL "${arch}")
        set(prms_xnack_args --xnack)
    endif()

    add_library(partial-rms-obj-${arch} OBJECT)
    target_compile_options(partial-rms-obj-${arch}
        PRIVATE
            -Wno-unused-command-line-argument -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=${offload_arch} -mwavefrontsize64 -c
    )
    target_sources(partial-rms-obj-${arch}
        PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/partial_rms_epilogue_${arch}.s"
    )
    # The generator now lives in the epilogues package (epilogues/tensilelite/); run it as a
    # module from the tensilelite root so its `Tensile`/`rocisa`/`epilogues` imports resolve
    # (a loose script would put the nested dir on sys.path[0] and fail to import Tensile).
    add_custom_command(
        OUTPUT
            "${CMAKE_CURRENT_BINARY_DIR}/partial_rms_epilogue_${arch}.s"
        COMMAND ${HIPBLASLT_PYTHON_COMMAND} -m epilogues.tensilelite.partial_rms_epilogue_generator -o "${CMAKE_CURRENT_BINARY_DIR}/partial_rms_epilogue_${arch}.s" --arch ${arch} --toolchain ${CMAKE_CXX_COMPILER} ${prms_xnack_args}
        WORKING_DIRECTORY "${ops_dir}"
        COMMENT "Creating partial_rms_epilogue assembly for ${arch}"
        DEPENDS ${HIPBLASLT_PYTHON_DEPS}
    )

    set(output_code_object_file "${CMAKE_CURRENT_BINARY_DIR}/partial_rms_epilogue_${arch}.co")
    add_custom_command(
        DEPENDS partial-rms-obj-${arch}
        OUTPUT "${output_code_object_file}"
        COMMAND ${CMAKE_CXX_COMPILER};-target;amdgcn-amdhsa;-Xlinker;$<TARGET_OBJECTS:partial-rms-obj-${arch}>;-o;${output_code_object_file}
        COMMENT "Creating partial_rms_epilogue_${arch}.co"
        COMMAND_EXPAND_LISTS
    )

    add_custom_target(partial-rms-library-${arch} ALL
        DEPENDS
            "${output_code_object_file}"
            ${previous_target}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${HIPBLASLT_TENSILE_LIBPATH}/library/${arch}"
        COMMAND ${CMAKE_COMMAND} -E copy "${output_code_object_file}" "${HIPBLASLT_TENSILE_LIBPATH}/library/${arch}"
        COMMENT "Installing partial_rms_epilogue_${arch}.co"
    )
    set(previous_target "partial-rms-library-${arch}")
endforeach()
