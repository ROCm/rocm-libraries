# ########################################################################
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# Need to support Release/Debug

# Keeps a "-strict" suffix while still dropping ":xnack+" (re-added below by
# tensilelite_offload_target). Truncating it would collapse gfx1250-strict onto
# gfx1250, leaving gfx1250-strict with code objects it cannot load.
string(REGEX MATCHALL "gfx[a-z0-9]+(-[a-z0-9]+)*" archs "${GPU_TARGETS}")
list(REMOVE_DUPLICATES archs)

set(matrix_transform_cpp "${CMAKE_CURRENT_SOURCE_DIR}/matrix_transform.cpp")

# clang derives the HIP compilation-unit ID (and the __hip_cuid_<hash> symbol
# it emits) from the input path and the full command line, including -o, so
# the .hsaco would differ between source/build directories. Pin it so the
# output is reproducible. A fixed value is safe: this is a single device-only
# TU loaded via hipModuleLoad, never linked with host code or other HIP TUs.
set(matrix_transform_cuid "hipblasltTransform")

foreach(arch IN LISTS archs)
    tensilelite_offload_target(offload_arch "${arch}")
    set(output_library "${CMAKE_CURRENT_BINARY_DIR}/hipblasltTransform_${arch}.hsaco")
    add_custom_command(
        COMMAND ${CMAKE_CXX_COMPILER} -x hip ${matrix_transform_cpp} -cuid=${matrix_transform_cuid} --offload-arch=${offload_arch} -c --offload-device-only -Xoffload-linker --build-id=sha1 -O3 -DNDEBUG -o ${output_library}
        OUTPUT ${output_library}
        COMMENT "Building hipblasltTransform_${arch}.hsaco"
    )
    add_custom_target(matrix-transform-cp-${arch} ALL
        DEPENDS ${output_library}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${HIPBLASLT_TENSILE_LIBPATH}/library/${arch}"
        COMMAND ${CMAKE_COMMAND} -E copy "${output_library}" "${HIPBLASLT_TENSILE_LIBPATH}/library/${arch}/hipblasltTransform_${arch}.hsaco"
    )
endforeach()

# The MXFP6 packers are fixed-shape gfx950 kernels kept as assembly. They are linked into their
# own code object, which the transform loads from the directory of hipblasltTransform_gfx950.hsaco.
if("gfx950" IN_LIST archs)
    set(pack_sources
        "${CMAKE_CURRENT_SOURCE_DIR}/gfx950/rocke_mxfp6_pack_op_4096x65536_r256_w128t4_xcd_c0c1p_nosc.s"
        "${CMAKE_CURRENT_SOURCE_DIR}/gfx950/rocke_mxfp6_pack_scales_4096_0x65536_16x16x128_xcd_nts.s"
    )
    set(pack_objects)
    foreach(pack_source IN LISTS pack_sources)
        get_filename_component(pack_name "${pack_source}" NAME_WE)
        set(pack_object "${CMAKE_CURRENT_BINARY_DIR}/${pack_name}.o")
        add_custom_command(
            OUTPUT "${pack_object}"
            COMMAND ${CMAKE_CXX_COMPILER} -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx950 -mwavefrontsize64 -c "${pack_source}" -o "${pack_object}"
            DEPENDS "${pack_source}"
            COMMENT "Assembling ${pack_name}.s"
            VERBATIM
        )
        list(APPEND pack_objects "${pack_object}")
    endforeach()
    set(pack_library "${CMAKE_CURRENT_BINARY_DIR}/hipblasltTransformPack_gfx950.co")
    add_custom_command(
        OUTPUT "${pack_library}"
        COMMAND ${CMAKE_CXX_COMPILER} -target amdgcn-amd-amdhsa ${pack_objects} -o "${pack_library}"
        DEPENDS ${pack_objects}
        COMMENT "Building hipblasltTransformPack_gfx950.co"
        VERBATIM
    )
    add_custom_target(matrix-transform-pack-cp-gfx950 ALL
        DEPENDS "${pack_library}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${HIPBLASLT_TENSILE_LIBPATH}/library/gfx950"
        COMMAND ${CMAKE_COMMAND} -E copy "${pack_library}" "${HIPBLASLT_TENSILE_LIBPATH}/library/gfx950/hipblasltTransformPack_gfx950.co"
    )

endif()
