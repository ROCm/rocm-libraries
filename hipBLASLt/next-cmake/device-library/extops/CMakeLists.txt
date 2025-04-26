# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
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

# This file creates targets of the form ExtOp<ID>_<arch> where ID is:
#   - Obj: targets that map *.s into *.o
#   - Library: target associated creation of .dat

set(ops_dir ${TENSILELITE_DIR}/Tensile/Ops)
set(output_dir "${CMAKE_CURRENT_BINARY_DIR}")
string(REGEX MATCHALL "gfx[a-z0-9]+" archs "${GPU_TARGETS}")
list(REMOVE_DUPLICATES archs)
set(extop_cp_depends "")

foreach(arch IN LISTS archs)
    add_library(extop-obj-${arch} OBJECT)
    target_compile_options(extop-obj-${arch}
        PRIVATE
            -Wno-unused-command-line-argument -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=${arch} -mwavefrontsize64 -c
    )
    target_sources(extop-obj-${arch}
        PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/L_256_4_1_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/L_256_4_0_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_8_32_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_16_16_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_4_64_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_2_128_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_1_256_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_256_4_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_H_H_256_4_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_H_S_256_4_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_256_4_${arch}.s"
    )
    add_custom_command(
        OUTPUT
            "${CMAKE_CURRENT_BINARY_DIR}/L_256_4_1_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/L_256_4_0_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_8_32_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_16_16_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_4_64_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_2_128_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_1_256_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_256_4_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_H_H_256_4_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_H_S_256_4_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_256_4_${arch}.s"
        COMMAND ${Python3_EXECUTABLE} LayerNormGenerator.py -o "${CMAKE_CURRENT_BINARY_DIR}/L_256_4_1_${arch}.s" -w 256 -c 4 --sweep-once 1 --arch ${arch}
        COMMAND ${Python3_EXECUTABLE} LayerNormGenerator.py -o "${CMAKE_CURRENT_BINARY_DIR}/L_256_4_0_${arch}.s" -w 256 -c 4 --sweep-once 0 --arch ${arch}
        COMMAND ${Python3_EXECUTABLE} SoftmaxGenerator.py -o "${CMAKE_CURRENT_BINARY_DIR}/S_8_32_${arch}.s" -m 8 -n 32 --arch ${arch}
        COMMAND ${Python3_EXECUTABLE} SoftmaxGenerator.py -o "${CMAKE_CURRENT_BINARY_DIR}/S_16_16_${arch}.s" -m 16 -n 16 --arch ${arch}
        COMMAND ${Python3_EXECUTABLE} SoftmaxGenerator.py -o "${CMAKE_CURRENT_BINARY_DIR}/S_4_64_${arch}.s" -m 4 -n 64 --arch ${arch}
        COMMAND ${Python3_EXECUTABLE} SoftmaxGenerator.py -o "${CMAKE_CURRENT_BINARY_DIR}/S_2_128_${arch}.s" -m 2 -n 128 --arch ${arch}
        COMMAND ${Python3_EXECUTABLE} SoftmaxGenerator.py -o "${CMAKE_CURRENT_BINARY_DIR}/S_1_256_${arch}.s" -m 1 -n 256 --arch ${arch}
        COMMAND ${Python3_EXECUTABLE} AMaxGenerator.py -o "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_256_4_${arch}.s" -t S -d S -w 256 -c 4 --arch ${arch}
        COMMAND ${Python3_EXECUTABLE} AMaxGenerator.py -o "${CMAKE_CURRENT_BINARY_DIR}/A_H_H_256_4_${arch}.s" -t H -d H -w 256 -c 4 --arch ${arch}
        COMMAND ${Python3_EXECUTABLE} AMaxGenerator.py -o "${CMAKE_CURRENT_BINARY_DIR}/A_H_S_256_4_${arch}.s" -t H -d S -w 256 -c 4 --arch ${arch}
        COMMAND ${Python3_EXECUTABLE} AMaxGenerator.py -o "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_256_4_${arch}.s" -t S -d H -w 256 -c 4 --arch ${arch}
        COMMENT "Creating Layer Norm, Softmax and Amax Assembly for ${arch}"
        WORKING_DIRECTORY ${ops_dir}
    )
    if(arch STREQUAL "gfx942")
        add_custom_command(
            OUTPUT
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_F8N_256_4_${arch}.s"
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_B8N_256_4_${arch}.s"
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_F8N_256_4_${arch}.s"
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_B8N_256_4_${arch}.s"
            COMMAND ${Python3_EXECUTABLE} AMaxGenerator.py --is-scale -o "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_F8N_256_4_${arch}.s" -t S -d S -s F8N -w 256 -c 4 --arch ${arch}
            COMMAND ${Python3_EXECUTABLE} AMaxGenerator.py --is-scale -o "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_B8N_256_4_${arch}.s" -t S -d S -s B8N -w 256 -c 4 --arch ${arch}
            COMMAND ${Python3_EXECUTABLE} AMaxGenerator.py --is-scale -o "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_F8N_256_4_${arch}.s" -t S -d H -s F8N -w 256 -c 4 --arch ${arch}
            COMMAND ${Python3_EXECUTABLE} AMaxGenerator.py --is-scale -o "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_B8N_256_4_${arch}.s" -t S -d H -s B8N -w 256 -c 4 --arch ${arch}
            COMMENT "Creating Extra Amax Assembly for gfx942"
            WORKING_DIRECTORY ${ops_dir}
        )
        target_sources(extop-obj-${arch}
            PRIVATE
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_F8N_256_4_${arch}.s"
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_B8N_256_4_${arch}.s"
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_F8N_256_4_${arch}.s"
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_B8N_256_4_${arch}.s"
        )
    endif()
    if(arch STREQUAL "gfx950")
        add_custom_command(
            OUTPUT
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_F8_256_4_${arch}.s"
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_B8_256_4_${arch}.s"
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_F8_256_4_${arch}.s"
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_B8_256_4_${arch}.s"
            COMMAND ${Python3_EXECUTABLE} AMaxGenerator.py --is-scale -o "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_F8_256_4_${arch}.s" -t S -d S -s F8 -w 256 -c 4 --arch ${arch}
            COMMAND ${Python3_EXECUTABLE} AMaxGenerator.py --is-scale -o "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_B8_256_4_${arch}.s" -t S -d S -s B8 -w 256 -c 4 --arch ${arch}
            COMMAND ${Python3_EXECUTABLE} AMaxGenerator.py --is-scale -o "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_F8_256_4_${arch}.s" -t S -d H -s F8 -w 256 -c 4 --arch ${arch}
            COMMAND ${Python3_EXECUTABLE} AMaxGenerator.py --is-scale -o "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_B8_256_4_${arch}.s" -t S -d H -s B8 -w 256 -c 4 --arch ${arch}
            COMMENT "Creating Extra Amax Assembly for gfx950"
            WORKING_DIRECTORY ${ops_dir}
        )
        target_sources(extop-obj-${arch}
            PRIVATE
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_F8_256_4_${arch}.s"
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_B8_256_4_${arch}.s"
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_F8_256_4_${arch}.s"
                "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_B8_256_4_${arch}.s"
        )
    endif()

    add_custom_command(
        DEPENDS extop-obj-${arch}
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/extop_${arch}.co
        COMMAND ${CMAKE_CXX_COMPILER};-target;amdgcn-amdhsa;-Xlinker;$<TARGET_OBJECTS:extop-obj-${arch}>;-o;${output_dir}/extop_${arch}.co
        COMMENT "Creating extop_${arch}"
        COMMAND_EXPAND_LISTS
    )
    list(APPEND extop_cp_depends "extop-library-${arch}")
    add_custom_target(extop-library-${arch}
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/extop_${arch}.co
        COMMAND ${Python3_EXECUTABLE} ExtOpCreateLibrary.py --src=${CMAKE_CURRENT_BINARY_DIR} --co=${output_dir}/extop_${arch}.co --output=${output_dir} --arch=${arch}
        COMMENT "Creating hipblasltExtOpLibrary.dat for ${arch}"
        WORKING_DIRECTORY ${ops_dir}
    )
endforeach()

add_custom_target(extop-cp ALL
    DEPENDS ${extop_cp_depends}
    COMMAND ${CMAKE_COMMAND} -E copy "${output_dir}/*.co" ${device_library_dir}
    COMMAND ${CMAKE_COMMAND} -E copy "${output_dir}/*.dat" ${device_library_dir}
)
