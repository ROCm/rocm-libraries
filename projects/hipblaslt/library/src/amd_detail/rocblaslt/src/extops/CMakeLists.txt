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
#   - Cp: target associated with cp of *.co and *.dat into Tensile/library

set(ops_path ${PROJECT_SOURCE_DIR}/tensilelite)

string(REGEX MATCHALL "gfx[a-z0-9]+" archs "${Tensile_ARCHITECTURE}")
list(REMOVE_DUPLICATES archs)

set(python_venv_executable "${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME}")
set(python_launch_prefix "PYTHONPATH=${PROJECT_BINARY_DIR}/lib" "${python_venv_executable}")
set(dat_depends "")

if($ENV{ENABLE_ADDRESS_SANITIZER})
    # Must populate LD_PRELOAD with ASAN runtime if ASAN is being used.
    # Find the ASAN RT with compiler and update env for Tensile call.
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=libclang_rt.asan-x86_64.so
        OUTPUT_VARIABLE ASAN_LIB_PATH
        COMMAND_ECHO STDOUT)
    string(STRIP ${ASAN_LIB_PATH} ASAN_LIB_PATH)
    set(python_launch_prefix "LD_PRELOAD=${ASAN_LIB_PATH}" "${python_launch_prefix}")
endif()

foreach(arch IN LISTS archs)
    add_library(ExtOpObj_${arch} OBJECT)
    add_dependencies(ExtOpObj_${arch} rocisa)
    target_compile_options(ExtOpObj_${arch}
        PRIVATE
        -Wno-unused-command-line-argument -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=${arch} -mwavefrontsize64 -c
    )
    target_sources(ExtOpObj_${arch}
        PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/L_256_4_1_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/L_256_4_0_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/S_8_32_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/S_16_16_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/S_4_64_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/S_2_128_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/S_1_256_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/A_S_S_256_4_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/A_H_H_256_4_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/A_H_S_256_4_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/A_S_H_256_4_${arch}.s
    )
    add_custom_command(
        OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/L_256_4_1_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/L_256_4_0_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/S_8_32_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/S_16_16_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/S_4_64_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/S_2_128_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/S_1_256_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/A_S_S_256_4_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/A_H_H_256_4_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/A_H_S_256_4_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/A_S_H_256_4_${arch}.s
        COMMAND ${python_launch_prefix} LayerNormGenerator.py -o ${CMAKE_CURRENT_BINARY_DIR}/L_256_4_1_${arch}.s -w 256 -c 4 --sweep-once 1 --arch ${arch}
        COMMAND ${python_launch_prefix} LayerNormGenerator.py -o ${CMAKE_CURRENT_BINARY_DIR}/L_256_4_0_${arch}.s -w 256 -c 4 --sweep-once 0 --arch ${arch}
        COMMAND ${python_launch_prefix} SoftmaxGenerator.py -o ${CMAKE_CURRENT_BINARY_DIR}/S_8_32_${arch}.s -m 8 -n 32 --arch ${arch}
        COMMAND ${python_launch_prefix} SoftmaxGenerator.py -o ${CMAKE_CURRENT_BINARY_DIR}/S_16_16_${arch}.s -m 16 -n 16 --arch ${arch}
        COMMAND ${python_launch_prefix} SoftmaxGenerator.py -o ${CMAKE_CURRENT_BINARY_DIR}/S_4_64_${arch}.s -m 4 -n 64 --arch ${arch}
        COMMAND ${python_launch_prefix} SoftmaxGenerator.py -o ${CMAKE_CURRENT_BINARY_DIR}/S_2_128_${arch}.s -m 2 -n 128 --arch ${arch}
        COMMAND ${python_launch_prefix} SoftmaxGenerator.py -o ${CMAKE_CURRENT_BINARY_DIR}/S_1_256_${arch}.s -m 1 -n 256 --arch ${arch}
        COMMAND ${python_launch_prefix} AMaxGenerator.py -o ${CMAKE_CURRENT_BINARY_DIR}/A_S_S_256_4_${arch}.s -t S -d S -w 256 -c 4 --arch ${arch}
        COMMAND ${python_launch_prefix} AMaxGenerator.py -o ${CMAKE_CURRENT_BINARY_DIR}/A_H_H_256_4_${arch}.s -t H -d H -w 256 -c 4 --arch ${arch}
        COMMAND ${python_launch_prefix} AMaxGenerator.py -o ${CMAKE_CURRENT_BINARY_DIR}/A_H_S_256_4_${arch}.s -t H -d S -w 256 -c 4 --arch ${arch}
        COMMAND ${python_launch_prefix} AMaxGenerator.py -o ${CMAKE_CURRENT_BINARY_DIR}/A_S_H_256_4_${arch}.s -t S -d H -w 256 -c 4 --arch ${arch}
        COMMENT "Creating Layer Norm, Softmax and Amax Assembly for ${arch}"
        WORKING_DIRECTORY ${ops_path}
    )
    if(arch STREQUAL "gfx942")
        add_custom_command(
            OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/A_S_S_F8N_256_4_${arch}.s
                ${CMAKE_CURRENT_BINARY_DIR}/A_S_S_B8N_256_4_${arch}.s
                ${CMAKE_CURRENT_BINARY_DIR}/A_S_H_F8N_256_4_${arch}.s
                ${CMAKE_CURRENT_BINARY_DIR}/A_S_H_B8N_256_4_${arch}.s
            COMMAND ${python_launch_prefix} AMaxGenerator.py --is-scale -o ${CMAKE_CURRENT_BINARY_DIR}/A_S_S_F8N_256_4_${arch}.s -t S -d S -s F8N -w 256 -c 4 --arch ${arch}
            COMMAND ${python_launch_prefix} AMaxGenerator.py --is-scale -o ${CMAKE_CURRENT_BINARY_DIR}/A_S_S_B8N_256_4_${arch}.s -t S -d S -s B8N -w 256 -c 4 --arch ${arch}
            COMMAND ${python_launch_prefix} AMaxGenerator.py --is-scale -o ${CMAKE_CURRENT_BINARY_DIR}/A_S_H_F8N_256_4_${arch}.s -t S -d H -s F8N -w 256 -c 4 --arch ${arch}
            COMMAND ${python_launch_prefix} AMaxGenerator.py --is-scale -o ${CMAKE_CURRENT_BINARY_DIR}/A_S_H_B8N_256_4_${arch}.s -t S -d H -s B8N -w 256 -c 4 --arch ${arch}
            COMMENT "Creating Extra Amax Assembly for gfx942"
            WORKING_DIRECTORY ${ops_path}
        )
        target_sources(ExtOpObj_${arch}
            PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/A_S_S_F8N_256_4_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/A_S_S_B8N_256_4_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/A_S_H_F8N_256_4_${arch}.s
            ${CMAKE_CURRENT_BINARY_DIR}/A_S_H_B8N_256_4_${arch}.s
        )
    endif()
    if(arch STREQUAL "gfx950")
        add_custom_command(
            OUTPUT
                ${CMAKE_CURRENT_BINARY_DIR}/A_S_S_F8_256_4_${arch}.s
                ${CMAKE_CURRENT_BINARY_DIR}/A_S_S_B8_256_4_${arch}.s
                ${CMAKE_CURRENT_BINARY_DIR}/A_S_H_F8_256_4_${arch}.s
                ${CMAKE_CURRENT_BINARY_DIR}/A_S_H_B8_256_4_${arch}.s
            COMMAND ${python_launch_prefix} AMaxGenerator.py --is-scale -o ${CMAKE_CURRENT_BINARY_DIR}/A_S_S_F8_256_4_${arch}.s -t S -d S -s F8 -w 256 -c 4 --arch ${arch}
            COMMAND ${python_launch_prefix} AMaxGenerator.py --is-scale -o ${CMAKE_CURRENT_BINARY_DIR}/A_S_S_B8_256_4_${arch}.s -t S -d S -s B8 -w 256 -c 4 --arch ${arch}
            COMMAND ${python_launch_prefix} AMaxGenerator.py --is-scale -o ${CMAKE_CURRENT_BINARY_DIR}/A_S_H_F8_256_4_${arch}.s -t S -d H -s F8 -w 256 -c 4 --arch ${arch}
            COMMAND ${python_launch_prefix} AMaxGenerator.py --is-scale -o ${CMAKE_CURRENT_BINARY_DIR}/A_S_H_B8_256_4_${arch}.s -t S -d H -s B8 -w 256 -c 4 --arch ${arch}
            COMMENT "Creating Extra Amax Assembly for gfx950"
            WORKING_DIRECTORY ${ops_path}
        )
        target_sources(ExtOpObj_${arch}
            PRIVATE
                ${CMAKE_CURRENT_BINARY_DIR}/A_S_S_F8_256_4_${arch}.s
                ${CMAKE_CURRENT_BINARY_DIR}/A_S_S_B8_256_4_${arch}.s
                ${CMAKE_CURRENT_BINARY_DIR}/A_S_H_F8_256_4_${arch}.s
                ${CMAKE_CURRENT_BINARY_DIR}/A_S_H_B8_256_4_${arch}.s
        )
    endif()
    add_custom_command(
        DEPENDS ExtOpObj_${arch}
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/extop_${arch}.co
        COMMAND ${CMAKE_CXX_COMPILER};-target;amdgcn-amdhsa;-Xlinker;$<TARGET_OBJECTS:ExtOpObj_${arch}>;-o;${CMAKE_CURRENT_BINARY_DIR}/extop_${arch}.co
        COMMENT "Creating extop_${arch}"
        COMMAND_EXPAND_LISTS
    )
    add_custom_target(ExtOpLibrary_${arch} ALL
        DEPENDS ${dat_depends} ${CMAKE_CURRENT_BINARY_DIR}/extop_${arch}.co
        COMMAND ${python_launch_prefix} ExtOpCreateLibrary.py --src=${CMAKE_CURRENT_BINARY_DIR} --co=${CMAKE_CURRENT_BINARY_DIR}/extop_${arch}.co --output=${CMAKE_CURRENT_BINARY_DIR} --arch=${arch}
        COMMENT "Creating hipblasltExtOpLibrary.dat"
        WORKING_DIRECTORY ${ops_path}
    )
    list(APPEND dat_depends "ExtOpLibrary_${arch}")
endforeach()

add_custom_target(ExtOpCp ALL
    DEPENDS ${dat_depends} TENSILE_LIBRARY_TARGET
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/*.co" ${PROJECT_BINARY_DIR}/Tensile/library
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/*.dat" ${PROJECT_BINARY_DIR}/Tensile/library
    COMMENT "Copying ExtOp Library"
)
