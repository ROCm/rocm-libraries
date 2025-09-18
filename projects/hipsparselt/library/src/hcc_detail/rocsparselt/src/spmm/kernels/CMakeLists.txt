# ########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
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

set(utils_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/hcc_detail/rocsparselt/utils)
set(KERNELS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/hcc_detail/rocsparselt/src/spmm/kernels)

macro(GENERATE_KERNEL_LIB arch)
  message(STATUS "GENERATE_KERNEL_LIB: " ${arch})
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${arch})
  file(GLOB_RECURSE SPMM_KERNELS_O_${arch} ${KERNELS_PATH}/${arch}/*.co)
  add_custom_target(spmm_kernels_o_${arch} DEPENDS ${SPMM_KERNELS_O_${arch}})
  add_custom_target(spmm_kernels_cpp_${arch} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${arch}/kernels.cpp)
  add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${arch}/kernels.cpp
        COMMAND addKernels ${CMAKE_CURRENT_BINARY_DIR}/${arch} kernels ${SPMM_KERNELS_O_${arch}}
        DEPENDS spmm_kernels_o_${arch} addKernels)
  add_custom_target(spmm_kernels_params_cpp_${arch} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${arch}/kernel_params.cpp)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${arch}/kernel_params.cpp
    COMMAND python3 ${utils_dir}/addKernels.py --filename ${CMAKE_CURRENT_BINARY_DIR}/${arch}/kernel_params.cpp --yaml ${KERNELS_PATH}/${arch}
    DEPENDS spmm_kernels_o_${arch})

  add_library(spmm_kernels_${arch} SHARED ${CMAKE_CURRENT_BINARY_DIR}/${arch}/kernels.cpp ${CMAKE_CURRENT_BINARY_DIR}/${arch}/kernel_params.cpp)
  set(SPMM_KERNEL_LIBS ${KERNEL_LIBS} spmm_kernels_${arch})
  add_dependencies(spmm_kernels_${arch} spmm_kernels_cpp_${arch} spmm_kernels_params_cpp_${arch})

  set_target_properties(spmm_kernels_${arch} PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/SPMM_KERNELS/library)
endmacro()

add_executable(addKernels ${utils_dir}/addKernels.cpp)

FILE(GLOB arch_folders RELATIVE ${KERNELS_PATH} ${KERNELS_PATH}/*)
FOREACH(f ${arch_folders})
  IF(IS_DIRECTORY ${KERNELS_PATH}/${f})
      GENERATE_KERNEL_LIB(${f})
  ENDIF()
ENDFOREACH()
