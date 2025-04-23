# ########################################################################
# Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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

message(STATUS "==========================================")
message(STATUS "\t==>rocalution_VERSION                : ${rocalution_VERSION}")
message(STATUS "\t==>CMAKE_BUILD_TYPE                  : ${CMAKE_BUILD_TYPE}")
message(STATUS "\t==>BUILD_SHARED_LIBS                 : ${BUILD_SHARED_LIBS}")
message(STATUS "\t==>CMAKE_INSTALL_PREFIX link         : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "\t==>CMAKE_MODULE_PATH link            : ${CMAKE_MODULE_PATH}")
message(STATUS "\t==>CMAKE_PREFIX_PATH link            : ${CMAKE_PREFIX_PATH}")
message(STATUS "==============")
message(STATUS "\t==>CMAKE_SYSTEM_NAME                 : ${CMAKE_SYSTEM_NAME}")
message(STATUS "\t>>=HIP_ROOT_DIR                      : ${HIP_ROOT_DIR}")
message(STATUS "\t==>CMAKE_CXX_COMPILER                : ${CMAKE_CXX_FLAGS}")
message(STATUS "\t==>CMAKE_CXX_COMPILER_VERSION        : ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "\t==>CMAKE_CXX_COMPILER debug          : ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "\t==>CMAKE_CXX_COMPILER release        : ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "\t==>CMAKE_CXX_COMPILER relwithdebinfo : ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "\t==>CMAKE_EXE_LINKER_FLAGS            : ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "\t==>CMAKE_EXE_LINKER_FLAGS_RELEASE    : ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
message(STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS         : ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_RELEASE : ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
message(STATUS "==============" )
message(STATUS "\t==>CMAKE_SHARED_LIBRARY_C_FLAGS      : ${CMAKE_SHARED_LIBRARY_C_FLAGS}")
message(STATUS "\t==>CMAKE_SHARED_LIBRARY_CXX_FLAGS    : ${CMAKE_SHARED_LIBRARY_CXX_FLAGS}")
message(STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS         : ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_DEBUG   : ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
message(STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_RELEASE : ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
message(STATUS "==============" )
message(STATUS "\t==>SUPPORT_HIP                       : ${SUPPORT_HIP}")
if(SUPPORT_HIP)
  message(STATUS "\t\t-->HIP_VERSION               : ${HIP_VERSION}")
endif()
message(STATUS "\t==>SUPPORT_OMP                       : ${SUPPORT_OMP}")
if(SUPPORT_OMP)
  message(STATUS "\t\t-->OpenMP_CXX_VERSION        : ${OpenMP_CXX_VERSION}")
  message(STATUS "\t\t-->OpenMP_CXX_LIBRARIES      : ${OpenMP_CXX_LIBRARIES}")
endif()
message(STATUS "\t==>SUPPORT_MPI                       : ${SUPPORT_MPI}")
if(SUPPORT_MPI)
  message(STATUS "\t\t-->MPI_CXX_VERSION           : ${MPI_CXX_VERSION}")
  message(STATUS "\t\t-->MPI_CXX_LIBRARIES         : ${MPI_CXX_LIBRARIES}")
endif()
message(STATUS "==============" )
message(STATUS "\t==>BUILD_CLIENTS_TESTS               : ${BUILD_CLIENTS_TESTS}")
message(STATUS "\t==>BUILD_CLIENTS_SAMPLES             : ${BUILD_CLIENTS_SAMPLES}")
message(STATUS "==============" )
message(STATUS "==========================================")
message(STATUS "\t==>AMDGPU_TARGETS                    : ${AMDGPU_TARGETS}")
