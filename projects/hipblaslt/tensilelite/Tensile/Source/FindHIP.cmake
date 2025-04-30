################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

include(CMakeFindDependencyMacro OPTIONAL RESULT_VARIABLE _CMakeFindDependencyMacro_FOUND)
if (NOT _CMakeFindDependencyMacro_FOUND)
  macro(find_dependency dep)
    if (NOT ${dep}_FOUND)
      set(cmake_fd_version)
      if (${ARGC} GREATER 1)
        set(cmake_fd_version ${ARGV1})
      endif()
      set(cmake_fd_exact_arg)
      if(${CMAKE_FIND_PACKAGE_NAME}_FIND_VERSION_EXACT)
        set(cmake_fd_exact_arg EXACT)
      endif()
      set(cmake_fd_quiet_arg)
      if(${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
        set(cmake_fd_quiet_arg QUIET)
      endif()
      set(cmake_fd_required_arg)
      if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
        set(cmake_fd_required_arg REQUIRED)
      endif()
      find_package(${dep} ${cmake_fd_version}
          ${cmake_fd_exact_arg}
          ${cmake_fd_quiet_arg}
          ${cmake_fd_required_arg}
      )
      if (NOT ${dep}_FOUND)
        set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE "${CMAKE_FIND_PACKAGE_NAME} could not be found because dependency ${dep} could not be found.")
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND False)
        return()
      endif()
      set(cmake_fd_version)
      set(cmake_fd_required_arg)
      set(cmake_fd_quiet_arg)
      set(cmake_fd_exact_arg)
    endif()
  endmacro()
endif()

find_package(CUDA QUIET)

if( CUDA_INCLUDE_DIRS AND CUDA_VERSION AND CUDA_NVCC_EXECUTABLE)
  message(STATUS "CUDA_VERSION = ${CUDA_VERSION}")
  message(STATUS "CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
  message(STATUS "CUDA_NVCC_EXECUTABLE = ${CUDA_NVCC_EXECUTABLE}")

    set( HIP_PLATFORM "nvcc" )

  #export the environment variable, so that HIPCC can find it.
    set(ENV{HIP_PLATFORM} nvcc)

else()
  if (NOT DEFINED HIP_PATH)
    find_path( HIP_INCLUDE_DIR
      NAMES
        hip/hip_runtime.h
      PATHS
        ENV HIP_PATH
        /opt/rocm
      PATH_SUFFIXES
        /include/hip
        /include
    )

    set (HIP_INCLUDE_DIRS ${HIP_INCLUDE_DIR})
    set (HIP_PATH ${HIP_INCLUDE_DIR})
    if (NOT DEFINED ENV{HIP_PATH})
      set( ENV{HIP_PATH} ${HIP_PATH})
    endif( )
  endif()

  message(STATUS "ENV HIP_PATH = $ENV{HIP_PATH}")

  find_program(HIPCC
    NAMES  hipcc
    PATHS
      ENV HIP_PATH
          /opt/rocm
    PATH_SUFFIXES
          /bin
      )

    message(STATUS "HIPCC = ${HIPCC}")

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    HIP
    FOUND_VAR HIP_FOUND
    REQUIRED_VARS HIP_PATH HIP_INCLUDE_DIRS HIPCC)

endif()
