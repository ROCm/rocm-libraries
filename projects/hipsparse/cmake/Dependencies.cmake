# ########################################################################
# Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
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

# Dependencies

if( NOT DEFINED ENV{HIP_PATH})
    if(WIN32)
        set( HIP_PATH "C:/hip" )
    else ()
        set( HIP_PATH "/opt/rocm" )
    endif()
else( )
    file(TO_CMAKE_PATH "$ENV{HIP_PATH}" HIP_PATH)
endif( )

# Either rocSPARSE or cuSPARSE is required
if(NOT USE_CUDA)
  if(WIN32)
        find_package(hip REQUIRED CONFIG PATHS ${HIP_PATH} ${ROCM_PATH})
        if( CUSTOM_ROCSPARSE )
            set ( ENV{rocsparse_DIR} ${CUSTOM_ROCSPARSE})
            find_package( rocsparse REQUIRED CONFIG NO_CMAKE_PATH )
        else()
            find_package( rocsparse REQUIRED CONFIG PATHS ${ROCSPARSE_PATH} )
        endif()
  else()
        find_package(hip REQUIRED CONFIG PATHS ${HIP_PATH} ${ROCM_PATH} /opt/rocm)
        find_package( rocsparse REQUIRED CONFIG PATHS /opt/rocm /opt/rocm/rocsparse /usr/local/rocsparse )
  endif()
else()
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${HIP_PATH}/cmake")
  find_package(HIP MODULE REQUIRED)
  list( APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include" )
  find_package(CUDA REQUIRED)
endif()

# ROCm cmake package
find_package(ROCM 0.7.3 QUIET CONFIG PATHS ${CMAKE_PREFIX_PATH})
if(NOT ROCM_FOUND)
  set(PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  file(DOWNLOAD https://github.com/ROCm/rocm-cmake/archive/${rocm_cmake_tag}.zip
       ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip STATUS status LOG log)

  list(GET status 0 status_code)
  list(GET status 1 status_string)

  if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "error: downloading
    'https://github.com/ROCm/rocm-cmake/archive/${rocm_cmake_tag}.zip' failed
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}
    ")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
                  WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})
  execute_process( COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_EXTERN_DIR}/rocm-cmake .
                  WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
  execute_process( COMMAND ${CMAKE_COMMAND} --build rocm-cmake-${rocm_cmake_tag} --target install
                  WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

  find_package( ROCM 0.7.3 REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake )
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMClients)
include(ROCMHeaderWrapper)
