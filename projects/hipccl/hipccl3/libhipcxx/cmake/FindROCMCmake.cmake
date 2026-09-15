# Copyright (c) 2025 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# ###########################
# libhipcxx dependencies
# ###########################

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

set(PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern)

# By default, rocm software stack is expected at /opt/rocm
# set environment variable ROCM_PATH to change location
if(NOT ROCM_PATH)
  set(ROCM_PATH /opt/rocm)
endif()

find_package(ROCM 0.7.3 CONFIG QUIET PATHS ${ROCM_PATH} /opt/rocm)
if(NOT ROCM_FOUND)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  set(rocm_cmake_url "https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip")
  set(rocm_cmake_path "${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}")
  set(rocm_cmake_archive "${rocm_cmake_path}.zip")
  file(DOWNLOAD "${rocm_cmake_url}" "${rocm_cmake_archive}" STATUS status LOG log)

  list(GET status 0 status_code)
  list(GET status 1 status_string)

  if(status_code EQUAL 0)
    message(STATUS "downloading... done")
  else()
    message(FATAL_ERROR "error: downloading\n'${rocm_cmake_url}' failed
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}\n")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzvf "${rocm_cmake_archive}"
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})
  execute_process( COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_EXTERN_DIR}/rocm-cmake .
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
  execute_process( COMMAND ${CMAKE_COMMAND} --build rocm-cmake-${rocm_cmake_tag} --target install
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

  find_package( ROCM 0.7.3 REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake )
endif()
