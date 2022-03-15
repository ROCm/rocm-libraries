# MIT License
#
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Find or download/install rocm-cmake project
find_package(ROCM 0.7.3 QUIET CONFIG PATHS /opt/rocm)
if(NOT ROCM_FOUND)
    set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
    file(
        DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
        ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
        STATUS rocm_cmake_download_status LOG rocm_cmake_download_log
    )
    list(GET rocm_cmake_download_status 0 rocm_cmake_download_error_code)
    if(rocm_cmake_download_error_code)
        message(FATAL_ERROR "Error: downloading "
            "https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip failed "
            "error_code: ${rocm_cmake_download_error_code} "
            "log: ${rocm_cmake_download_log} "
        )
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        RESULT_VARIABLE rocm_cmake_unpack_error_code
    )
    execute_process( COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake .
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag} )
    execute_process( COMMAND ${CMAKE_COMMAND} --build rocm-cmake-${rocm_cmake_tag} --target install
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  
    if(rocm_cmake_unpack_error_code)
        message(FATAL_ERROR "Error: unpacking ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip failed")
    endif()
    find_package( ROCM 0.7.3 REQUIRED CONFIG PATHS ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake )
endif()
