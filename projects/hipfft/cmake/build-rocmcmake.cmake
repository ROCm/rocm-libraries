# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Build ROCmCMakeBuildTools from source as a fallback when no installed version is found.

include(FetchContent)

set(ROCM_CMAKE_SRC_URL
  https://github.com/ROCm/rocm-cmake
  CACHE STRING
  "Location of rocm-cmake git repository"
)
  
FetchContent_Declare( rocm_cmake_local
  GIT_REPOSITORY ${ROCM_CMAKE_SRC_URL}
  GIT_TAG therock-10.1
  GIT_SHALLOW ON
)

FetchContent_MakeAvailable( rocm_cmake_local )

execute_process( COMMAND
  ${CMAKE_COMMAND}
  -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/extern/rocm-cmake .
  WORKING_DIRECTORY ${rocm_cmake_local_SOURCE_DIR}
)
execute_process( COMMAND
  ${CMAKE_COMMAND}
  --build ${rocm_cmake_local_SOURCE_DIR}
  --target install
  WORKING_DIRECTORY
  ${rocm_cmake_local_SOURCE_DIR}
)
find_package( ROCmCMakeBuildTools REQUIRED CONFIG PATHS ${PROJECT_BINARY_DIR}/extern/rocm-cmake )
