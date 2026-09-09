# MIT License
#
# Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

include(dependencies/rocm-cmake)
include(dependencies/monorepo)

# Test dependencies
if(BUILD_TEST)
  include(dependencies/googletest)
endif()

# CUB (only for CUDA platform)
if(HIP_COMPILER STREQUAL "nvcc")
  set(CCCL_MINIMUM_VERSION 3.1.4)
  if(NOT DOWNLOAD_CUB)
    find_package(CCCL ${CCCL_MINIMUM_VERSION} CONFIG)
  endif()

  if(NOT CCCL_FOUND)
    message(STATUS "CCCL not found, downloading and extracting CCCL ${CCCL_MINIMUM_VERSION}")
    file(DOWNLOAD https://github.com/NVIDIA/cccl/archive/refs/tags/v${CCCL_MINIMUM_VERSION}.zip
      ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip
      STATUS cccl_download_status LOG cccl_download_log)

    list(GET cccl_download_status 0 cccl_download_error_code)

    if(cccl_download_error_code)
      message(FATAL_ERROR "Error: downloading "
        "https://github.com/NVIDIA/cccl/archive/refs/tags/v${CCCL_MINIMUM_VERSION}.zip failed "
        "error_code: ${cccl_download_error_code} "
        "log: ${cccl_download_log}")
    endif()

    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
      file(ARCHIVE_EXTRACT INPUT ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip)
    else()
      execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xf ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        RESULT_VARIABLE cccl_unpack_error_code)

      if(cccl_unpack_error_code)
        message(FATAL_ERROR "Error: unpacking ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip failed")
      endif()
    endif()

    find_package(CCCL ${CCCL_MINIMUM_VERSION} CONFIG REQUIRED NO_DEFAULT_PATH
      PATHS ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION})
  endif()
else()
  # rocPRIM (only for ROCm platform)
  # NOTE(hipccl3): if rocprim is already a target in this build (e.g. the
  # unified hipccl3 superbuild, which runs add_subdirectory(rocprim) before
  # hipcub), reuse it directly instead of calling fetch_monorepo_dep(). Its
  # MONOREPO path calls FetchContent_MakeAvailable()/add_subdirectory() on a
  # second copy of rocprim's source, which would try to re-declare the
  # `rocprim`/`rocprim_hip` targets and fatally collide with the ones already
  # defined by the sibling add_subdirectory(rocprim) call.
  if(TARGET rocprim)
    message(STATUS "rocprim target already exists in this build - reusing it instead of fetching a separate copy")

    if(NOT TARGET roc::rocprim)
      add_library(roc::rocprim ALIAS rocprim)
    endif()

    if(TARGET rocprim_hip AND NOT TARGET roc::rocprim_hip)
      add_library(roc::rocprim_hip ALIAS rocprim_hip)
    endif()
  else()
    fetch_monorepo_dep(
      PACKAGE rocprim
      VERSION ${MIN_ROCPRIM_PACKAGE_VERSION}
    )
  endif()

  # libhipcxx (only for ROCm platform) - hard dependency, see require_libhipcxx.cmake
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules")
  include(require_libhipcxx)
  hipccl_require_libhipcxx(3.4)
endif()
