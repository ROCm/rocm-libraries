# ########################################################################
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

if(BUILD_HIPSPARSE_TESTS)
  if(NOT TARGET hipsparse)
    if(CUSTOM_HIPSPARSE)
      set(ENV{hipsparse_DIR} ${CUSTOM_HIPSPARSE})
      find_package(hipsparse REQUIRED CONFIG NO_CMAKE_PATH)
    else()
      find_package(hipsparse REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm)
    endif()
  endif()
endif()

add_executable(hipsolver-bench client.cpp)

# Internal header includes
target_include_directories(hipsolver-bench
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# External header includes included as system files
target_include_directories(hipsolver-bench
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

target_link_libraries(hipsolver-bench PRIVATE
  ${LAPACK_LIBRARIES}
  hipsolver-common
  clients-common
  roc::hipsolver
)
if(BUILD_HIPSPARSE_TESTS)
  set_source_files_properties(client.cpp
    PROPERTIES
      COMPILE_DEFINITIONS HAVE_HIPSPARSE
  )
  target_link_libraries(hipsolver-bench PRIVATE roc::hipsparse)
endif()
if(BUILD_FORTRAN_BINDINGS)
  target_link_libraries(hipsolver-bench PRIVATE hipsolver_fortran_client)
endif()

add_armor_flags(hipsolver-bench "${ARMOR_LEVEL}")

if(NOT USE_CUDA)
  target_link_libraries(hipsolver-bench PRIVATE hip::host)

  if(CUSTOM_TARGET)
    target_link_libraries(hipsolver-bench PRIVATE hip::${CUSTOM_TARGET})
  endif()

  if(UNIX AND CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
    # hip-clang needs specific flag to turn on pthread and m
    target_link_libraries(hipsolver-bench PRIVATE -lpthread -lm)
  endif()
else()
  target_compile_definitions(hipsolver-bench PRIVATE __HIP_PLATFORM_NVIDIA__)

  target_include_directories(hipsolver-bench
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )

  target_link_libraries(hipsolver-bench PRIVATE ${CUDA_LIBRARIES} Threads::Threads)
endif()

set_target_properties(hipsolver-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

rocm_install(TARGETS hipsolver-bench COMPONENT benchmarks)

target_compile_definitions(hipsolver-bench PRIVATE HIPSOLVER_BENCH ROCM_USE_FLOAT16)
