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

add_executable(example-c-basic example_basic.c)
add_executable(example-cpp-basic example_basic.cpp)

# We test for C99 compatibility in the example-c.c test
set_source_files_properties(example_basic.c PROPERTIES LANGUAGE CXX)
set_source_files_properties(example_basic.c PROPERTIES COMPILE_FLAGS "-xc -std=c99")

# Test for C++11 compatibility in one of the samples
set_property(TARGET example-cpp-basic PROPERTY CXX_STANDARD 11)

foreach(exe example-c-basic;example-cpp-basic;)

  # External header includes included as SYSTEM files
  target_include_directories(${exe}
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )

  target_include_directories(${exe} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

  target_link_libraries(${exe} PRIVATE roc::hipsolver)

  if(NOT USE_CUDA)
    target_link_libraries(${exe} PRIVATE hip::host)

    if(CUSTOM_TARGET)
      target_link_libraries(${exe} PRIVATE hip::${CUSTOM_TARGET})
    endif()

  else()
    target_compile_definitions(${exe} PRIVATE __HIP_PLATFORM_NVIDIA__)

    target_include_directories(${exe}
      PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )

    target_link_libraries(${exe} PRIVATE ${CUDA_LIBRARIES})
  endif()

  set_target_properties(${exe} PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(${exe} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

endforeach()
