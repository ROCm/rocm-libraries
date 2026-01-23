# Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
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

# For every public header build a TU that directly includes it
# without anything else but also pretents to be a std header
add_custom_target(libcudacxx.test.public_headers_host_only)

if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  find_package(NVHPC)
else()
  find_package(CUDAToolkit)
endif()

# Grep all public headers
file(GLOB public_headers_host_only
  LIST_DIRECTORIES false
  RELATIVE "${libhipcxx_SOURCE_DIR}/include/"
  CONFIGURE_DEPENDS
  "${libhipcxx_SOURCE_DIR}/include/cuda/*"
)

# mdspan is currently not supported on msvc outside of C++20
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" AND NOT "${CMAKE_CXX_STANDARD}" MATCHES "20")
  list(FILTER public_headers_host_only EXCLUDE REGEX "mdspan")
endif()

function(libcudacxx_add_std_header_test header)
  # ${header} contains the "/" from the subfolder, replace by "_" for actual names
  string(REPLACE "/" "_" header_name "${header}")

  # NOTE(HIP/AMD): Skip unsupported headers for AMD devices (no support as of November 2025).
  string(REGEX MATCH "ptx|barrier|latch|semaphore|annotated_ptr|pipeline|memcpy_async" match "${header}")
  if(match)
    return()
  endif()

  # Create the source file for the header target from the template and add the file to the global project
  set(headertest_src "headers/${header_name}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/header_test.cpp.in" "${headertest_src}.cpp")

  # Create the default target for that file
  set(headertest_std_${header_name} verify_${header_name})
  add_library(headertest_std_${header_name} SHARED "${headertest_src}.cpp")
  set_source_files_properties(${headertest_src}.cpp PROPERTIES LANGUAGE HIP)

  target_include_directories(headertest_std_${header_name} PRIVATE "${libhipcxx_SOURCE_DIR}/include")
  target_compile_options(headertest_std_${header_name} PRIVATE ${headertest_warning_levels_host})
  target_compile_definitions(headertest_std_${header_name} PRIVATE CCCL_ENABLE_ASSERTIONS)
  target_compile_definitions(headertest_std_${header_name} PRIVATE CCCL_IGNORE_DEPRECATED_CPP_DIALECT)
  target_compile_definitions(headertest_std_${header_name} PRIVATE CCCL_ENABLE_OPTIONAL_REF)

  # We want to ensure that we can build headers within <cuda/> with a host compiler but we need cuda_runtime_api.h
  if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    target_link_libraries(headertest_std_${header_name} NVHPC::CUDART)
  else()
    target_link_libraries(headertest_std_${header_name} hip::device)
  endif()

  add_dependencies(libcudacxx.test.public_headers_host_only headertest_std_${header_name})
endfunction()

foreach(header IN LISTS public_headers_host_only)
  libcudacxx_add_std_header_test(${header})
endforeach()
