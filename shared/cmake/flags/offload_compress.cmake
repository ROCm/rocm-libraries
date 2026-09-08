# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

include_guard(GLOBAL)

option(BUILD_OFFLOAD_COMPRESS "Build with offload compression" ON)

include(CheckCXXCompilerFlag)

# Function that checks if a HIP compiler function exists.
function(hip_check_compiler_flag _flag _var)
  if(USE_HIPCXX)
    cmake_check_compiler_flag(HIP "${_flag}" ${_var})
  else()
    check_cxx_compiler_flag("-xhip ${_flag}" ${_var})
  endif()
endfunction(hip_check_compiler_flag)

# Enable offload compress.
if(BUILD_OFFLOAD_COMPRESS)
  # Detect capability and add flag.
  hip_check_compiler_flag("--offload-compress" HIP_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
  if(HIP_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
    if(USE_HIPCXX)
      add_compile_options($<$<COMPILE_LANGUAGE:HIP>:--offload-compress>)
    else()
      add_compile_options("--offload-compress")
    endif()
  else()
    message(WARNING "BUILD_OFFLOAD_COMPRESS=ON but flag not supported by compiler. Ignoring option.")
  endif()
endif()
