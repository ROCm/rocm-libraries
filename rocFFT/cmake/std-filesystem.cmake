# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

include(CheckCXXSourceCompiles)

set(HAVE_STD_FILESYSTEM_TEST [[
  #include <filesystem>

  int main()
  {
  std::filesystem::path p{"/"};
  return 0;
  }
  ]])

set(CMAKE_REQUIRED_FLAGS -std=c++17)
check_cxx_source_compiles("${HAVE_STD_FILESYSTEM_TEST}" HAVE_STD_FILESYSTEM)

if(NOT HAVE_STD_FILESYSTEM)
  message(STATUS "std::filesystem include not present, will use std::experimental::filesystem")
endif()

# Link to the experimental filesystem library if it's not available
# in the standard library.  Experimental filesystem library is not
# ABI-compatible with later libstdc++ so link that statically too.
function(target_link_std_experimental_filesystem target)
  if(NOT HAVE_STD_FILESYSTEM)
    target_link_options( ${target} PRIVATE "SHELL:-lstdc++fs -static-libstdc++ -Xlinker --exclude-libs=ALL")
  endif()
endfunction()
