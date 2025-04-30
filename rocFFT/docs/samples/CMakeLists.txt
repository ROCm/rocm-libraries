# #############################################################################
# Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
# #############################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

# project name
PROJECT(rocfft_samples CXX)

find_package( hip REQUIRED PATHS /opt/rocm/lib/cmake/hip/ )
find_package(rocfft)

set (samples complex_1d complex_2d complex_3d real2complex_1d real2complex_2d real2complex_3d)

foreach( exe ${samples} )
  add_executable(${exe} "${exe}.cpp")

  target_link_libraries( ${exe} PRIVATE hip::device )
  target_link_libraries( ${exe} PRIVATE roc::rocfft )
  target_include_directories( ${exe} PRIVATE ${rocfft_INCLUDE_DIR} )
  set_target_properties( ${exe} PROPERTIES CXX_STANDARD_REQUIRED ON)
  set_target_properties( ${exe} PROPERTIES CXX_STANDARD 14)
  
  # prevent issue where __float128 is not supported
  set_target_properties( ${exe} PROPERTIES CXX_EXTENSIONS OFF)
endforeach()

# Build release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
