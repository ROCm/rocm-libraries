# MIT License
#
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

# Only header target, does not include dependencies
add_library(thrust_headers INTERFACE)
target_include_directories(thrust_headers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

add_library(hipthrust INTERFACE)
target_link_libraries(hipthrust
  INTERFACE
    thrust_headers
    hcc::hccrt
    hcc::hc_am
)
# Remove this check when we no longer build with older rocm stack(ie < 1.8.2)
if(TARGET hip::device)
  target_link_libraries(hipthrust INTERFACE hip::device)
else()
  target_link_libraries(hipthrust INTERFACE hip::hip_hcc hip::hip_device)
endif()

# We will use rocPRIM HIP API
target_compile_definitions(hipthrust
  INTERFACE
    ROCPRIM_HIP_API=1
)
