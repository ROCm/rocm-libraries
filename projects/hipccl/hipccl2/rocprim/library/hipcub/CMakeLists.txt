# MIT License
#
# Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

# hipCUB header-only library

# Configure a header file to pass the hipCUB version
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/hipcub_version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/hipcub_version.hpp"
    @ONLY
)

# Only header target, does not include dependencies
add_library(hipcub_headers INTERFACE)
target_include_directories(hipcub_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:hipcub/include>
)

if(HIP_PLATFORM STREQUAL "hcc")
    add_library(hipcub INTERFACE)
    target_link_libraries(hipcub
        INTERFACE
            hipcub_headers
            rocprim_hip
    )
else()
    add_library(hipcub INTERFACE)
    target_include_directories(hipcub
        INTERFACE
            ${CUB_INCLUDE_DIR}
    )
    target_link_libraries(hipcub
        INTERFACE
            hipcub_headers
    )
endif()
