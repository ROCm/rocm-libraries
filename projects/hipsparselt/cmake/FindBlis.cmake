# ########################################################################
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################
set(BLIS_FOUND FALSE)
if(NOT ("${BUILD_DIR}" STREQUAL ""))
    if(EXISTS          "${BUILD_DIR}/deps/blis/lib/libblis.a")
        set(BLIS_INCLUDE_DIR ${BUILD_DIR}/deps/blis/include/blis)
        set(BLIS_LIB ${BUILD_DIR}/deps/blis/lib/libblis.a)
        set(BLIS_FOUND TRUE)
    endif()
endif()

if(NOT ${BLIS_FOUND})
    set( BLIS_FOUND TRUE )
    if(EXISTS          "/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libblis-mt.a")
        set(BLIS_LIB /opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libblis-mt.a)
        set(BLIS_INCLUDE_DIR /opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/include_ILP64)
    elseif(EXISTS      "/opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/lib_ILP64/libblis-mt.a")
        set(BLIS_LIB /opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/lib_ILP64/libblis-mt.a)
        set(BLIS_INCLUDE_DIR /opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/include_ILP64)
    elseif(EXISTS      "/opt/AMD/aocl/aocl-linux-aocc-4.0/lib_ILP64/libblis-mt.a")
        set(BLIS_LIB /opt/AMD/aocl/aocl-linux-aocc-4.0/lib_ILP64/libblis-mt.a)
        set(BLIS_INCLUDE_DIR /opt/AMD/aocl/aocl-linux-aocc-4.0/include_ILP64)
    elseif(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/../deps/amd-blis/lib/ILP64/libblis-mt.a") # 4.0 and 4.1.0
        set(BLIS_LIB ${CMAKE_CURRENT_BINARY_DIR}/../deps/amd-blis/lib/ILP64/libblis-mt.a)
        set(BLIS_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/../deps/amd-blis/include/ILP64)
    elseif(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/../deps/blis/lib/libblis.a")
        set(BLIS_LIB ${CMAKE_CURRENT_BINARY_DIR}/../deps/blis/lib/libblis.a)
        set(BLIS_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/../deps/blis/include/blis)
    elseif(EXISTS      "/usr/local/lib/libblis.a")
        set(BLIS_LIB /usr/local/lib/libblis.a)
        set(BLIS_INCLUDE_DIR /usr/local/include/blis)
    else()
        set(BLIS_FOUND FALSE)
        message(FATAL_ERROR "BLIS lib not found.")
    endif()
endif()

set(BLIS_INCLUDE_DIR ${BLIS_INCLUDE_DIR} PARENT_SCOPE)
set(BLIS_LIB ${BLIS_LIB} PARENT_SCOPE)

message("BLIS header directory found: ${BLIS_INCLUDE_DIR}")
message("BLIS lib found: ${BLIS_LIB}")
