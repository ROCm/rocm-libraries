# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
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

set(_CMAKE_CURRENT_SOURCE_DIR "${HIPBLASLT_LIB_DIR}/include")

configure_file(
    ${HIPBLASLT_LIB_DIR}/include/hipblaslt-version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-version.h
)

generate_export_header(hipblaslt
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-export.h
    BASE_NAME hipblaslt
)

target_sources(hipblaslt
    PUBLIC
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt/hipblaslt.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-export.h>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt/hipblaslt_float8.h>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt/hipblaslt_float8_bc.h>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt/hipblaslt-ext.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt/hipblaslt-ext-op.h>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt/hipblaslt-types.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-version.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt/hipblaslt.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt-export.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt/hipblaslt-ext.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt/hipblaslt-ext-op.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt/hipblaslt_float8.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt/hipblaslt_float8_bc.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt/hipblaslt-types.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt-version.h>
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt/hipblaslt_xfloat32.h"
)
