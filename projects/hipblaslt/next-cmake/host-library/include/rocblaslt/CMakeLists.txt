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

set(_CMAKE_CURRENT_SOURCE_DIR1 "${ROCBLASLT_LIB_DIR}/include")
set(_CMAKE_CURRENT_SOURCE_DIR2 "${ROCBLASLT_LIB_DIR}/src/include")

target_sources(hipblaslt
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR1}/Debug.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR1}/rocblaslt-types.h"
        "${_CMAKE_CURRENT_SOURCE_DIR1}/rocblaslt.h"
        "${_CMAKE_CURRENT_SOURCE_DIR1}/rocblaslt-functions.h"
        "${_CMAKE_CURRENT_SOURCE_DIR1}/rocblaslt-auxiliary.h"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/utility.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/rocroller_host.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/rocblaslt_mat_utils.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/tuple_helper.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/tensile_host.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/UserDrivenTuningParser.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/handle.h"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/status.h"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/definitions.h"
        "${_CMAKE_CURRENT_SOURCE_DIR2}/debug.h"
)
