#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#####################################################################################

function(register_op TARGET_NAME)
    set(options)
    set(oneValueArgs HEADER)
    set(multiValueArgs OPERATORS INCLUDES)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    string(MAKE_C_IDENTIFIER "${PARSE_HEADER}" BASE_NAME)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ops)
    set(FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/ops/${BASE_NAME}.cpp)
    file(WRITE "${FILE_NAME}" "")
    foreach(INCLUDE ${PARSE_INCLUDES})
        file(APPEND "${FILE_NAME}" "
#include <${INCLUDE}>
")
    endforeach()
    file(APPEND "${FILE_NAME}" "
#include <migraphx/register_op.hpp>
#include <${PARSE_HEADER}>
")


        file(APPEND "${FILE_NAME}" "
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
")
    foreach(OPERATOR ${PARSE_OPERATORS})
        file(APPEND "${FILE_NAME}" "
MIGRAPHX_REGISTER_OP(${OPERATOR})
")
    endforeach()
    file(APPEND "${FILE_NAME}" "
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
")
    target_sources(${TARGET_NAME} PRIVATE ${FILE_NAME})
endfunction()
