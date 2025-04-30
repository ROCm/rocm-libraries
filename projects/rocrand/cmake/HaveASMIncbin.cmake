# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

# ${CMAKE_VERSION} < 3.17 doesn't have ${CMAKE_CURRENT_FUNCTION_LIST_DIR}
set(_ROCRAND_HAVE_ASM_INCBIN_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(rocrand_check_have_asm_incbin RESULT_VAR)
    if (DEFINED ${RESULT_VAR})
        return()
    endif()

    enable_language(ASM)
    # ${CMAKE_VERSION} < 3.17 doesn't have message(CHECK_(START|PASS|FAIL)
    message(STATUS "Performing Test ${RESULT_VAR}")

    try_compile(${RESULT_VAR}
        ${PROJECT_BINARY_DIR}/have_asm_incbin_test # bindir
        ${_ROCRAND_HAVE_ASM_INCBIN_BASE_DIR}/have_asm_incbin_test # srcdir
        rocrand_have_asm_incbin_test # projectName
        OUTPUT_VARIABLE OUTPUT
    )

    if (${RESULT_VAR})
        message(STATUS "Performing Test ${RESULT_VAR} -- success")
    else()
        message(STATUS "Performing Test ${RESULT_VAR} -- failed to compile")
    endif()
endfunction()
