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

function(usage)
    message("Write an assembly file to <output> that embeds <symbol> as data with the"
                   " contents from <input>.")
    message("Usage: cmake -DSYMBOL=<symbol> -DINPUT_BIN=<input> -DOUTPUT=<output>")
endfunction()

foreach(PARAM_NAME IN ITEMS "SYMBOL" "INPUT_BIN" "OUTPUT")
    if(NOT DEFINED ${PARAM_NAME})
        usage()
        message(FATAL_ERROR "${PARAM_NAME} is required.\n"
                            "Please call the script as 'cmake -D${PARAM_NAME}=<value> ... -P <script>'")
    endif()
endforeach()

# The name of this file is written into the output
get_filename_component(GENERATOR_NAME ${CMAKE_CURRENT_LIST_FILE} NAME)

configure_file(${CMAKE_CURRENT_LIST_DIR}/asm_embed_symbol.S.in ${OUTPUT} @ONLY)
