# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

## List the objects for each gfx architecture
execute_process( COMMAND roc-obj-ls librccl.so
    RESULT_VARIABLE list_result
    OUTPUT_VARIABLE cmd_output
)

if(list_result EQUAL 0)
    ## Convert cmd output to list of lines
    string(REGEX REPLACE "\n$" "" cmd_output "${cmd_output}")
    string(REPLACE "\n" ";" cmd_output "${cmd_output}")

    ## Extract file paths for the selected gfx archs
    foreach(line ${cmd_output})
        if(line MATCHES "(gfx90a|gfx942|gfx950)")
            string(REGEX MATCH "\\file://(.*)" file_match ${line})
            if(file_match)
                list(APPEND file_paths ${file_match})
            endif()
        endif()
    endforeach()

    ## Extract objects from files
    foreach(file ${file_paths})
        execute_process(
          COMMAND roc-obj-extract ${file}
          RESULT_VARIABLE extraction_result
        )
        if(NOT extraction_result EQUAL 0)
          message(WARNING "Could not extract objects from ${file}")
        endif()
    endforeach()
else()
    ## We don't want to stop building unit-tests if this command fails.
    message(WARNING "Command failed with error code ${result}")
endif()