# ########################################################################
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

find_program(ROCSPARSE_MTX2CSR rocsparse_mtx2csr PATHS /opt/rocm/bin ${ROCM_PATH}/bin ./)

if(NOT CMAKE_MATRICES_DIR)
  set(CMAKE_MATRICES_DIR "./")
  message(WARNING "Unspecified CMAKE_MATRICES_DIR, the default value of CMAKE_MATRICES_DIR is set to './'")
endif()

# convert relative path to absolute
get_filename_component(CMAKE_MATRICES_DIR "${CMAKE_MATRICES_DIR}"
                       ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")

list(LENGTH TEST_MATRICES len)
math(EXPR len1 "${len} - 1")

foreach(i RANGE 0 ${len1})
    list(GET TEST_MATRICES ${i} m)

    string(REPLACE "/" ";" sep_m ${m})
    list(GET sep_m 0 dir)
    list(GET sep_m 1 mat)

    # Download test matrices if not already downloaded
    if(NOT EXISTS "${CMAKE_MATRICES_DIR}/${mat}.csr")
        message("-- Downloading and extracting test matrix ${m}.tar.gz")
        file(DOWNLOAD https://sparse.tamu.edu/MM/${m}.tar.gz ${CMAKE_MATRICES_DIR}/${mat}.tar.gz
            SHOW_PROGRESS INACTIVITY_TIMEOUT 20
            STATUS DL)

        list(GET DL 0 stat)
        list(GET DL 1 msg)

        if(NOT stat EQUAL 0)
            message("-- Timeout has been reached, trying mirror ...")
            # Try again using ufl links
            file(DOWNLOAD https://www.cise.ufl.edu/research/sparse/MM/${m}.tar.gz ${CMAKE_MATRICES_DIR}/${mat}.tar.gz
                SHOW_PROGRESS INACTIVITY_TIMEOUT 20
                STATUS DL)

            list(GET DL 0 stat)
            list(GET DL 1 msg)

            if(NOT stat EQUAL 0)
                message(FATAL_ERROR "${msg}")
            endif()
        endif()

        execute_process(COMMAND tar xf ${mat}.tar.gz
            RESULT_VARIABLE STATUS
            WORKING_DIRECTORY ${CMAKE_MATRICES_DIR})
        if(STATUS AND NOT STATUS EQUAL 0)
            message(FATAL_ERROR "uncompressing failed, aborting.")
        endif()

        file(RENAME ${CMAKE_MATRICES_DIR}/${mat}/${mat}.mtx ${CMAKE_MATRICES_DIR}/${mat}.mtx)

        execute_process(COMMAND ${ROCSPARSE_MTX2CSR} ${mat}.mtx ${mat}.csr
            RESULT_VARIABLE STATUS
            WORKING_DIRECTORY ${CMAKE_MATRICES_DIR})
        if(STATUS AND NOT STATUS EQUAL 0)
            message(FATAL_ERROR "${ROCSPARSE_MTX2CSR} failed, aborting.")
        else()
            message(STATUS "${mat} success.")
        endif()

        file(REMOVE_RECURSE ${CMAKE_MATRICES_DIR}/${mat}.tar.gz ${CMAKE_MATRICES_DIR}/${mat} ${CMAKE_MATRICES_DIR}/${mat}.mtx)
    endif()
endforeach()
