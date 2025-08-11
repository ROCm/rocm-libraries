# ##############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
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
#
# ##############################################################################

list(JOIN GPU_TARGETS "$<SEMICOLON>" TENSILE_GPU_TARGETS_SEMI_ESCAPED)
set(TENSILE_VERBOSITY
    ""
    CACHE STRING "Verbosity level for the TensileCreateLibrary command")
set(TENSILE_JOBS
    ""
    CACHE STRING "Number of jobs to run in parallel")
set(TENSILE_CODE_OBJECT_VERSION
    "V4"
    CACHE STRING "Path to directory containing logic files")
set(TENSILE_LOGIC_DIR
    "${tensile_SOURCE_DIR}/../../../projects/rocblas/library/src/blas3/Tensile/Logic/asm_full"
    CACHE STRING "Path to directory containing logic files")

set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--merge-files")
set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--separate-architectures")
set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--lazy-library-loading")
set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--no-short-file-names")
set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS}
                       "--architecture=\"${TENSILE_GPU_TARGETS_SEMI_ESCAPED}\"")
set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS}
                       "--cxx-compiler=${CMAKE_CXX_COMPILER}")
set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS}
                       "--code-object-version=${TENSILE_CODE_OBJECT_VERSION}")
if(TENSILE_VERBOSITY OR TENSILE_VERBOSITY STREQUAL "0")
  set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--verbose=${TENSILE_VERBOSITY}")
endif()
if(TENSILE_JOBS)
  set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--jobs=${TENSILE_JOBS}")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
# Use `PROJECT_BINARY_DIR` as the output directory so that the device libraries are built
# into the top-level Tensile binary directory, and are locatable by rocBLAS at runtime.
set(TENSILE_CREATE_LIBRARY_COMMAND
    Tensile/bin/TensileCreateLibrary ${TENSILE_BUILD_OPTS} ${TENSILE_LOGIC_DIR}
    ${PROJECT_BINARY_DIR} HIP)
set(output_stamp "${CMAKE_CURRENT_BINARY_DIR}/Tensile.stamp")
add_custom_command(
  COMMENT "Building device libraries..."
  COMMAND ${TENSILE_CREATE_LIBRARY_COMMAND}
  COMMAND ${CMAKE_COMMAND} -E touch "${output_stamp}"
  OUTPUT "${output_stamp}"
  WORKING_DIRECTORY "${TENSILE_SOURCE_TREE}/../..")

block(SCOPE_FOR VARIABLES)
list(JOIN TENSILE_CREATE_LIBRARY_COMMAND " " FORMATTED_TCL)
message(VERBOSE "Device lib build command: ${FORMATTED_TCL}")
endblock()

add_custom_target(tensile-device-libraries ALL DEPENDS "${output_stamp}")
